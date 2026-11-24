#!/usr/bin/env python3
"""Serve an annotation packet over HTTP so annotators open a link instead of a file.

The standalone HTML page works, but it makes the *file* the unit of work: someone
fills 150 items, exports, and then the file has to travel back by mail. This server
removes that step — answers are POSTed as they are picked, appended to disk, and the
annotator downloads exactly what the server holds.

Two properties the whole evaluation rests on, and how this server keeps them:

**Independence.** Each annotator gets a URL with its own unguessable token. The page
contains only *that* annotator's answers; there is no route that lists another
annotator's answers, no shared state, and nothing on the page reveals which items the
other person has done. Both users can be told to open their link and nothing else.

**Nothing lives only in memory.** Every answer is appended to
`<out>/<prefix>_<who>.jsonl` and fsynced before the response is sent. A crash, a
restart or a closed laptop loses at most the answer in flight, and the browser keeps
any answer whose POST failed and retries it on the next change or reload.

No third-party dependency: stdlib http.server only, because the project's install is
numpy + stdlib and an annotation tool is not a reason to add a web framework. It also
serves nothing but the routes below — no static file handler, no directory listing, so
there is no path to the corpus or the ground-truth labels through this port.

    uv run python tools/annotate_server.py --packet annotation --out annotations \\
        --annotator A --annotator B --port 8787

Prints one URL per annotator plus an admin URL showing progress (counts only, never
labels). Tokens are random per run; pass --token A=... to pin them.
"""
from __future__ import annotations

import argparse
import json
import os
import secrets
import socket
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import annotation_page as ap_page  # noqa: E402

LOCK = threading.Lock()


def read_jsonl(path):
    out = []
    if not os.path.exists(path):
        return out
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


class Store:
    """One jsonl per annotator, append-only.

    Append-only rather than rewrite-everything: an annotation session is 2-3 hours of
    human attention, and a bug in an in-memory rebuild must not be able to destroy
    earlier answers. The last line for an id wins when reading.
    """

    def __init__(self, out_dir, prefix, who):
        os.makedirs(out_dir, exist_ok=True)
        self.path = os.path.join(out_dir, f"{prefix}_{who}.jsonl")
        self.who = who

    def answers(self):
        merged = {}
        for o in read_jsonl(self.path):
            merged[o["id"]] = {"primary": o.get("primary"), "note": o.get("note") or ""}
        return merged

    def add(self, entries):
        lines = []
        for e in entries:
            primary = e.get("primary") or None
            lines.append(json.dumps({"id": e["id"], "primary": primary,
                                     "labels": [primary] if primary and primary != "UNKNOWN"
                                     else [],
                                     "note": e.get("note") or "",
                                     "ts": time.strftime("%Y-%m-%dT%H:%M:%S%z")},
                                    ensure_ascii=False))
        with LOCK:
            with open(self.path, "a", encoding="utf-8") as f:
                f.write("\n".join(lines) + "\n")
                f.flush()
                os.fsync(f.fileno())

    def export(self, item_ids):
        """Every item, blanks as null — the shape tools/kappa.py reads."""
        got = self.answers()
        out = []
        for i in item_ids:
            a = got.get(i) or {}
            primary = a.get("primary") or None
            out.append(json.dumps({"id": i, "primary": primary,
                                   "labels": [primary] if primary and primary != "UNKNOWN"
                                   else [],
                                   "note": a.get("note") or ""}, ensure_ascii=False))
        return "\n".join(out) + "\n"

    def counts(self, item_ids):
        got = self.answers()
        labelled = sum(1 for i in item_ids if (got.get(i) or {}).get("primary"))
        unknown = sum(1 for i in item_ids
                      if (got.get(i) or {}).get("primary") == "UNKNOWN")
        notes = sum(1 for i in item_ids if ((got.get(i) or {}).get("note") or "").strip())
        return {"labelled": labelled, "unknown": unknown, "notes": notes,
                "total": len(item_ids)}


class App:
    def __init__(self, args):
        self.args = args
        with open(os.path.join(args.packet, "packet.json"), encoding="utf-8") as f:
            self.meta = json.load(f)
        if args.set not in self.meta:
            sys.exit(f"[fatal] --set must be one of {sorted(self.meta)}")
        self.set = self.meta[args.set]
        with open(os.path.join(args.packet, self.set["items"]), encoding="utf-8") as f:
            self.items = [json.loads(line) for line in f if line.strip()]
        with open(os.path.join(args.packet, "codebook_cards.json"), encoding="utf-8") as f:
            self.cards = json.load(f)
        self.ids = [o["id"] for o in self.items]

        pinned = {}
        for spec in args.token or []:
            who, _, tok = spec.partition("=")
            pinned[who] = tok
        self.tokens = {}
        for who in args.annotator:
            self.tokens[pinned.get(who) or secrets.token_urlsafe(9)] = who
        if len(set(self.tokens.values())) != len(self.tokens):
            sys.exit("[fatal] two annotators resolved to the same token")
        self.admin_token = args.admin_token or secrets.token_urlsafe(12)
        self.stores = {who: Store(args.out, self.set["prefix"], who)
                       for who in args.annotator}

    def base(self, host):
        return f"http://{host}:{self.args.port}"

    def page(self, who):
        token = [t for t, w in self.tokens.items() if w == who][0]
        return ap_page.render(
            self.set["set_zh"], [{"id": o["id"], "text": o["text"]} for o in self.items],
            self.cards, None, prefix=self.set["prefix"],
            set_name_en=self.set["set_en"], mode="server", who=who,
            answers=self.stores[who].answers(),
            post_url=f"/{token}/answer", export_url=f"/{token}/export",
            return_html=True)


def make_handler(app):
    class Handler(BaseHTTPRequestHandler):
        server_version = "annotate-server"
        protocol_version = "HTTP/1.1"

        def log_message(self, format, *args):  # noqa: A002 - stdlib signature
            sys.stderr.write(f"  {self.address_string()} {format % args}\n")

        def _send(self, code, body, ctype="text/html; charset=utf-8", extra=None):
            raw = body.encode("utf-8") if isinstance(body, str) else body
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(raw)))
            self.send_header("Cache-Control", "no-store")
            for k, v in (extra or {}).items():
                self.send_header(k, v)
            self.end_headers()
            self.wfile.write(raw)

        def _404(self):
            self._send(404, "<h1>404</h1><p>check the link you were given.</p>")

        def do_GET(self):  # noqa: N802 - stdlib naming
            path = self.path.split("?")[0].rstrip("/") or "/"
            if path == "/favicon.ico":
                self._send(204, b"")
                return
            if path == "/admin":
                self._admin()
                return
            parts = path.strip("/").split("/")
            if len(parts) == 1:
                who = app.tokens.get(parts[0])
                if who:
                    self._send(200, app.page(who))
                    return
            if len(parts) == 2 and parts[1] == "export":
                who = app.tokens.get(parts[0])
                if who:
                    name = f'{app.set["prefix"]}_{who}.jsonl'
                    self._send(200, app.stores[who].export(app.ids),
                               "application/x-ndjson",
                               {"Content-Disposition": f'attachment; filename="{name}"'})
                    return
            self._404()

        def do_POST(self):  # noqa: N802
            path = self.path.split("?")[0].rstrip("/")
            parts = path.strip("/").split("/")
            if len(parts) != 2 or parts[1] != "answer":
                self._404()
                return
            who = app.tokens.get(parts[0])
            if not who:
                self._404()
                return
            try:
                n = int(self.headers.get("Content-Length") or 0)
                body = json.loads(self.rfile.read(n) or b"{}")
                entries = body.get("answers") or []
                valid = [e for e in entries if e.get("id") in set(app.ids)]
                if not valid:
                    self._send(400, json.dumps({"error": "no valid answers"}),
                               "application/json")
                    return
                app.stores[who].add(valid)
                c = app.stores[who].counts(app.ids)
                self._send(200, json.dumps(c), "application/json")
            except Exception as e:  # noqa: BLE001 - report, never crash the server
                self._send(500, json.dumps({"error": str(e)}), "application/json")

        def _admin(self):
            q = self.path.split("?")[1] if "?" in self.path else ""
            tok = ""
            for pair in q.split("&"):
                if pair.startswith("token="):
                    tok = pair[6:]
            if not secrets.compare_digest(tok, app.admin_token):
                self._404()
                return
            rows = []
            for who in sorted(app.tokens.values()):
                c = app.stores[who].counts(app.ids)
                pct = f"{100 * c['labelled'] / c['total']:.0f}%"
                unk = f"{100 * c['unknown'] / c['labelled']:.0f}%" if c["labelled"] else "-"
                rows.append(f"<tr><td>{who}</td><td>{c['labelled']}/{c['total']}</td>"
                            f"<td>{pct}</td><td>{c['unknown']}</td><td>{unk}</td>"
                            f"<td>{c['notes']}</td>"
                            f"<td><code>{app.stores[who].path}</code></td></tr>")
            html = ("<!DOCTYPE html><meta charset='utf-8'><title>annotation progress</title>"
                    "<style>body{font:14px system-ui;background:#0f1115;color:#e6e8ee;"
                    "padding:24px}table{border-collapse:collapse}"
                    "td,th{border:1px solid #2a2f3a;padding:6px 10px;text-align:left}"
                    "th{color:#9aa3b2;font-weight:600}</style>"
                    f"<h1 style='font-size:16px'>进度 · {app.set['set_zh']}</h1>"
                    "<p>只显示进度，不含任何标签内容——管理员也不该看到别人的答案，"
                    "否则分歧会被“讨论”掉。</p>"
                    "<table><tr><th>标注员</th><th>已标</th><th>进度</th><th>UNKNOWN</th>"
                    "<th>UNKNOWN 占比</th><th>备注数</th><th>落盘文件</th></tr>"
                    + "".join(rows) + "</table>")
            self._send(200, html)

    return Handler


def lan_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("192.0.2.1", 9))  # no packet is sent; this just picks a route
        return s.getsockname()[0]
    except OSError:
        return "127.0.0.1"
    finally:
        s.close()


def main():
    ap = argparse.ArgumentParser(
        description="Serve an annotation packet over HTTP.")
    ap.add_argument("--packet", default="annotation",
                    help="packet directory written by tools/make_annotation_packet.py")
    ap.add_argument("--out", default="annotations",
                    help="where the per-annotator jsonl files are appended")
    ap.add_argument("--set", default="main", choices=["main", "pilot"])
    ap.add_argument("--annotator", action="append", default=[],
                    help="annotator name; repeat for each person (e.g. --annotator A)")
    ap.add_argument("--token", action="append", default=[],
                    help="pin a token: --token A=secret (else one is generated)")
    ap.add_argument("--admin-token", default="", help="token for /admin?token=...")
    ap.add_argument("--port", type=int, default=8787)
    ap.add_argument("--host", default="0.0.0.0",
                    help="0.0.0.0 exposes it on the LAN; 127.0.0.1 keeps it local")
    args = ap.parse_args()
    if not args.annotator:
        sys.exit("[fatal] pass at least one --annotator (e.g. --annotator A --annotator B)")

    app = App(args)
    httpd = ThreadingHTTPServer((args.host, args.port), make_handler(app))
    host = lan_ip() if args.host in ("0.0.0.0", "::") else args.host
    print(f"serving set {args.set!r}: {len(app.items)} items, "
          f"{len(app.cards)} classes")
    print(f"answers -> {os.path.abspath(args.out)}/"
          f"{app.set['prefix']}_<annotator>.jsonl")
    print()
    for tok, who in sorted(app.tokens.items(), key=lambda kv: kv[1]):
        print(f"  annotator {who}:  {app.base(host)}/{tok}/")
    print(f"  admin (counts only): {app.base(host)}/admin?token={app.admin_token}")
    print("\n  one link per person; the page only ever holds that person's answers.")
    print("  Ctrl-C to stop — answers already on disk stay there.\n")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")
    finally:
        httpd.server_close()


if __name__ == "__main__":
    main()
