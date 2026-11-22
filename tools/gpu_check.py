#!/usr/bin/env python3
"""
Sanity-check the torch build: version, CUDA availability, and a real GPU matmul.

Run with the torch extra installed:

    uv sync --extra embed --extra torch
    uv run python tools/gpu_check.py

Two correctness anchors, because a bare bf16-vs-fp64 comparison is meaningless
here: a random dot product of length 4096 has a relative error around 10% in
bf16 purely from input rounding (|sum| ~ sqrt(n) while sum|terms| ~ n), which
looks like a bug and is not. So instead:

  1. ones @ ones  -- exact anchor, the accumulation path must return exactly n
  2. bf16 gemm vs fp32 gemm on the *same* bf16 values -- isolates accumulation
     error only (input rounding is identical on both sides), so a 1e-2 bound
     actually means something.

Exit code is 0 only if both anchors pass.
"""
import platform
import time

import torch


def main():
    print(f"python      {platform.python_version()} ({platform.machine()})")
    print(f"torch       {torch.__version__}")
    print(f"cuda build  {torch.version.cuda}")
    print(f"cudnn       {torch.backends.cudnn.version()}")
    print(f"available   {torch.cuda.is_available()}  devices={torch.cuda.device_count()}")

    if not torch.cuda.is_available():
        print("\nFAIL: no CUDA device visible")
        raise SystemExit(1)

    props = torch.cuda.get_device_properties(0)
    print(f"device      {props.name}  cc={props.major}.{props.minor}  "
          f"mem={props.total_memory / 1e9:.1f} GB  sm={props.multi_processor_count}")

    n = 4096
    m = 64
    torch.manual_seed(0)
    a = torch.randn(n, n, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(n, n, device="cuda", dtype=torch.bfloat16)

    # ---- anchor 1: exact
    ones = torch.ones(1, n, device="cuda", dtype=torch.bfloat16)
    total = (ones @ ones.T).item()
    ok1 = total == float(n)
    print(f"anchor 1    ones@ones = {total:.0f} (expect {n})  {'OK' if ok1 else 'FAIL'}")

    # ---- timing
    c = a @ b
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    reps = 20
    for _ in range(reps):
        c = a @ b
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / reps
    print(f"matmul      {n}x{n} bf16  {dt * 1e3:.1f} ms/iter  "
          f"{2 * n ** 3 / dt / 1e12:.2f} TFLOP/s")

    # ---- anchor 2: accumulation error only (same bf16 inputs on both sides)
    ref = a[:m, :].float() @ b[:, :m].float()   # 64 x 4096 @ 4096 x 64 -> 64 x 64
    got = c[:m, :m].float()
    err = (ref - got).abs().max().item()
    rel = err / ref.abs().max().item()
    ok2 = rel < 1e-2
    print(f"anchor 2    max_abs_err={err:.3e}  rel={rel:.3e}  "
          f"{'OK' if ok2 else 'MISMATCH'}  (vs fp32 gemm, same inputs)")

    free, total_mem = torch.cuda.mem_get_info()
    print(f"memory      allocated={torch.cuda.memory_allocated() / 1e9:.2f} GB  "
          f"free={free / 1e9:.1f}/{total_mem / 1e9:.1f} GB")

    if not (ok1 and ok2):
        raise SystemExit(1)
    print("\nGPU OK")


if __name__ == "__main__":
    main()
