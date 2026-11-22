#!/usr/bin/env python3
"""Decide whether a customer utterance carries complaint content or is procedure.

Naming the clusters on the 500-dialogue run showed where the signal was going:
four of ten clusters were not complaints at all -- credential exchange (usernames,
emails, order ids), bare person names, acknowledgements and thanks. Together they
held 1,201 of 3,148 sentences (38%). The boilerplate DF filter cannot see them,
because every customer's name and account id is unique: nothing repeats often
enough to look templated.

Three families of rule, applied after PII masking so the placeholders are visible:

  credential  the utterance is field labels plus placeholder values
              ("用户名：<NAME>，电子邮箱：<EMAIL>，订单号：<N>")
  name        a bare person name ("诺曼·布沙尔", "Crystal Minh")
  ack         acknowledgement, thanks, or a closing line ("好的，就这些了", "谢谢")

Deliberately conservative. A short *complaint* ("退货", "太慢了") must survive, so a
bare two-to-five character utterance only counts as a name when it is not made of
complaint-bearing words, and the rule can be turned off. Everything dropped is
exported so the decision is auditable rather than silent.
"""
import re

PLACEHOLDER = re.compile(r"<[A-Z][A-Z_]*>")
IDISH = re.compile(r"[A-Za-z]{1,16}\d{2,}\b|\b\d{3,}\b|\b\d+\.\d+\b")
FILLER = re.compile(
    r"我的|我们是|我|是|为|叫做|叫|地址|号码|如下|以下|分别|以及|还有|请|帮忙|查一下"
)
FIELD = re.compile(
    r"用户名|用户名称|用户 ?id|账号|账户|订单号|订单 ?id|订单编号|订单|邮编号?|邮编|"
    r"电子邮箱|邮箱|邮件地址|电话|手机号|联系方式|密码|卡号|地址|姓名|名字|名是|"
    r"黄金会员|会员|高级订阅|订阅|"
    r"\b(?:user ?name|account|order|email|e-mail|zip|phone|password|member)\b",
    re.I,
)

# Words that mean the utterance is about a problem. A short string containing any of
# these is kept even if it is only a couple of characters long.
CUE = re.compile(
    r"退|换|坏|破|损|错|慢|迟到|延误|缺|假|骗|改|修|投诉|收费|扣|刷|退款|取消|"
    r"没|未|不能|不行|无法|不|少|多|差|等|丢|漏|过期|失效|报错|崩溃|卡住|登录|"
    r"忘|封|冻结|失败|问题|故障|bug|"
    r"\b(?:refund|return|broken|damaged|late|delay|wrong|missing|charged|charge|"
    r"cancel|error|failed|fail|can't|cannot|not|never|issue|problem|expired|"
    r"locked|password|login)\b",
    re.I,
)

ACK_WORDS = {
    # acknowledgements / confirmations
    "好", "好的", "好吧", "好呀", "好啊", "行", "可以", "可以了", "没问题", "没事",
    "嗯", "嗯嗯", "对", "对的", "是的", "是", "没错", "知道了", "明白了", "懂了",
    "就这样", "就这些", "就这些了", "没有", "没有了", "不用了", "不需要了",
    "那好吧", "那好吧就这些", "太好了", "那太好了", "听起来不错", "可以没问题",
    # thanks / closings
    "谢谢", "谢谢你", "谢谢您", "谢谢你的帮助", "谢谢您的帮助", "多谢", "非常感谢",
    "太感谢了", "感谢", "感谢你", "感谢您", "辛苦了", "再见", "拜拜",
    # english
    "ok", "okay", "sure", "yes", "yeah", "yep", "no", "nope", "yup", "alright",
    "thanks", "thank you", "thanks a lot", "thx", "great", "perfect", "got it",
    "sounds good", "that's all", "that is all", "nothing else", "no thanks",
    "no thank you", "cool", "nice", "fine", "i see", "understood",
}

ACK_TOKENS = sorted(
    {
        # acknowledgements
        "好的", "好", "好吧", "好呀", "好啊", "行", "可以", "可以了", "没问题", "没事",
        "嗯", "对", "对的", "是的", "是", "没错", "知道了", "明白了", "懂了", "懂",
        "就这样", "这样", "就这些", "这些", "没有了", "没有", "不用了", "不需要了",
        "那好吧", "那就这样", "太好了", "听起来不错", "不错", "清楚了", "了解了",
        # thanks and closings
        "非常感谢", "太感谢了", "谢谢你的帮助", "谢谢您的帮助", "谢谢你", "谢谢您",
        "谢谢", "多谢", "感谢你", "感谢您", "感谢", "辛苦了", "再见", "拜拜",
        "祝您愉快", "祝你愉快", "祝您开心", "祝你开心", "祝您有个愉快的一天",
        "祝你有美好的一天", "祝您生活愉快",
        # particles and connectives that carry no content
        "吧", "呢", "呀", "啊", "了", "的", "就", "那", "这",
    },
    key=len,
    reverse=True,
)

EN_ACK_TOKENS = sorted(
    {
        "ok", "okay", "sure", "yes", "yeah", "yep", "yup", "no", "nope", "alright",
        "thanks", "thank you", "thanks a lot", "thx", "great", "perfect", "cool",
        "nice", "fine", "got it", "i see", "understood", "sounds good",
        "that's all", "that is all", "nothing else", "works for me", "that works",
        "have a nice day", "have a good one", "goodbye", "bye", "please",
    },
    key=len,
    reverse=True,
)


def _strip_tokens(text, tokens, casefold=False):
    """Remove a closed vocabulary, longest token first, and report what is left."""
    t = text.strip()
    if casefold:
        t = t.lower()
    changed = True
    while changed and t:
        changed = False
        for tok in tokens:
            if t.startswith(tok):
                t = t[len(tok):].lstrip(" ,.!?;:。，、！？；：")
                changed = True
                break
    return t


PUNCT = re.compile(r"[\s:：，,。.、；;！!？?\-—_/\\|~\"'“”‘’()（）\\[\\]【】]+")
NAME_CJK = re.compile(r"^[\u4e00-\u9fff]{2,6}$")
FUNC = set("我的你他她它您我们你们了是很有要在想说问就都也还过么什这那个很太别不没未和与或所以头紧")
NAME_CJK_DOT = re.compile(r"^[\u4e00-\u9fff]{1,6}(?:·[\u4e00-\u9fff]{1,6})+$")
NAME_LATIN = re.compile(r"^(?:[A-Z][a-z]{1,14}[\s.]*){1,3}$")


def has_cue(text):
    return bool(CUE.search(text))


def is_procedural(text, drop_names=True):
    """Return (True, reason) when the utterance is procedure, not complaint."""
    t = (text or "").strip()
    if not t:
        return True, "empty"

    # 1) credential / PII exchange: field labels + placeholder values.
    # Anything carrying a complaint cue is kept: "我无法访问我的账户" is a complaint
    # that happens to contain the word 账户, and an earlier version dropped it.
    if not has_cue(t):
        fields = FIELD.findall(t)
        placeholders = PLACEHOLDER.findall(t)
        ids = re.findall(IDISH, t)
        # Several identifying items in one utterance is a data exchange no matter
        # how much wrapping text there is ("我的名字是 X，我的账户ID是 Y"). A single
        # item only counts when almost nothing else is left once it is removed.
        if len(fields) + len(placeholders) + len(ids) >= 2:
            return True, "credential"
    if len(FIELD.findall(t)) >= 1 and not has_cue(t):
        core = FIELD.sub("", t)
        core = PLACEHOLDER.sub("", core)
        core = IDISH.sub("", core)
        core = FILLER.sub("", core)
        core = PUNCT.sub("", core)
        if len(core) <= 4:
            return True, "credential"

    # 2) acknowledgement / thanks / closing: nothing survives removing the
    # closed vocabulary of ack words, thanks, particles and well-wishes
    bare = PUNCT.sub("", t)
    if bare in ACK_WORDS or t.strip().lower() in ACK_WORDS:
        return True, "ack"
    if len(bare) <= 20:
        if not _strip_tokens(t, ACK_TOKENS):
            return True, "ack"
        if not _strip_tokens(t, EN_ACK_TOKENS, casefold=True):
            return True, "ack"

    # 3) bare person name. Two-to-four CJK characters with no function word in them,
    # or a title-cased Latin name. "退货" survives because 退 is a complaint cue, and
    # "所以手头很紧" survives because 很 is a function word.
    if drop_names and not has_cue(t):
        t2 = t.strip()
        if NAME_CJK_DOT.match(t2):
            return True, "name"
        if NAME_CJK.match(t2) and 2 <= len(t2) <= 4 and not (set(t2) & FUNC):
            return True, "name"
        if NAME_LATIN.match(t2) and len(t2) <= 14:
            return True, "name"

    return False, ""


def audit(sentences, drop_names=True):
    """Count what each rule removes; used for the printed stats and the export."""
    from collections import Counter

    reasons = Counter()
    kept, dropped = [], []
    for s in sentences:
        flag, why = is_procedural(s, drop_names=drop_names)
        if flag:
            reasons[why] += 1
            dropped.append({"text": s, "reason": why})
        else:
            kept.append(s)
    return kept, dropped, reasons
