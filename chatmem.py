"""聊天记忆核心：只保留 L 区（聊天）的「分层 → 压缩 → 归档」+ L5。

和完整版 TMC 的区别：没有 D/E/F/G 四个区、不注册工具、不依赖 function calling。
所有事都是插件自己做的，模型只要正常聊天就行。

    l1.md      最近对话（短，满了压进 L2）
    l2.md      压缩摘要（再满就归档）
    l3.jsonl   归档索引（summary + tags，注入时列最近的几条）
    l4.jsonl   归档原文（永久）
    l5.json    对对方的理解（注入时给模型校准态度）
"""

from __future__ import annotations

import json
import math
import os
import re
import time
from pathlib import Path

DEFAULT_UTC_OFFSET = 8

L1_MAX = 1200
L2_MAX = 2000

PROMPT_L1 = (
    "你在整理一段聊天记录。保留称呼、约定、情绪变化、对方提到的事实和偏好；"
    "用第三人称短句，不要评价，不要编造。压到 {target} 字以内，只输出结果。"
)

PROMPT_INDEX = (
    "给下面这段聊天记录写一条索引：60~120 字，说清发生了什么、和对方约定了什么。"
    '另外给 3~6 个标签。只输出 JSON：{"summary":"...","tags":["..."]}，不要代码块。'
)

PROMPT_L5 = (
    "你在维护「她对对方的核心记忆」。第一人称，80~150 字，只写态度、理解、相处方式，"
    "不要复述事件流水账。根据下面的新记录更新，输出 JSON："
    '{"summary":"..."}，不要代码块。'
)

PROMPT_FACTS = (
    "你在为「她」整理关于聊天对象（TA）的事实。\n"
    "身份说明：「她」是机器人自己；TA 是正在和她聊天的人。\n"
    "规则：\n"
    "1. 主语一律写 TA，不要写「对方」、不要写「她」。\n"
    "2. 不要把她自己的名字、群昵称、自称当成 TA 的称呼。\n"
    "3. 每条是一个陈述句，第三人称，20~40 字。\n"
    "4. 不要引用对话原话，不要有语气词、破折号、括号里的动作描写。\n"
    "5. 不要抽取插件、系统、命令、费用、文件、模型这类技术内容。\n"
    "6. type 只能是：约定 / 偏好 / 称呼 / 雷点 / 事件。\n"
    "7. 没有值得长期记住的就输出空数组。\n"
    '输出 JSON：{"facts":[{"type":"约定","text":"..."}]}，最多 8 条，不要代码块。'
)

FACT_TYPES = ("约定", "偏好", "称呼", "雷点", "事件")

PROMPT_CONSOLIDATE = (
    "下面是一份事实列表，里面有语义重复的条目。合并规则：\n"
    "1. 同一个对象的多个细节，合并成一条（橘子、白饭、减肥、超市这类各留一条）。\n"
    "2. 同一个约定只保留最新的一条，别保留三份。\n"
    "3. 同一类偏好合成一句概括，不要拆成好几条。\n"
    "4. 每条不超过 40 字，不要新增内容，不要改变含义。\n"
    "type 只能是：约定 / 偏好 / 称呼 / 雷点 / 事件。\n"
    '输出 JSON：{"facts":[{"type":"...","text":"..."}]}，最多 {limit} 条，不要代码块。'
)

FACT_STOPWORDS = {
    "机器人", "自己", "时候", "今天", "明天", "晚上", "最近", "已经", "一个",
    "一起", "还是", "可以", "觉得", "现在", "之前", "然后", "所以",
}

FACT_PRIORITY = {"雷点": 0, "称呼": 1, "偏好": 2, "约定": 3, "事件": 4, "日期": 4, "其他": 5}


def stamp(utc_offset: int = DEFAULT_UTC_OFFSET) -> str:
    return time.strftime("%Y-%m-%d %H:%M", time.gmtime(time.time() + utc_offset * 3600))


class ChatStore:
    def __init__(self, base_dir: str | os.PathLike):
        self.root = Path(base_dir)
        self.files = {
            "l1": self.root / "l1.md",
            "l2": self.root / "l2.md",
            "l3": self.root / "l3.jsonl",
            "l4": self.root / "l4.jsonl",
            "l5": self.root / "l5.json",
            "facts": self.root / "facts.json",
            "vectors": self.root / "vectors.json",
            "state": self.root / "state.json",
            "log": self.root / "tmc.log",
        }

    def ensure(self) -> None:
        try:
            self.root.mkdir(parents=True, exist_ok=True)
        except OSError:
            pass

    def read(self, name: str) -> str:
        try:
            return self.files[name].read_text(encoding="utf-8")
        except (OSError, KeyError):
            return ""

    def write(self, name: str, text: str) -> bool:
        try:
            self.ensure()
            self.files[name].write_text(text, encoding="utf-8")
            return True
        except OSError:
            return False

    def write_atomic(self, name: str, text: str) -> bool:
        """先写临时文件再替换，断电/强杀时不会留下半个 JSON。"""
        try:
            self.ensure()
            target = self.files[name]
            tmp = target.with_suffix(target.suffix + ".tmp")
            tmp.write_text(text, encoding="utf-8")
            os.replace(tmp, target)
            return True
        except OSError:
            return self.write(name, text)

    def append(self, name: str, text: str) -> bool:
        try:
            self.ensure()
            with self.files[name].open("a", encoding="utf-8") as fh:
                fh.write(text)
            return True
        except OSError:
            return False

    def size(self, name: str) -> int:
        return len(self.read(name))

    def trim_tail(self, name: str, cap: int) -> bool:
        """压缩失败时的兜底：只保留末尾 cap 字，按行边界切。"""
        text = self.read(name).strip()
        if not cap or len(text) <= cap:
            return False
        cut = text[-cap:]
        newline = cut.find("\n")
        if 0 <= newline < 200:
            cut = cut[newline + 1 :]
        return self.write(name, cut.strip() + "\n")

    def read_jsonl(self, name: str) -> list[dict]:
        out = []
        for line in self.read(name).split("\n"):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(item, dict):
                out.append(item)
        return out

    def append_jsonl(self, name: str, obj: dict) -> bool:
        return self.append(name, json.dumps(obj, ensure_ascii=False) + "\n")

    def l3(self) -> list[dict]:
        return self.read_jsonl("l3")

    def l4(self) -> list[dict]:
        return self.read_jsonl("l4")

    def text_of(self, p) -> str:
        for page in self.read_jsonl("l4"):
            if str(page.get("p")) == str(p):
                return str(page.get("text") or "")
        return ""

    def record_page(self, text: str, summary: str, tags, src: str = "chat") -> int:
        # 上一次归档写完 l4/l3 但没来得及清空 l2 就崩了，会重复写同一页；这里挡一下
        pages = self.read_jsonl("l4")
        if pages and str(pages[-1].get("text") or "") == str(text):
            return int(pages[-1].get("p") or 0)
        p = max([int(page.get("p") or 0) for page in pages] or [0]) + 1
        ts = stamp()
        self.append_jsonl("l4", {"p": p, "ts": ts, "text": text})
        self.append_jsonl("l3", {"p": p, "ts": ts, "summary": summary, "tags": list(tags or []), "src": src})
        return p

    def l5(self) -> dict:
        try:
            data = json.loads(self.read("l5"))
        except (json.JSONDecodeError, ValueError):
            data = None
        if isinstance(data, dict) and isinstance(data.get("summary"), str):
            return {"summary": data["summary"], "updated": str(data.get("updated") or "")}
        return {"summary": "", "updated": ""}

    def write_l5(self, summary: str) -> None:
        self.write_atomic("l5", json.dumps({"summary": summary, "updated": stamp()}, ensure_ascii=False, indent=2))

    def facts(self) -> list[dict]:
        try:
            data = json.loads(self.read("facts"))
        except (json.JSONDecodeError, ValueError):
            data = None
        items = data if isinstance(data, list) else (data or {}).get("facts") if isinstance(data, dict) else None
        if not isinstance(items, list):
            return []
        return [
            item
            for item in items
            if isinstance(item, dict) and str(item.get("text") or "").strip()
        ]

    def write_facts(self, facts: list[dict]) -> None:
        self.write_atomic("facts", json.dumps(facts, ensure_ascii=False, indent=2))

    def log(self, msg: str) -> None:
        try:
            self.ensure()
            with self.files["log"].open("a", encoding="utf-8") as fh:
                fh.write(f"{stamp()} {msg}\n")
        except OSError:
            pass

    def has_data(self) -> bool:
        return any(self.size(name) for name in ("l1", "l2", "l3", "l4", "l5"))

    def archive_raw(self, text: str) -> bool:
        """压缩/裁剪之前，把逐字原文追加到 raw/日期.md，永久保存，不注入、不花 token。"""
        body = str(text or "").strip()
        if not body:
            return False
        try:
            day = time.strftime("%Y-%m-%d", time.gmtime(time.time() + DEFAULT_UTC_OFFSET * 3600))
            folder = self.root / "raw"
            folder.mkdir(parents=True, exist_ok=True)
            path = folder / f"{day}.md"
            header = "" if path.exists() else f"# {day}\n\n"
            with path.open("a", encoding="utf-8") as fh:
                fh.write(header + body + "\n\n")
            return True
        except OSError:
            return False


# ---------- 压缩 ----------

_JSON_BLOCK = re.compile(r"\{.*\}", re.S)


def _clean(text: str, cap: int) -> str:
    body = re.sub(r"\s+", " ", str(text or "")).strip()
    if len(body) <= cap:
        return body
    # 在句子边界截断，别把半句话写进记忆
    window = body[:cap]
    for sep in ("。", "！", "？", "；", "，", ".", "!", "?"):
        index = window.rfind(sep)
        if index >= int(cap * 0.6):
            return window[: index + 1]
    return window


async def _call(llm, instruction: str, text: str) -> str:
    try:
        out = await llm(instruction, text)
    except Exception:
        return ""
    return str(out or "").strip()


async def compress(store: ChatStore, llm, source: str, target: int) -> str:
    """把 source 层压到 target 字以内，写进下一层。"""
    text = store.read(source).strip()
    if not text:
        return ""
    instruction = PROMPT_L1.format(target=target)
    pieces = []
    for start in range(0, len(text), 8000):
        chunk = text[start : start + 8000]
        out = await _call(llm, instruction, chunk)
        if out:
            pieces.append(out)
    merged = "\n".join(pieces).strip()
    if len(merged) > target:
        # 先让模型再压一次，压不下来才按句子边界截
        again = await _call(llm, instruction, merged)
        if again:
            merged = again.strip()
    return _clean(merged, target)


async def make_index(llm, text: str) -> tuple[str, list[str]]:
    """给一页归档生成索引和标签。"""
    raw = await _call(llm, PROMPT_INDEX, str(text)[:12000])
    summary, tags = "", []
    match = _JSON_BLOCK.search(raw)
    if match:
        try:
            data = json.loads(match.group(0))
            summary = _clean(data.get("summary"), 300)
            tags = [str(tag).strip() for tag in (data.get("tags") or []) if str(tag).strip()][:6]
        except (json.JSONDecodeError, AttributeError):
            pass
    return summary, tags


async def seal_l2(store: ChatStore, llm) -> str:
    """L2 归档：索引 + 原文进档案，清空 L2，重算 L5。"""
    text = store.read("l2").strip()
    if not text:
        return ""
    summary, tags = await make_index(llm, text)
    summary = summary or _clean(text, 200)
    store.record_page(text, summary, tags)
    store.write("l2", "")
    await update_l5(store, llm, text)
    await update_facts(store, llm, text)
    return summary


def _fact_tokens(text: str) -> set[str]:
    return {token for token in tokenize(text) if len(token) >= 2}


def _normalize_fact_type(value) -> str:
    text = str(value or "")
    for kind in FACT_TYPES:
        if kind in text:
            return kind
    if "日期" in text or "重要" in text:
        return "事件"
    return "其他"


def _content_tokens(text: str) -> set[str]:
    return {token for token in tokenize(text) if len(token) >= 2} - FACT_STOPWORDS


def dedupe_facts(facts: list[dict], min_shared: int = 2) -> list[dict]:
    """同类型、共同实词 ≥2 个的事实视为重复，保留字数更多的那条。"""
    kept: list[dict] = []
    for fact in sorted(facts, key=lambda item: len(str(item.get("text") or "")), reverse=True):
        tokens = _content_tokens(str(fact.get("text") or ""))
        duplicate = False
        for other in kept:
            if other.get("type") != fact.get("type"):
                continue
            if len(tokens & _content_tokens(str(other.get("text") or ""))) >= min_shared:
                duplicate = True
                break
        if not duplicate:
            kept.append(fact)
    return kept


def write_l3(store: ChatStore, pages: list[dict]) -> None:
    store.write_atomic("l3", "".join(json.dumps(page, ensure_ascii=False) + "\n" for page in pages))


def needs_index(page: dict) -> bool:
    """导入留下的页、超长索引、以省略号结尾的索引，都算待重建。"""
    if str(page.get("src") or "") == "import":
        return True
    if str(page.get("src") or "") == "index":
        return False
    summary = str(page.get("summary") or "")
    return len(summary) > 150 or summary.endswith("…")


async def rebuild_indexes(store: ChatStore, llm, limit: int = 30) -> tuple[int, int]:
    """把导入时留下的占位索引逐页重建成正式摘要，并清掉对应向量等后台重算。"""
    pages = store.l3()
    targets = [page for page in pages if needs_index(page)][:limit]
    if not targets:
        return 0, 0
    vectors = load_vectors(store)
    done = 0
    for page in targets:
        text = store.text_of(page.get("p"))
        if not text.strip():
            continue
        summary, tags = await make_index(llm, text)
        if not summary:
            continue
        page["summary"] = summary
        page["tags"] = tags
        page["src"] = "index"
        vectors.pop(str(page.get("p")), None)
        done += 1
    write_l3(store, pages)
    save_vectors(store, vectors)
    return done, len(targets)


async def update_facts(store: ChatStore, llm, text: str, limit: int | None = None,
                       merge: bool = True) -> list[dict]:
    """从一段记录里抽事实，合并进 facts.json。"""
    limit = limit or int(getattr(store, "facts_limit", 24) or 24)
    block = [str(word).strip().lower() for word in (getattr(store, "facts_block", None) or []) if str(word).strip()]
    caller = getattr(store, "facts_llm", None) or llm
    known = [str(fact.get("text") or "") for fact in store.facts() if fact.get("type") == "称呼"]
    hint = f"\n已知称呼：{'；'.join(known[:4])}" if known else ""
    raw = await _call(caller, PROMPT_FACTS, hint + "\n\n" + str(text)[:8000])
    match = _JSON_BLOCK.search(raw)
    if not match:
        return store.facts()
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return store.facts()
    incoming = data.get("facts")
    if not isinstance(incoming, list):
        return store.facts()

    facts = store.facts()
    for item in incoming[:8]:
        if not isinstance(item, dict):
            continue
        body = _clean(item.get("text"), 80).strip("“”\"'「」『』 ")
        body = body.replace("对方", "TA")
        if len(body) < 4:
            continue
        if re.fullmatch(r"[\d\s年月日:\-—.]+", body):
            continue
        lowered = body.lower()
        if block and any(word in lowered for word in block):
            continue
        kind = _normalize_fact_type(item.get("type"))
        bot_names = [str(name).strip() for name in (getattr(store, "bot_names", None) or []) if str(name).strip()]
        if kind == "称呼" and bot_names and any(name in body for name in bot_names):
            # 把机器人自己的名字当成 TA 的称呼，丢掉
            continue
        tokens = _fact_tokens(body)
        merged = False
        for fact in facts:
            if fact.get("type") != kind:
                continue
            other = _fact_tokens(str(fact.get("text") or ""))
            old_text = str(fact.get("text") or "")
            if tokens and other:
                overlap = len(tokens & other) / len(tokens | other)
                contained = old_text and (body in old_text or old_text in body)
                if overlap >= 0.5 or contained:
                    # 保留信息更全的那条
                    if len(body) >= len(old_text):
                        fact["text"] = body
                    fact["ts"] = stamp()
                    merged = True
                    break
        if not merged:
            facts.append({"type": kind, "text": body, "ts": stamp()})

    facts = dedupe_facts(facts)
    if len(facts) > limit:
        facts.sort(key=lambda item: (FACT_PRIORITY.get(str(item.get("type")), 5), str(item.get("ts") or "")))
        facts = facts[-limit:]
    store.write_facts(facts)
    if merge and len(facts) >= 8:
        facts = await consolidate_facts(store, caller, limit)
    return facts


async def consolidate_facts(store: ChatStore, llm, limit: int = 24) -> list[dict]:
    """事实多了以后做一次合并，去掉语义重复。失败就保持原样。"""
    facts = store.facts()
    if len(facts) < 8:
        return facts
    payload = json.dumps(
        [{"type": item.get("type"), "text": item.get("text")} for item in facts],
        ensure_ascii=False,
    )
    raw = await _call(llm, PROMPT_CONSOLIDATE.replace("{limit}", str(limit)), payload)
    match = _JSON_BLOCK.search(raw)
    if not match:
        return facts
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return facts
    incoming = data.get("facts")
    if not isinstance(incoming, list) or not incoming:
        return facts
    merged = []
    for item in incoming[:limit]:
        if not isinstance(item, dict):
            continue
        body = _clean(item.get("text"), 60).strip()
        if not body:
            continue
        merged.append(
            {
                "type": _normalize_fact_type(item.get("type")),
                "text": body,
                "ts": stamp(),
            }
        )
    if not merged:
        return facts
    merged = dedupe_facts(merged)
    store.write_facts(merged)
    return merged


def facts_block(store: ChatStore, max_chars: int = 400) -> str:
    facts = store.facts()
    if not facts:
        return ""
    facts.sort(key=lambda item: (FACT_PRIORITY.get(str(item.get("type")), 5), str(item.get("ts") or "")))
    lines, used = [], 0
    for fact in facts:
        line = f"[{fact.get('type') or '其他'}] {fact.get('text')}"
        if used + len(line) + 1 > max_chars:
            break
        lines.append(line)
        used += len(line) + 1
    return "；".join(lines)


async def update_l5(store: ChatStore, llm: str, text: str) -> None:
    current = store.l5().get("summary") or ""
    body = f"现有记忆：{current or '（还没有）'}\n\n新的记录：\n{str(text)[:8000]}"
    raw = await _call(llm, PROMPT_L5, body)
    match = _JSON_BLOCK.search(raw)
    if not match:
        return
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return
    summary = _clean(data.get("summary"), 500)
    if summary:
        store.write_l5(summary)


async def enforce(store: ChatStore, llm, l1_max: int = L1_MAX, l2_max: int = L2_MAX,
                  archive_raw: bool = True) -> list[str]:
    actions = []
    archived = False
    for _ in range(4):
        if store.size("l1") > l1_max:
            if archive_raw and not archived:
                if store.archive_raw(store.read("l1")):
                    actions.append("raw")
                archived = True
            merged = await compress(store, llm, "l1", max(400, l1_max // 2))
            if merged:
                store.append("l2", merged + "\n")
                store.write("l1", "")
                actions.append("l1->l2")
                continue
            # 模型不可用时不能让它无限涨：保留最近一半
            if store.trim_tail("l1", max(400, l1_max // 2)):
                actions.append("l1-trim")
                continue
        if store.size("l2") > l2_max:
            if await seal_l2(store, llm):
                actions.append("sealL2")
                continue
            if store.trim_tail("l2", max(400, l2_max // 2)):
                actions.append("l2-trim")
                continue
        break
    return actions


# ---------- 注入 ----------

# ---------- 本地检索（不调模型） ----------

_ASCII = re.compile(r"[a-z0-9][a-z0-9._-]*")
_HAN = re.compile(r"[^\u4e00-\u9fff]+")


def tokenize(text) -> set[str]:
    raw = str(text or "").lower()
    out = set(_ASCII.findall(raw))
    for seg in _HAN.sub(" ", raw).split():
        if not seg:
            continue
        if len(seg) == 1:
            out.add(seg)
        else:
            out.add(seg)
            out.update(seg[i : i + 2] for i in range(len(seg) - 1))
    return out


def _weight(token: str) -> int:
    return min(4, max(1, len(token)))


def score_page(page: dict, text: str, tokens: set[str]) -> int:
    summary = str(page.get("summary") or "").lower()
    body = str(text or "").lower()
    tags = [str(tag).lower() for tag in (page.get("tags") or [])]
    score = 0
    for token in tokens:
        weight = _weight(token)
        if token in summary:
            score += 3 * weight
        if token in body:
            score += weight
        if any(token in tag or tag in token for tag in tags):
            score += 2 * weight
    return score


def load_vectors(store: ChatStore) -> dict:
    try:
        data = json.loads(store.read("vectors"))
    except (json.JSONDecodeError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def _encode_vector(vector) -> dict:
    """int8 量化：每维压成 [-127,127] 的整数，体积大约是原来的三分之一。"""
    values = [float(item) for item in (vector or [])]
    if not values:
        return {"s": 1.0, "v": []}
    scale = max((abs(item) for item in values), default=0.0) or 1.0
    quantized = [int(round(item / scale * 127)) for item in values]
    return {"s": round(scale, 6), "v": quantized}


def _decode_vector(entry) -> list[float]:
    if isinstance(entry, dict) and "v" in entry:
        scale = float(entry.get("s") or 1.0)
        return [int(item) * scale / 127 for item in entry["v"]]
    if isinstance(entry, list):
        return [float(item) for item in entry]
    return []


def save_vectors(store: ChatStore, data: dict) -> None:
    encoded = {
        key: (_encode_vector(value) if isinstance(value, list) else value)
        for key, value in data.items()
    }
    store.write_atomic("vectors", json.dumps(encoded, ensure_ascii=False))


def compact_vectors(store: ChatStore) -> int:
    """把旧的浮点数组格式转成 int8 量化，启动时跑一次即可。"""
    data = load_vectors(store)
    if not data:
        return 0
    if not any(isinstance(value, list) for value in data.values()):
        return 0
    decoded = {key: _decode_vector(value) for key, value in data.items()}
    save_vectors(store, decoded)
    return len(decoded)


def integrity(store: ChatStore) -> dict:
    """自检：行数和解析结果是否对得上，有没有半截行。"""
    l3_text = store.read("l3")
    l4_text = store.read("l4")
    l3_lines = len([line for line in l3_text.splitlines() if line.strip()])
    l4_lines = len([line for line in l4_text.splitlines() if line.strip()])
    return {
        "l3_lines": l3_lines,
        "l3_pages": len(store.l3()),
        "l4_lines": l4_lines,
        "l4_pages": len(store.l4()),
        "l1": store.size("l1"),
        "l2": store.size("l2"),
        "facts": len(store.facts()),
        "vectors": len(load_vectors(store)),
        "pending_index": sum(1 for page in store.l3() if needs_index(page)),
    }


def cosine(a, b) -> float:
    if not a or not b:
        return 0.0
    size = min(len(a), len(b))
    dot = sum(a[i] * b[i] for i in range(size))
    na = math.sqrt(sum(a[i] * a[i] for i in range(size)))
    nb = math.sqrt(sum(b[i] * b[i] for i in range(size)))
    if not na or not nb:
        return 0.0
    return dot / (na * nb)


async def backfill_vectors(store: ChatStore, embedder, limit: int = 32) -> int:
    """给还没有向量的归档页补 embedding。后台慢慢补，不阻塞聊天。"""
    if embedder is None:
        return 0
    vectors = load_vectors(store)
    texts = {str(page.get("p")): str(page.get("text") or "") for page in store.l4()}
    todo = [(p, text) for p, text in texts.items() if p not in vectors and text.strip()][:limit]
    if not todo:
        return 0
    try:
        embedded = await embedder.get_embeddings([text[:1500] for _p, text in todo])
    except Exception:
        return 0
    if not embedded or len(embedded) != len(todo):
        return 0
    for (page_id, _text), vector in zip(todo, embedded):
        vectors[page_id] = vector
    save_vectors(store, vectors)
    return len(todo)


def recall(store: ChatStore, query: str, limit: int = 2, min_score: float = 8.0,
           query_vector=None, vectors: dict | None = None) -> list[dict]:
    """按当前这句话去 L3 索引里找旧事。

    带 IDF：在越多页里出现的词权重越低，避免「我 / 你 / 今天」这种高频词把
    真正的旧事挤掉。纯本地，不花 token。
    """
    tokens = tokenize(query)
    if not tokens:
        return []
    # 单个汉字几乎每页都有，留着只会把排序拉平
    content_tokens = {token for token in tokens if len(token) >= 2}
    tokens = content_tokens or tokens
    pages = store.l3()
    if not pages:
        return []
    texts = {str(page.get("p")): str(page.get("text") or "") for page in store.l4()}
    docs = []
    for page in pages:
        text = texts.get(str(page.get("p")), "")
        docs.append((page, text, set(tokenize(str(page.get("summary") or "") + " " + text))))

    total = len(docs)
    df: dict[str, int] = {}
    for _page, _text, doc_tokens in docs:
        for token in doc_tokens:
            df[token] = df.get(token, 0) + 1

    scored = []
    for page, text, _doc_tokens in docs:
        summary = str(page.get("summary") or "").lower()
        body = str(text or "").lower()
        tags = [str(tag).lower() for tag in (page.get("tags") or [])]
        score = 0.0
        for token in tokens:
            weight = _weight(token)
            idf = math.log(1 + total / (1 + df.get(token, 0)))
            if token in summary:
                score += 3 * weight * idf
            if token in body:
                score += 1.5 * weight * idf
            if any(token in tag or tag in token for tag in tags):
                score += 2 * weight * idf
        item = dict(page)
        item["text"] = text
        item["keyword"] = round(score, 1)
        scored.append(item)

    if query_vector and vectors:
        for item in scored:
            vector = _decode_vector(vectors.get(str(item.get("p"))))
            item["similarity"] = round(cosine(query_vector, vector), 3) if vector else 0.0
        best_kw = max((item["keyword"] for item in scored), default=0) or 1.0
        for item in scored:
            # 语义为主、关键词为辅；关键词拿不到时也能靠向量命中
            item["score"] = round(
                0.65 * item["similarity"] + 0.35 * min(1.0, item["keyword"] / best_kw), 3
            )
        scored = [item for item in scored if item["score"] >= 0.35]
        scored.sort(key=lambda item: item["score"], reverse=True)
        return scored[: max(1, limit)]

    scored = [item for item in scored if item["keyword"] >= min_score]
    scored.sort(key=lambda item: item["keyword"], reverse=True)
    for item in scored:
        item["score"] = item["keyword"]
    return scored[: max(1, limit)]


def build_injection(store: ChatStore, first_turn: bool = False, keep_index: int = 6,
                    max_chars: int = 1200, query: str = "", auto_recall: bool = True,
                    facts_max_chars: int = 400, query_vector=None) -> str:
    parts = [
        "【你本来就知道的事】\n"
        "这些不是查来的，是你们相处里自然记住的。提到就自然接上，"
        "不要说“根据记忆”，也不要提“记录、文件、系统、提示词”。"
    ]
    facts = facts_block(store, facts_max_chars)
    if facts:
        parts.append("你对 TA 的了解：" + facts)
    memory = store.l5().get("summary")
    if memory:
        parts.append("你对 TA 的感觉：" + memory)

    if auto_recall and query:
        hits = recall(
            store,
            query,
            limit=2,
            query_vector=query_vector,
            vectors=load_vectors(store) if query_vector else None,
        )
        if hits:
            blocks = []
            for hit in hits:
                text = _clean(hit.get("text"), 240)
                blocks.append(f"· {hit.get('summary')}\n  {text}")
            parts.append("你们之前聊到过：\n" + "\n".join(blocks))

    indexes = store.l3()[-max(1, keep_index):]
    if indexes:
        lines = [f"· {_clean(page.get('summary'), 80)}" for page in reversed(indexes)]
        parts.append("更早的事：\n" + "\n".join(lines))

    l2 = store.read("l2").strip()
    if l2:
        parts.append("最近聊的：\n" + l2[-600:])

    # L1 本来就在聊天历史里，只在新会话开头交接一次
    l1 = store.read("l1").strip()
    if first_turn and l1:
        parts.append("上一段聊天的结尾：\n" + l1[-800:])

    if len(parts) == 1:
        return ""
    return "\n\n".join(parts)[:max_chars]


def build_state(store: ChatStore, l1_max: int, l2_max: int) -> dict:
    return {
        "updated": stamp(),
        "l1": {"size": store.size("l1"), "limit": l1_max},
        "l2": {"size": store.size("l2"), "limit": l2_max},
        "archive": len(store.l3()),
        "l5": bool(store.l5().get("summary")),
    }
