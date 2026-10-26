"""把安装插件之前的聊天记录导入 TMC 记忆。

用法（用 AstrBot 的 venv python 跑）：

    # 1. 看看有哪些会话
    python import_history.py --list

    # 2. 先试运行，不写入
    python import_history.py --all --dry-run

    # 3. 正式导入（全部会话）
    python import_history.py --all

    # 只导入某一个会话
    python import_history.py --conversation 577a3bc9-1a48-45a9-b00b-ab9b59a58a66

    # 4. 导入 QQ 导出的聊天文本（txt / md / json 都行，UTF-8 或 GBK 自动识别）
    python import_history.py --file "D:\\备份\\和她的聊天.txt"

默认从 AstrBot 的 data/data_v4.db 读，只读打开，不会改动原库。
导入后旧对话会：
  · 原文落到 chat/raw/import-日期.md
  · 按 ~1200 字切块写进 L3/L4（带索引），自动回忆能直接检索到
  · 最近一段写进 L1，接上当前对话
不调用模型，不花钱。
"""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from chatmem import ChatStore, stamp  # noqa: E402

PLUGIN_DIR = Path(__file__).resolve().parent
DATA_ROOT = PLUGIN_DIR.parents[1]  # <AstrBot>/data
DEFAULT_DB = str(DATA_ROOT / "data_v4.db")
DEFAULT_DATA = str(DATA_ROOT / "plugin_data" / PLUGIN_DIR.name / "chat")

CHUNK = 1200

_REMINDER = re.compile(r"<system_reminder>.*?</system_reminder>", re.S)


def clean_text(text: str) -> str:
    """去掉 AstrBot 注入的 <system_reminder> 时间和提醒，别记进记忆。"""
    return _REMINDER.sub("", str(text or "")).strip()


def parts_to_text(content) -> str:
    """content 是 [{'type':'text','text':...}, ...]；跳过 think。"""
    if not isinstance(content, list):
        return str(content or "")
    out = []
    for part in content:
        if not isinstance(part, dict):
            continue
        if part.get("type") != "text":
            continue
        text = str(part.get("text") or "").strip()
        if text:
            out.append(clean_text(text))
    return "\n".join(out).strip()


def load_conversations(db_path: str) -> list[dict]:
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        rows = conn.execute(
            "select conversation_id, platform_id, user_id, persona_id, content, updated_at "
            "from conversations order by updated_at asc"
        ).fetchall()
    finally:
        conn.close()
    result = []
    for cid, platform_id, user_id, persona_id, content, updated in rows:
        try:
            messages = json.loads(content or "[]")
        except json.JSONDecodeError:
            messages = []
        result.append(
            {
                "id": cid,
                "platform": platform_id,
                "user": user_id,
                "persona": persona_id,
                "updated": updated,
                "messages": messages if isinstance(messages, list) else [],
            }
        )
    return result


def format_messages(messages) -> str:
    lines = []
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        role = msg.get("role")
        text = parts_to_text(msg.get("content"))
        if not text:
            continue
        if role == "user":
            lines.append(f"对方：{text}")
        elif role == "assistant":
            lines.append(f"她：{text}")
    return "\n".join(lines)


def kind_of(user_id: str) -> str:
    text = str(user_id or "")
    if "GroupMessage" in text:
        return "群聊"
    if "FriendMessage" in text:
        return "私聊"
    return "其他"


def chunk_text(text: str, size: int = CHUNK) -> list[str]:
    chunks, current = [], ""
    for line in text.split("\n"):
        while len(line) > size:
            if current:
                chunks.append(current)
                current = ""
            chunks.append(line[:size])
            line = line[size:]
        if len(current) + len(line) + 1 > size:
            chunks.append(current)
            current = line
        else:
            current = f"{current}\n{line}" if current else line
    if current:
        chunks.append(current)
    return [chunk.strip() for chunk in chunks if chunk.strip()]


def summarize(chunk: str) -> str:
    flat = " ".join(chunk.split())
    return flat[:120] + ("…" if len(flat) > 120 else "")


def read_text_file(path: str) -> str:
    """QQ 导出的 txt 常见 UTF-8 / GBK / UTF-16，逐个试。"""
    raw = Path(path).read_bytes()
    for encoding in ("utf-8-sig", "utf-8", "gb18030", "utf-16"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")


def import_files(args) -> int:
    total_pages = 0
    targets = []
    for item in args.file:
        path = Path(item)
        if not path.exists():
            print(f"找不到文件：{item}")
            continue
        targets.append(path)

    if not targets:
        print("没有可导入的文件。")
        return 1

    store = ChatStore(args.data_dir)
    for path in targets:
        text = clean_text(read_text_file(str(path)))
        if not text.strip():
            print(f"跳过 {path.name}：没有文本")
            continue
        chunks = chunk_text(text)
        print(f"{path.name}: {len(text)} 字，切成 {len(chunks)} 块")
        if args.dry_run:
            continue
        store.archive_raw(f"# 导入 {path.name}\n{text}")
        for chunk in chunks:
            store.record_page(chunk, summarize(chunk), [], src="import")
            total_pages += 1
        store.write("l1", text[-CHUNK:].strip() + "\n")
        print(f"  -> 写入 {len(chunks)} 页，L1 已接上最后 {min(len(text), CHUNK)} 字")

    if args.dry_run:
        print("\n[dry-run] 没有写入任何文件。")
    else:
        print(f"\n完成：新增 {total_pages} 页记忆，时间 {stamp()}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="导入安装前的聊天记录")
    parser.add_argument("--db", default=DEFAULT_DB)
    parser.add_argument("--data-dir", default=DEFAULT_DATA)
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--conversation", default="")
    parser.add_argument("--file", action="append", default=[], help="导入导出的聊天文本（txt/md/json），可重复")
    parser.add_argument("--include-groups", action="store_true", help="默认只导私聊，加这个才导群聊")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.file:
        return import_files(args)

    conversations = load_conversations(args.db)
    if not args.include_groups:
        conversations = [conv for conv in conversations if kind_of(conv["user"]) != "群聊"]

    if args.list or (not args.all and not args.conversation):
        print(f"共 {len(conversations)} 个私聊会话（群聊已跳过，加 --include-groups 可包含）：\n")
        for conv in conversations:
            text = format_messages(conv["messages"])
            print(f"  {conv['id']}")
            print(f"     类型={kind_of(conv['user'])} 平台={conv['platform']} 用户={conv['user']}")
            print(f"     消息={len(conv['messages'])} 字数={len(text)} 更新={conv['updated']}")
        print("\n用 --all 或 --conversation <id> 导入。")
        return 0

    if args.all:
        targets = conversations
    else:
        targets = [conv for conv in conversations if conv["id"] == args.conversation]
        if not targets:
            print(f"找不到会话 {args.conversation}（如果是群聊，要加 --include-groups）")
            return 1

    store = ChatStore(args.data_dir)
    total_pages = 0
    for conv in targets:
        text = format_messages(conv["messages"])
        if not text.strip():
            print(f"跳过 {conv['id']}：没有可导入的文本")
            continue
        chunks = chunk_text(text)
        print(f"{conv['id']}: {len(text)} 字，切成 {len(chunks)} 块")
        if args.dry_run:
            continue
        store.archive_raw(f"# 导入 {conv['id']}\n{text}")
        for chunk in chunks:
            store.record_page(chunk, summarize(chunk), [], src="import")
            total_pages += 1
        # 最近一段接进 L1，让当前对话能延续
        store.write("l1", text[-CHUNK:].strip() + "\n")
        print(f"  -> 写入 {len(chunks)} 页，L1 已接上最后 {min(len(text), CHUNK)} 字")

    if args.dry_run:
        print("\n[dry-run] 没有写入任何文件。")
    else:
        print(f"\n完成：新增 {total_pages} 页记忆，时间 {stamp()}")
        print("在 WebUI 重载插件后，自动回忆即可检索到这些旧对话。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
