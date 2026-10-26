"""把一份「记忆本」Markdown 灌进事实卡。

用法：

    python seed_facts.py --file "D:\\记忆本.md"
    python seed_facts.py --file "...md" --section 用户 --dry-run
    python seed_facts.py --file "...md" --section 用户,环境

默认只取「用户」这一节（关于 TA 的事实），「我是谁」是机器人自己，不该进事实卡。
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from chatmem import ChatStore, load_vectors  # noqa: F401,E402
from chatmem import stamp  # noqa: E402

PLUGIN_DIR = Path(__file__).resolve().parent
DATA_ROOT = PLUGIN_DIR.parents[1]  # <AstrBot>/data
DEFAULT_DATA = str(DATA_ROOT / "plugin_data" / PLUGIN_DIR.name / "chat")


def parse_sections(text: str) -> dict[str, list[str]]:
    sections: dict[str, list[str]] = {}
    current = ""
    for raw in text.splitlines():
        line = raw.rstrip()
        if line.startswith("## "):
            current = line[3:].strip().lstrip("#").strip()
            sections.setdefault(current, [])
            continue
        if line.startswith("# ") and not line.startswith("## "):
            sections.setdefault("_title", [])
            continue
        bullet = re.match(r"^\s*-\s+(.*)$", line)
        if bullet and current:
            item = re.sub(r"\*\*(.+?)\*\*", r"\1", bullet.group(1)).strip()
            if item:
                sections[current].append(item)
    return sections


def guess_type(text: str) -> str:
    if any(word in text for word in ("约定", "答应", "说好")):
        return "约定"
    if any(word in text for word in ("称呼", "叫", "昵称")):
        return "称呼"
    if any(word in text for word in ("喜欢", "爱", "玩", "偏好", "分享过")):
        return "偏好"
    if any(word in text for word in ("不要", "禁止", "讨厌", "雷点", "怕")):
        return "雷点"
    return "事件"


def main() -> int:
    parser = argparse.ArgumentParser(description="把记忆本 Markdown 灌进事实卡")
    parser.add_argument("--file", required=True)
    parser.add_argument("--data-dir", default=DEFAULT_DATA)
    parser.add_argument("--section", default="用户", help="要导入的章节，逗号分隔，默认「用户」")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    path = Path(args.file)
    if not path.exists():
        print(f"找不到文件：{path}")
        return 1
    text = path.read_text(encoding="utf-8")
    sections = parse_sections(text)
    wanted = [name.strip() for name in args.section.replace("，", ",").split(",") if name.strip()]

    picked: list[tuple[str, str]] = []
    for name in wanted:
        items = sections.get(name)
        if not items:
            print(f"跳过「{name}」：没找到这一节或没有条目")
            continue
        for item in items:
            body = re.sub(r"\s+", " ", item).replace("`", "").strip()
            if len(body) < 4:
                continue
            picked.append((guess_type(body), body[:80]))

    if not picked:
        print("没有可导入的条目。")
        return 1

    for kind, body in picked:
        print(f"  [{kind}] {body}")

    if args.dry_run:
        print(f"\n[dry-run] 共 {len(picked)} 条，没有写入。")
        return 0

    store = ChatStore(args.data_dir)
    facts = store.facts()
    existing = {str(item.get("text") or "") for item in facts}
    added = 0
    for kind, body in picked:
        if body in existing:
            continue
        facts.append({"type": kind, "text": body, "ts": stamp()})
        existing.add(body)
        added += 1
    store.write_facts(facts)
    print(f"\n完成：新增 {added} 条，事实卡共 {len(facts)} 条。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
