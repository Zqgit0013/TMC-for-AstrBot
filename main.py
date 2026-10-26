"""只聊天的 AstrBot 长期记忆插件（TMC 轻量版）。

不做工具调用、不做多分区，插件自己记录 / 压缩 / 注入：
每轮把「对对方的理解 + 最近记忆索引 + 稍早摘要」拼进 system prompt，
新会话开头额外接上一段聊天的结尾。
"""

from __future__ import annotations

import asyncio
import re
import time

from astrbot.api import AstrBotConfig, logger
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.star import Context, Star, StarTools, register
from astrbot.core.star.filter.command import GreedyStr

from .chatmem import (
    L1_MAX,
    L2_MAX,
    ChatStore,
    backfill_vectors,
    build_injection,
    build_state,
    compact_vectors,
    consolidate_facts,
    enforce,
    integrity,
    needs_index,
    rebuild_indexes,
    update_facts,
    update_l5,
)

PLUGIN_NAME = "astrbot_plugin_tmc_chat"

HELP_TEXT = """TMC 聊天记忆
/tmc            查看状态
/tmc 事实       查看事实卡
/tmc 事实 重算  从归档原文重新抽取事实
/tmc l5         查看对对方的理解
/tmc l5 重算    用归档原文重新生成理解
/tmc seal       立即压缩一次
/tmc log        看最近压缩记录"""


@register(
    PLUGIN_NAME,
    "TMC",
    "只聊天的长期记忆：自动记录、压缩归档、每轮注入，不依赖工具调用。",
    "0.1.0",
    "",
)
class TmcChatPlugin(Star):
    def __init__(self, context: Context, config: AstrBotConfig | dict | None = None):
        super().__init__(context)
        self.config = config or {}
        base_dir = str(self.config.get("base_dir") or "").strip()
        if not base_dir:
            base_dir = str(StarTools.get_data_dir(PLUGIN_NAME) / "chat")
        self.store = ChatStore(base_dir)
        self.store.facts_limit = int(self.config.get("facts_limit", 24) or 24)
        self.store.facts_block = [
            word.strip()
            for word in str(self.config.get("facts_block", "") or "").replace("，", ",").split(",")
            if word.strip()
        ]
        self.store.bot_names = [
            name.strip()
            for name in str(self.config.get("bot_names", "") or "").replace("，", ",").split(",")
            if name.strip()
        ]
        self.l1_max = int(self.config.get("l1_max", L1_MAX) or L1_MAX)
        self.l2_max = int(self.config.get("l2_max", L2_MAX) or L2_MAX)
        self._task: asyncio.Task | None = None
        self._lock = asyncio.Lock()
        self._seen: dict[str, float] = {}
        logger.info(f"TMC 聊天记忆已加载：{self.store.root}")

    # ---------- 生命周期 ----------
    async def initialize(self) -> None:
        self.store.facts_llm = self._facts_llm
        self._self_check()
        try:
            converted = compact_vectors(self.store)
            if converted:
                logger.info(f"TMC 向量已压缩为 int8：{converted} 条")
        except Exception:
            logger.exception("TMC 向量压缩失败")
        self._task = asyncio.create_task(self._loop(), name="tmc-chat-tick")

    def _self_check(self) -> None:
        """启动时自检一次：行数对不上就写日志，不打扰聊天。"""
        try:
            data = integrity(self.store)
        except Exception:
            logger.exception("TMC 自检执行失败")
            return
        broken = data["l3_lines"] != data["l3_pages"] or data["l4_lines"] != data["l4_pages"]
        self.store.log(
            "integrity ok"
            if not broken
            else f"integrity warn l3={data['l3_pages']}/{data['l3_lines']} l4={data['l4_pages']}/{data['l4_lines']}"
        )
        if broken:
            logger.warning(f"TMC 自检发现损坏行：{data}")
        else:
            logger.info(
                f"TMC 自检通过：{data['l3_pages']} 页索引、{data['l4_pages']} 页原文、"
                f"{data['facts']} 条事实、{data['vectors']} 条向量"
            )

    async def terminate(self) -> None:
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("TMC 聊天记忆已停止")

    async def _loop(self) -> None:
        interval = max(15, int(self.config.get("tick_interval", 60)))
        while True:
            try:
                await asyncio.sleep(interval)
                await self._tick()
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("TMC 聊天记忆压缩出错")

    async def _tick(self) -> None:
        if self._lock.locked():
            return
        async with self._lock:
            actions = await enforce(
                self.store,
                self._llm,
                self.l1_max,
                self.l2_max,
                archive_raw=bool(self.config.get("archive_raw", True)),
            )
            embedder = self._embedder()
            if embedder is not None:
                added = await backfill_vectors(self.store, embedder, 24)
                if added:
                    actions.append(f"vectors+{added}")
            if actions:
                self.store.log("tick " + " ".join(actions))
                logger.info(f"TMC 聊天记忆：{actions}")

    # ---------- 模型 ----------
    def _embedder(self):
        """向量模型：默认用第一个可用的 embedding 提供商（比如 ollama/bge-m3）。"""
        if not self.config.get("vector_recall", True):
            return None
        try:
            providers = self.context.get_all_embedding_providers()
        except Exception:
            return None
        if not providers:
            return None
        wanted = str(self.config.get("embedding_provider_id") or "").strip()
        if wanted:
            for provider in providers:
                config = getattr(provider, "provider_config", {}) or {}
                if str(config.get("id") or "") == wanted:
                    return provider
        return providers[0]

    def _provider(self, provider_id: str = ""):
        """压缩/抽取用的模型：优先用传入的 id，否则 llm_provider_id，再否则当前对话模型。"""
        provider_id = (provider_id or str(self.config.get("llm_provider_id") or "")).strip()
        if provider_id:
            try:
                provider = self.context.get_provider_by_id(provider_id)
            except Exception:
                logger.exception(f"TMC 找不到提供商 {provider_id}，回退默认模型")
                provider = None
            if provider is not None:
                return provider
        return self.context.get_using_provider()

    async def _llm(self, instruction: str, text: str) -> str:
        return await self._call(self._provider(), instruction, text)

    async def _facts_llm(self, instruction: str, text: str) -> str:
        """事实抽取默认走当前对话模型（质量好），也可以单独指定。"""
        provider_id = str(self.config.get("facts_provider_id") or "").strip()
        provider = self._provider(provider_id) if provider_id else self.context.get_using_provider()
        return await self._call(provider, instruction, text)

    async def _call(self, provider, instruction: str, text: str) -> str:
        if provider is None:
            return ""
        try:
            response = await provider.text_chat(
                prompt=str(text),
                system_prompt=instruction,
                contexts=[],
            )
            return str(getattr(response, "completion_text", "") or "").strip()
        except Exception:
            logger.exception("TMC 调用模型失败")
            return ""

    # ---------- 记录 ----------
    def _capture(self) -> bool:
        return bool(self.config.get("auto_capture", True))

    def _append(self, line: str) -> None:
        # 一行一条：把换行压成空格，避免一条回复占十几行、把 L1 撑爆
        line = re.sub(r"\s+", " ", (line or "")).strip()
        if not line:
            return
        cap = max(80, int(self.config.get("line_max_chars", 300)))
        self.store.append("l1", line[:cap] + "\n")

    def _is_first_turn(self, umo: str) -> bool:
        now = time.time()
        if umo in self._seen:
            return False
        self._seen[umo] = now
        if len(self._seen) > 512:
            for key, ts in sorted(self._seen.items(), key=lambda item: item[1])[:256]:
                self._seen.pop(key, None)
        return True

    def _allowed(self, event: AstrMessageEvent) -> bool:
        """only_users 白名单：为空表示所有人都记。"""
        raw = str(self.config.get("only_users", "") or "").strip()
        if not raw:
            return True
        wanted = [item for item in re.split(r"[,，\s]+", raw) if item]
        sender = str(event.get_sender_id() or "")
        umo = str(getattr(event, "unified_msg_origin", "") or "")
        return any(item == sender or item in umo for item in wanted)

    @filter.on_llm_request()
    async def on_llm_request(self, event: AstrMessageEvent, req) -> None:
        if not self.config.get("enabled", True):
            return
        if not self._allowed(event):
            return
        message = (event.message_str or "").strip()
        if not message:
            return
        if self.config.get("ignore_groups", False) and event.get_group_id():
            return

        umo = getattr(event, "unified_msg_origin", "") or ""
        first_turn = self._is_first_turn(umo)
        try:
            event.set_extra("tmc_llm_turn", True)
        except Exception:
            pass

        if self._capture() and not message.startswith("/"):
            name = ""
            try:
                name = event.get_sender_name() or ""
            except Exception:
                name = ""
            label = f"对方（{name}）：" if name else "对方："
            self._append(f"{label}{message}")

        if not self.config.get("inject", True):
            return
        query_vector = None
        embedder = self._embedder()
        if embedder is not None:
            try:
                query_vector = await embedder.get_embedding(message[:500])
            except Exception:
                logger.debug("TMC 向量化失败，回退关键词检索", exc_info=True)
                query_vector = None
        section = build_injection(
            self.store,
            first_turn=first_turn,
            keep_index=int(self.config.get("keep_index", 6)),
            max_chars=int(self.config.get("inject_max_chars", 1200)),
            query=message,
            auto_recall=bool(self.config.get("auto_recall", True)),
            facts_max_chars=int(self.config.get("facts_max_chars", 400)),
            query_vector=query_vector,
        )
        if section:
            try:
                # 贴在用户消息后面，比塞在 system prompt 末尾更容易被模型采用
                from astrbot.core.agent.message import TextPart

                req.extra_user_content_parts.append(TextPart(text=section))
            except Exception:
                logger.debug("TMC 附加内容失败，退回 system prompt", exc_info=True)
                base = str(getattr(req, "system_prompt", "") or "")
                req.system_prompt = f"{base}\n\n{section}" if base else section

    @filter.on_llm_response()
    async def on_llm_response(self, event: AstrMessageEvent, response) -> None:
        if not self._capture():
            return
        if not self._allowed(event):
            return
        if self.config.get("ignore_groups", False) and event.get_group_id():
            return
        text = str(getattr(response, "completion_text", "") or "").strip()
        if text:
            self._append(f"她：{text}")

    def _last_line(self) -> str:
        for line in reversed(self.store.read("l1").splitlines()):
            if line.strip():
                return line.strip()
        return ""

    @filter.on_decorating_result()
    async def on_decorating_result(self, event: AstrMessageEvent) -> None:
        """补记真正发出去的内容：分段发送、主动消息都不会漏。"""
        if not self._capture():
            return
        try:
            if not event.get_extra("tmc_llm_turn", False):
                return
        except Exception:
            return
        if not self._allowed(event):
            return
        if self.config.get("ignore_groups", False) and event.get_group_id():
            return
        result = event.get_result()
        chain = getattr(result, "chain", None) or []
        pieces = [str(getattr(part, "text", "") or "") for part in chain]
        text = re.sub(r"\s+", " ", "".join(pieces)).strip()
        if not text:
            return
        try:
            event.set_extra("tmc_llm_turn", False)
        except Exception:
            pass
        last = self._last_line()
        if last.startswith("她：") and (text[:30] in last or last[2:32] in text):
            return  # 已经在 on_llm_response 记过
        self._append(f"她：{text}")

    # ---------- 指令 ----------
    @filter.command("tmc", alias={"记忆"}, desc="TMC 聊天记忆：状态 / 理解 / 压缩")
    async def tmc_command(self, event: AstrMessageEvent, args: GreedyStr):
        """TMC 聊天记忆指令。"""
        parts = str(args or "").strip().split(maxsplit=1)
        action = parts[0].lower() if parts else ""
        rest = parts[1].strip().lower() if len(parts) > 1 else ""
        if not action:
            yield event.plain_result(self._status())
            return
        if action in ("l5", "理解"):
            if rest in ("重算", "rebuild", "refresh"):
                pages = self.store.l4()[-5:]
                if not pages:
                    yield event.plain_result("还没有归档原文，没法重算。")
                    return
                text = "\n\n".join(str(page.get("text") or "") for page in pages)
                await update_l5(self.store, self._llm, text)
                summary = self.store.l5().get("summary")
                yield event.plain_result(summary or "模型没有返回结果，稍后再试。")
                return
            summary = self.store.l5().get("summary")
            yield event.plain_result(summary or "还没有形成对对方的理解。多聊一阵就有了。")
            return
        if action in ("事实", "facts"):
            if rest.startswith(("清空", "clear")):
                self.store.write_facts([])
                yield event.plain_result("事实卡已清空。")
                return
            if rest.startswith(("重算", "rebuild", "refresh")):
                pieces = rest.split()
                count = 5
                if len(pieces) > 1 and pieces[1].isdigit():
                    count = max(1, min(200, int(pieces[1])))
                pages = self.store.l4()[-count:]
                if not pages:
                    yield event.plain_result("还没有归档原文，没法重算。")
                    return
                text = "\n\n".join(str(page.get("text") or "") for page in pages)
                facts = self.store.facts()
                done = 0
                # 每 8000 字一批，避免一次塞太多被截断；不清空旧事实，是合并
                for start in range(0, len(text), 8000):
                    facts = await update_facts(
                        self.store, self._llm, text[start : start + 8000], merge=False
                    )
                    done += 1
                if len(facts) >= 8:
                    facts = await consolidate_facts(self.store, self._facts_llm)
                yield event.plain_result(
                    f"处理 {len(pages)} 页、{done} 批。\n" + (self._format_facts(facts) or "没有抽到新事实。")
                )
                return
            yield event.plain_result(self._format_facts(self.store.facts()))
            return
        if action in ("索引", "index"):
            if rest.startswith(("重建", "rebuild", "refresh")):
                pieces = rest.split()
                count = 30
                if len(pieces) > 1 and pieces[1].isdigit():
                    count = max(1, min(200, int(pieces[1])))
                pending = sum(1 for page in self.store.l3() if needs_index(page))
                yield event.plain_result(f"开始重建索引：待重建 {pending} 页，本次处理 {count} 页，可能要几分钟…")
                done, total = await rebuild_indexes(self.store, self._llm, count)
                yield event.plain_result(f"重建完成 {done}/{total} 页。对应向量已清空，后台会重算。")
                return
            pages = self.store.l3()
            pending = sum(1 for page in pages if needs_index(page))
            yield event.plain_result(
                f"共 {len(pages)} 页，其中 {pending} 页索引待重建。\n用 /tmc 索引 重建 50 开始。"
            )
            return
        if action in ("自检", "check"):
            data = integrity(self.store)
            lines = [
                f"索引 {data['l3_pages']} 页（文件 {data['l3_lines']} 行）",
                f"原文 {data['l4_pages']} 页（文件 {data['l4_lines']} 行）",
                f"L1 {data['l1']} 字  L2 {data['l2']} 字",
                f"事实 {data['facts']} 条  向量 {data['vectors']} 条",
                f"待重建索引 {data['pending_index']} 页",
            ]
            if data["l3_lines"] != data["l3_pages"]:
                lines.append("⚠ 索引里有坏行，读取时已跳过")
            if data["l4_lines"] != data["l4_pages"]:
                lines.append("⚠ 原文里有坏行，读取时已跳过")
            yield event.plain_result("\n".join(lines))
            return
        if action in ("预览", "preview"):
            if not rest:
                yield event.plain_result("用法：/tmc 预览 你还记得汤汤吗")
                return
            text = build_injection(
                self.store,
                first_turn=False,
                keep_index=int(self.config.get("keep_index", 6)),
                max_chars=int(self.config.get("inject_max_chars", 1200)),
                query=rest,
                auto_recall=True,
                facts_max_chars=int(self.config.get("facts_max_chars", 400)),
            )
            yield event.plain_result(f"注入 {len(text)} 字：\n\n{text[:1800] or '（空）'}")
            return
        if action in ("seal", "压缩"):
            await self._tick()
            yield event.plain_result("压缩完成。\n" + self._status())
            return
        if action in ("log", "日志"):
            body = self.store.read("log").strip()
            yield event.plain_result(body[-1500:] if body else "还没有压缩记录。")
            return
        yield event.plain_result(HELP_TEXT)

    @staticmethod
    def _format_facts(facts) -> str:
        if not facts:
            return "还没有抽到事实。多聊一阵，或者用 /tmc 事实 重算。"
        return "\n".join(f"[{item.get('type') or '其他'}] {item.get('text')}" for item in facts)

    def _status(self) -> str:
        state = build_state(self.store, self.l1_max, self.l2_max)
        return (
            "TMC 聊天记忆\n"
            f"L1 {state['l1']['size']}/{state['l1']['limit']} 字  "
            f"L2 {state['l2']['size']}/{state['l2']['limit']} 字\n"
            f"已归档 {state['archive']} 页  事实：{len(self.store.facts())} 条  "
            f"理解：{'有' if state['l5'] else '暂无'}"
        )
