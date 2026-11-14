import re

from astrbot.api.event import filter
from astrbot.api.star import Context, Star
from astrbot.core import AstrBotConfig
from astrbot.core.message.components import At, BaseMessageComponent, Image, Plain
from astrbot.core.platform import AstrMessageEvent
from astrbot.core.star.filter.event_message_type import EventMessageType

from .core.ban import RereadBanHandler
from .core.config import PluginConfig
from .core.engine import RereadEngine
from .core.state import StateManager

# QQ 官方机器人会把表情转成纯文本，例如 "[表情:[满头问号]]"
FACE_TEXT_RE = re.compile(r"^\[表情(?::.*)?\]$", re.S)


class RereadPlugin(Star):
    def __init__(self, context: Context, config: AstrBotConfig):
        super().__init__(context)
        self.cfg = PluginConfig(config)
        self.state_mgr = StateManager(self.cfg.window_sizes)
        self.engine = RereadEngine(self.cfg)
        self.ban_handler = RereadBanHandler(self.cfg)

    async def terminate(self):
        await self.engine.close()

    @filter.event_message_type(EventMessageType.GROUP_MESSAGE)
    async def reread_handle(self, event: AstrMessageEvent):
        if event.is_at_or_wake_command:
            return

        seg, seg_type = self._pick_segment(event.get_messages())
        if seg is None or not self.cfg.is_supported_type(seg_type):
            return

        if isinstance(seg, Plain) and any(
            word in seg.text for word in self.cfg.blocked_words
        ):
            return

        group_id = event.get_group_id()
        send_id = event.get_sender_id()

        if self.cfg.group_whitelist and not self.cfg.is_white_group(group_id):
            return

        # 图片指纹需要读取图片内容，放在锁外避免阻塞同群其他消息
        fingerprint = await self.engine.fingerprint(seg, seg_type)

        state = self.state_mgr.get_state(group_id)

        async with state.lock:
            plan = self.engine.evaluate(state, seg_type, send_id, seg, fingerprint)

        if not plan:
            return

        if plan.action == "ban":
            await self.ban_handler.handle(event, plan.repeat_count)
            return

        output_segment = plan.output_segment
        if not output_segment:
            return

        # 表情包转成单帧 GIF 再发，避免以普通照片形式刷屏
        if (
            plan.action == "follow"
            and seg_type == "Face"
            and isinstance(output_segment, Image)
        ):
            output_segment = await self.engine.sticker_image(output_segment)

        msg = event.chain_result([output_segment])
        # QQ 官方平台默认 use_markdown=True，会把复读文本当成 Markdown（msg_type=2）
        # 发出，导致在 QQ 里不是普通文本气泡、也无法参与「+1」连击。
        # 这里强制走纯文本（msg_type=0）。
        msg.use_markdown_ = False
        if self.cfg.skip_pipeline:
            await event.send(msg)
            event.stop_event()
        else:
            yield msg

    @classmethod
    def _pick_segment(
        cls, chain
    ) -> tuple[BaseMessageComponent | None, str]:
        """从 QQ 官方消息链中取出用于复读判定的那一段。

        QQ 官方适配器会给每条群消息附加 Plain 和 @ 段；表情包消息还会同时
        带一个表情标记和一张图片。这里统一归一化：

        - 表情包（表情标记 + 图片）按 Face 阈值计数，但以图片形式复读
        - 纯表情保持 Face 类型，按文字回显
        - 其余消息仍要求只有一段有效内容
        """
        has_image = any(isinstance(seg, Image) for seg in chain)
        face_marker = False
        segments: list[BaseMessageComponent] = []

        for seg in chain:
            if isinstance(seg, Plain):
                if not seg.text.strip():
                    continue
                if has_image and FACE_TEXT_RE.match(seg.text):
                    face_marker = True
                    continue
            elif isinstance(seg, At):
                continue
            segments.append(seg)

        if len(segments) != 1:
            return None, ""

        seg = segments[0]
        if isinstance(seg, Image) and face_marker:
            return seg, "Face"
        return seg, cls._segment_type(seg)

    @staticmethod
    def _segment_type(seg: BaseMessageComponent) -> str:
        if isinstance(seg, Plain) and FACE_TEXT_RE.match(seg.text):
            return "Face"
        return str(seg.type).split(".")[-1]
