from __future__ import annotations

import asyncio
import hashlib
import os
import random
import uuid
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

import aiohttp
from astrbot.api import logger
from astrbot.core.message.components import BaseMessageComponent, Face, Image, Plain
from astrbot.core.utils.astrbot_path import get_astrbot_temp_path

from .config import PluginConfig
from .state import GroupState


@dataclass(slots=True)
class ActionPlan:
    repeat_count: int
    action: str
    output_segment: BaseMessageComponent | None = None


class RereadEngine:
    ACTIONS = ("follow", "interrupt", "ban")
    # 图片只需采样开头一段内容即可判重，避免整图读取/下载
    IMAGE_SAMPLE_BYTES = 256 * 1024
    FACE_PREFIX = "[表情:"

    def __init__(self, cfg: PluginConfig):
        self.cfg = cfg
        pool = []
        for action in self.ACTIONS:
            weight = cfg.weight_of(action)
            if weight > 0:
                pool.append((action, weight))

        self._actions, self._weights = zip(*pool) if pool else ((), ())
        if not self._actions:
            logger.warning(
                "[reread] all action weights are 0, reread actions are disabled"
            )

        self._session: aiohttp.ClientSession | None = None

    async def close(self) -> None:
        if self._session is not None and not self._session.closed:
            await self._session.close()
        self._session = None

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                trust_env=True,
                timeout=aiohttp.ClientTimeout(total=10),
            )
        return self._session

    async def fingerprint(self, seg: BaseMessageComponent, seg_type: str) -> str:
        """生成稳定的消息指纹；图片读取内容摘要，其他类型直接由字段生成。"""
        if seg_type == "Image" or isinstance(seg, Image):
            return await self._image_fingerprint(seg)
        return self.make_fingerprint(seg, seg_type)

    @staticmethod
    def make_fingerprint(
        seg: BaseMessageComponent, seg_type: str = ""
    ) -> str:
        if isinstance(seg, Plain):
            # QQ 官方表情被转换成纯文本，与普通文本区分开
            return f"face:{seg.text}" if seg_type == "Face" else f"text:{seg.text}"

        if isinstance(seg, Image):
            return f"image:{seg.file or seg.url or seg.path}"

        if isinstance(seg, Face):
            return f"face:{seg.id}"

        return f"unknown:{seg.type}"

    @staticmethod
    def _local_path(ref: str) -> str | None:
        """把本地路径 / file:// URI 转成可读路径；HTTP 地址返回 None。"""
        if ref.startswith(("http://", "https://")):
            return None
        if ref.startswith("file://"):
            return unquote(urlparse(ref).path)
        try:
            return ref if os.path.exists(ref) else None
        except OSError:
            return None

    async def _image_fingerprint(self, seg: Image) -> str:
        """按图片内容判重。

        QQ 官方适配器可能把附件下载成本地临时文件（file 为本地路径），
        也可能保留远程 URL；URL 里的 rkey 每次发送都会变化，因此都不能
        直接拿地址当指纹。
        """
        ref = seg.file or seg.url or seg.path or ""
        if not ref:
            return "image:unknown"

        local_path = self._local_path(ref)
        if local_path:
            try:
                digest = hashlib.sha256()
                with open(local_path, "rb") as fh:
                    digest.update(fh.read(self.IMAGE_SAMPLE_BYTES))
                return f"image:{digest.hexdigest()}"
            except OSError as exc:
                logger.debug(f"[reread] 读取本地图片失败，回退到路径: {exc}")

        try:
            digest = hashlib.sha256()
            session = await self._get_session()
            async with session.get(ref) as resp:
                resp.raise_for_status()
                remaining = self.IMAGE_SAMPLE_BYTES
                async for chunk in resp.content.iter_chunked(64 * 1024):
                    digest.update(chunk)
                    remaining -= len(chunk)
                    if remaining <= 0:
                        break
            return f"image:{digest.hexdigest()}"
        except Exception as exc:
            logger.debug(f"[reread] 图片指纹计算失败，回退到地址: {exc}")
            return f"image:{self._url_key(ref)}"

    @staticmethod
    def _url_key(url: str) -> str:
        """下载失败时的兜底：fileid 在一次上传内稳定，且已去掉会变化的 rkey。"""
        parsed = urlparse(url)
        fileid = parse_qs(parsed.query).get("fileid", [""])[0]
        return fileid or f"{parsed.netloc}{parsed.path}"

    @classmethod
    def _face_display_text(cls, text: str) -> str:
        """QQ 官方群机器人只能发文本，表情用可读文字回显。"""
        if text.startswith(cls.FACE_PREFIX) and text.endswith("]"):
            inner = text[len(cls.FACE_PREFIX) : -1]
            return inner or "[表情]"
        return text

    async def sticker_image(self, seg: Image) -> Image:
        """把表情包转成单帧 GIF，避免以普通照片形式发送。"""
        ref = seg.file or seg.path or seg.url or ""
        local_path = self._local_path(ref)
        if not local_path:
            return seg

        try:
            gif_path = await asyncio.to_thread(self._convert_to_gif, local_path)
        except Exception as exc:
            logger.debug(f"[reread] 表情包转 GIF 失败，按原图发送: {exc}")
            return seg

        return Image(file=gif_path, path=gif_path)

    @staticmethod
    def _convert_to_gif(src: str) -> str:
        from PIL import Image as PILImage

        with PILImage.open(src) as im:
            im.seek(0)
            frame = im.convert("RGBA")

        out = str(
            Path(get_astrbot_temp_path()) / f"echo_sticker_{uuid.uuid4().hex}.gif"
        )
        # 单帧 GIF，保留透明背景
        frame.save(out, format="GIF", save_all=False)
        return out

    def evaluate(
        self,
        state: GroupState,
        seg_type: str,
        send_id: str,
        seg: BaseMessageComponent,
        fingerprint: str,
    ) -> ActionPlan | None:
        state.clear_if_same_sender(seg_type, send_id, self.cfg.need_different)

        state.push_message(seg_type, send_id, fingerprint)

        reread_threshold = self.cfg.get_threshold(seg_type)
        reread_fingerprint = state.get_uniform_tail_fingerprint(
            seg_type, reread_threshold
        )
        if not reread_fingerprint or state.is_same_as_last_action(reread_fingerprint):
            return None

        if not self.cfg.should_reread():
            return None

        if not self._actions:
            return None

        state.mark_handled(reread_fingerprint)
        action = random.choices(self._actions, self._weights, k=1)[0]
        if action == "ban":
            return ActionPlan(repeat_count=reread_threshold, action=action)

        if action == "interrupt":
            output_segment = Plain(self.cfg.interrupt.text)
        elif seg_type == "Face" and isinstance(seg, Plain):
            output_segment = Plain(self._face_display_text(seg.text))
        else:
            output_segment = seg

        return ActionPlan(
            repeat_count=reread_threshold,
            action=action,
            output_segment=output_segment,
        )
