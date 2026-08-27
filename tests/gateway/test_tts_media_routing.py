"""
Tests for cross-platform audio/voice media routing.

These tests pin the expected delivery path for audio media files across
Telegram (where Bot-API sendAudio only accepts MP3/M4A and .ogg/.opus
only renders as a voice bubble when explicitly flagged) and via
``GatewayRunner._deliver_media_from_response``.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import (
    BasePlatformAdapter,
    MessageEvent,
    MessageType,
    SendResult,
    report_media_delivery_failure,
)
from gateway.run import GatewayRunner
from gateway.session import SessionSource, build_session_key


class _MediaRoutingAdapter(BasePlatformAdapter):
    def __init__(self):
        super().__init__(PlatformConfig(enabled=True, token="test"), Platform.TELEGRAM)

    async def connect(self):
        return True

    async def disconnect(self):
        pass

    async def send(self, chat_id, content=None, **kwargs):
        return SendResult(success=True, message_id="text")

    async def get_chat_info(self, chat_id):
        return {"id": chat_id, "type": "dm"}


def _event(thread_id=None):
    source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="chat-1",
        chat_type="dm",
        thread_id=thread_id,
    )
    return MessageEvent(
        text="make speech",
        message_type=MessageType.TEXT,
        source=source,
        message_id="msg-1",
    )


def _slack_event(*, chat_id="C123", chat_type="channel", thread_id=None):
    source = SessionSource(
        platform=Platform.SLACK,
        chat_id=chat_id,
        chat_type=chat_type,
        thread_id=thread_id,
    )
    return MessageEvent(
        text="make report",
        message_type=MessageType.TEXT,
        source=source,
        message_id="1785782926.578209",
    )


@pytest.mark.asyncio
async def test_media_failure_notice_retries_and_reports_delivery():
    adapter = SimpleNamespace(
        name="slack",
        _send_with_retry=AsyncMock(
            return_value=SendResult(success=True, message_id="notice")
        ),
    )

    delivered = await report_media_delivery_failure(
        adapter,
        chat_id="C123",
        thread_id="thread-1",
        file_path="/tmp/report.pdf",
        metadata={"thread_ts": "thread-1"},
        detail="upload failed",
    )

    assert delivered is True
    adapter._send_with_retry.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "side_effect",
    [
        SendResult(success=False, error="notice rejected"),
        RuntimeError("notice transport failed"),
    ],
)
async def test_media_failure_notice_reports_failed_delivery(side_effect):
    sender = AsyncMock()
    if isinstance(side_effect, Exception):
        sender.side_effect = side_effect
    else:
        sender.return_value = side_effect
    adapter = SimpleNamespace(name="slack", _send_with_retry=sender)

    delivered = await report_media_delivery_failure(
        adapter,
        chat_id="C123",
        thread_id="thread-1",
        file_path="/tmp/report.pdf",
        metadata={"thread_ts": "thread-1"},
        detail="upload failed",
    )

    assert delivered is False


@pytest.mark.asyncio
async def test_base_adapter_routes_telegram_flac_media_tag_to_document_sender():
    adapter = _MediaRoutingAdapter()
    event = _event()
    adapter._message_handler = AsyncMock(return_value="MEDIA:/tmp/speech.flac")
    adapter.send_voice = AsyncMock(return_value=SendResult(success=True, message_id="voice"))
    adapter.send_document = AsyncMock(return_value=SendResult(success=True, message_id="doc"))

    await adapter._process_message_background(event, build_session_key(event.source))

    adapter.send_document.assert_awaited_once_with(
        chat_id="chat-1",
        file_path="/tmp/speech.flac",
        metadata=None,
    )
    adapter.send_voice.assert_not_awaited()


@pytest.mark.asyncio
async def test_base_adapter_routes_non_voice_telegram_ogg_media_tag_to_document_sender():
    adapter = _MediaRoutingAdapter()
    event = _event()
    adapter._message_handler = AsyncMock(return_value="MEDIA:/tmp/speech.ogg")
    adapter.send_voice = AsyncMock(return_value=SendResult(success=True, message_id="voice"))
    adapter.send_document = AsyncMock(return_value=SendResult(success=True, message_id="doc"))

    await adapter._process_message_background(event, build_session_key(event.source))

    adapter.send_document.assert_awaited_once_with(
        chat_id="chat-1",
        file_path="/tmp/speech.ogg",
        metadata=None,
    )
    adapter.send_voice.assert_not_awaited()


@pytest.mark.asyncio
async def test_base_adapter_routes_voice_tagged_telegram_ogg_media_tag_to_voice_sender():
    adapter = _MediaRoutingAdapter()
    event = _event()
    adapter._message_handler = AsyncMock(
        return_value="[[audio_as_voice]]\nMEDIA:/tmp/speech.ogg"
    )
    adapter.send_voice = AsyncMock(return_value=SendResult(success=True, message_id="voice"))
    adapter.send_document = AsyncMock(return_value=SendResult(success=True, message_id="doc"))

    await adapter._process_message_background(event, build_session_key(event.source))

    adapter.send_voice.assert_awaited_once_with(
        chat_id="chat-1",
        audio_path="/tmp/speech.ogg",
        metadata=None,
    )
    adapter.send_document.assert_not_awaited()


@pytest.mark.asyncio
async def test_base_adapter_surfaces_html_upload_failure_in_same_thread():
    adapter = _MediaRoutingAdapter()
    event = _event(thread_id="topic-1")
    adapter._message_handler = AsyncMock(return_value="MEDIA:/tmp/report.html")
    adapter.send_document = AsyncMock(
        return_value=SendResult(success=False, error="artifact path rejected")
    )
    adapter.send = AsyncMock(return_value=SendResult(success=True, message_id="notice"))

    await adapter._process_message_background(event, build_session_key(event.source))

    adapter.send.assert_awaited_once()
    assert adapter.send.await_args is not None
    notice = adapter.send.await_args.kwargs
    assert notice["metadata"]["thread_id"] == "topic-1"
    assert "was not attached" in notice["content"]
    assert "artifact path rejected" in notice["content"]


def _fake_runner(thread_meta):
    """Build a fake GatewayRunner-like object with the helper methods needed by
    _deliver_media_from_response."""
    runner = SimpleNamespace(
        _thread_metadata_for_source=lambda source, anchor=None: thread_meta,
        _reply_anchor_for_event=lambda event: None,
    )
    return runner


@pytest.mark.asyncio
async def test_streaming_plain_text_has_no_media_delivery_outcome():
    event = _event(thread_id="topic-1")
    adapter = SimpleNamespace(
        name="test",
        extract_media=BasePlatformAdapter.extract_media,
        extract_images=BasePlatformAdapter.extract_images,
        extract_local_files=BasePlatformAdapter.extract_local_files,
    )

    delivered = await GatewayRunner._deliver_media_from_response(
        _fake_runner({"thread_id": "topic-1"}),
        "plain streamed response",
        event,
        adapter,
    )

    assert delivered is None


@pytest.mark.asyncio
async def test_streaming_delivery_routes_telegram_flac_media_tag_to_document_sender():
    event = _event(thread_id="topic-1")
    adapter = SimpleNamespace(
        name="test",
        extract_media=BasePlatformAdapter.extract_media,
        extract_images=BasePlatformAdapter.extract_images,
        extract_local_files=BasePlatformAdapter.extract_local_files,
        send_voice=AsyncMock(return_value=SendResult(success=True, message_id="voice")),
        send_document=AsyncMock(return_value=SendResult(success=True, message_id="doc")),
        send_image_file=AsyncMock(return_value=SendResult(success=True, message_id="image")),
        send_video=AsyncMock(return_value=SendResult(success=True, message_id="video")),
    )

    await GatewayRunner._deliver_media_from_response(
        _fake_runner({"thread_id": "topic-1"}),
        "MEDIA:/tmp/speech.flac",
        event,
        adapter,
    )

    adapter.send_document.assert_awaited_once_with(
        chat_id="chat-1",
        file_path="/tmp/speech.flac",
        metadata={"thread_id": "topic-1"},
    )
    adapter.send_voice.assert_not_awaited()


@pytest.mark.asyncio
async def test_streaming_media_tag_is_not_rediscovered_as_a_bare_path(tmp_path):
    event = _event(thread_id="topic-1")
    artifact = tmp_path / "report.pdf"
    artifact.write_bytes(b"report")
    adapter = SimpleNamespace(
        name="test",
        extract_media=BasePlatformAdapter.extract_media,
        extract_images=BasePlatformAdapter.extract_images,
        extract_local_files=BasePlatformAdapter.extract_local_files,
        send_voice=AsyncMock(return_value=SendResult(success=True)),
        send_document=AsyncMock(return_value=SendResult(success=True, message_id="doc")),
        send_multiple_images=AsyncMock(return_value=SendResult(success=True)),
        send_video=AsyncMock(return_value=SendResult(success=True)),
    )

    delivered = await GatewayRunner._deliver_media_from_response(
        _fake_runner({"thread_id": "topic-1"}),
        f"MEDIA:{artifact}",
        event,
        adapter,
    )

    assert delivered is True
    adapter.send_document.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("event", "thread_meta"),
    [
        (_slack_event(), None),
        (
            _slack_event(thread_id="1785782926.578209"),
            {"thread_ts": "1785782926.578209"},
        ),
        (
            _slack_event(
                chat_id="G123",
                chat_type="group",
                thread_id="1785782926.578209",
            ),
            {"thread_ts": "1785782926.578209"},
        ),
    ],
    ids=["root-channel", "channel-thread", "group-dm-thread"],
)
async def test_streaming_delivery_routes_html_media_to_slack_destination(
    event, thread_meta
):
    adapter = SimpleNamespace(
        name="slack",
        extract_media=BasePlatformAdapter.extract_media,
        extract_images=BasePlatformAdapter.extract_images,
        extract_local_files=BasePlatformAdapter.extract_local_files,
        send_voice=AsyncMock(return_value=SendResult(success=True, message_id="voice")),
        send_document=AsyncMock(
            return_value=SendResult(success=True, message_id="F123")
        ),
        send_multiple_images=AsyncMock(
            return_value=SendResult(success=True, message_id="images")
        ),
        send_video=AsyncMock(return_value=SendResult(success=True, message_id="video")),
        send=AsyncMock(return_value=SendResult(success=True, message_id="notice")),
    )
    artifact = "/opt/data/artifacts/slack/abc-leadership-recap.html"

    await GatewayRunner._deliver_media_from_response(
        _fake_runner(thread_meta),
        f"Your report is attached.\nMEDIA:{artifact}",
        event,
        adapter,
    )

    adapter.send_document.assert_awaited_once_with(
        chat_id=event.source.chat_id,
        file_path=artifact,
        metadata=thread_meta,
    )


@pytest.mark.asyncio
async def test_streaming_delivery_surfaces_html_upload_rejection_in_same_thread():
    thread_ts = "1785782926.578209"
    event = _slack_event(thread_id=thread_ts)
    adapter = SimpleNamespace(
        name="slack",
        extract_media=BasePlatformAdapter.extract_media,
        extract_images=BasePlatformAdapter.extract_images,
        extract_local_files=BasePlatformAdapter.extract_local_files,
        send_voice=AsyncMock(return_value=SendResult(success=True)),
        send_document=AsyncMock(
            return_value=SendResult(
                success=False,
                error="Slack upload denied: file is outside approved generated-artifact roots.",
            )
        ),
        send_multiple_images=AsyncMock(return_value=SendResult(success=True)),
        send_video=AsyncMock(return_value=SendResult(success=True)),
        send=AsyncMock(return_value=SendResult(success=True, message_id="notice")),
    )
    untrusted = "/tmp/not-a-trusted-artifact.html"

    await GatewayRunner._deliver_media_from_response(
        _fake_runner({"thread_ts": thread_ts}),
        f"MEDIA:{untrusted}",
        event,
        adapter,
    )

    adapter.send.assert_awaited_once()
    assert adapter.send.await_args is not None
    notice = adapter.send.await_args.kwargs
    assert notice["chat_id"] == "C123"
    assert notice["metadata"] == {"thread_ts": thread_ts}
    assert "was not attached" in notice["content"]
    assert "outside approved generated-artifact roots" in notice["content"]


@pytest.mark.asyncio
async def test_streaming_delivery_surfaces_image_batch_failure_in_same_thread():
    thread_ts = "1785782926.578209"
    event = _slack_event(thread_id=thread_ts)
    adapter = SimpleNamespace(
        name="slack",
        extract_media=BasePlatformAdapter.extract_media,
        extract_images=BasePlatformAdapter.extract_images,
        extract_local_files=BasePlatformAdapter.extract_local_files,
        send_voice=AsyncMock(return_value=SendResult(success=True)),
        send_document=AsyncMock(return_value=SendResult(success=True)),
        send_multiple_images=AsyncMock(
            return_value=SendResult(success=False, error="Slack image upload failed")
        ),
        send_video=AsyncMock(return_value=SendResult(success=True)),
        send=AsyncMock(return_value=SendResult(success=True, message_id="notice")),
    )
    image = "/opt/data/artifacts/slack/chart.png"

    delivered = await GatewayRunner._deliver_media_from_response(
        _fake_runner({"thread_ts": thread_ts}),
        f"MEDIA:{image}",
        event,
        adapter,
    )

    assert delivered is False
    assert event.delivery_state.reply_failed is False
    assert event.delivery_state.failure_notice_delivered is True
    adapter.send_multiple_images.assert_awaited_once()
    adapter.send.assert_awaited_once()
    notice = adapter.send.await_args.kwargs
    assert notice["metadata"] == {"thread_ts": thread_ts}
    assert "Slack image upload failed" in notice["content"]


@pytest.mark.asyncio
async def test_streaming_delivery_routes_non_voice_telegram_ogg_media_tag_to_document_sender():
    event = _event(thread_id="topic-1")
    adapter = SimpleNamespace(
        name="test",
        extract_media=BasePlatformAdapter.extract_media,
        extract_images=BasePlatformAdapter.extract_images,
        extract_local_files=BasePlatformAdapter.extract_local_files,
        send_voice=AsyncMock(return_value=SendResult(success=True, message_id="voice")),
        send_document=AsyncMock(return_value=SendResult(success=True, message_id="doc")),
        send_image_file=AsyncMock(return_value=SendResult(success=True, message_id="image")),
        send_video=AsyncMock(return_value=SendResult(success=True, message_id="video")),
    )

    await GatewayRunner._deliver_media_from_response(
        _fake_runner({"thread_id": "topic-1"}),
        "MEDIA:/tmp/speech.ogg",
        event,
        adapter,
    )

    adapter.send_document.assert_awaited_once_with(
        chat_id="chat-1",
        file_path="/tmp/speech.ogg",
        metadata={"thread_id": "topic-1"},
    )
    adapter.send_voice.assert_not_awaited()


@pytest.mark.asyncio
async def test_streaming_delivery_routes_telegram_mp3_media_tag_to_voice_sender():
    """MP3 audio on Telegram must go through send_voice (which routes to
    sendAudio internally); Telegram accepts MP3 for the audio player."""
    event = _event(thread_id="topic-1")
    adapter = SimpleNamespace(
        name="test",
        extract_media=BasePlatformAdapter.extract_media,
        extract_images=BasePlatformAdapter.extract_images,
        extract_local_files=BasePlatformAdapter.extract_local_files,
        send_voice=AsyncMock(return_value=SendResult(success=True, message_id="voice")),
        send_document=AsyncMock(return_value=SendResult(success=True, message_id="doc")),
        send_image_file=AsyncMock(return_value=SendResult(success=True, message_id="image")),
        send_video=AsyncMock(return_value=SendResult(success=True, message_id="video")),
    )

    await GatewayRunner._deliver_media_from_response(
        _fake_runner({"thread_id": "topic-1"}),
        "MEDIA:/tmp/speech.mp3",
        event,
        adapter,
    )

    adapter.send_voice.assert_awaited_once_with(
        chat_id="chat-1",
        audio_path="/tmp/speech.mp3",
        metadata={"thread_id": "topic-1"},
    )
    adapter.send_document.assert_not_awaited()
