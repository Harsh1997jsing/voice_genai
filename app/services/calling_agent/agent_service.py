"""
agent_service.py — ElevenLabs Conversational AI Agent WebSocket service.

Replaces: stt_service.py + tts_service.py + transcript_flow.py

The Agent handles STT + LLM + TTS internally.
Your server only needs to:
  1. Bridge raw audio from Twilio → Agent
  2. Bridge audio from Agent → Twilio
  3. Handle tool_call events (e.g. transfer_to_human)
"""

import asyncio
import base64
import json
from collections.abc import Callable, Awaitable

import structlog
from websockets.asyncio.client import connect as ws_connect

from app.core.config import settings

log = structlog.get_logger(__name__)

AGENT_WS_URL = "wss://api.elevenlabs.io/v1/convai/conversation"


# ──────────────────────────────────────────────────────────────────────────────
# Connection
# ──────────────────────────────────────────────────────────────────────────────

async def connect_agent(
    agent_id: str,
    system_prompt: str = "",
    first_message: str = "",
) -> tuple:
    """
    Open a WebSocket connection to ElevenLabs Conversational Agent.

    Returns (ws, conversation_id).

    The agent handles STT + LLM + TTS internally.
    Audio in:  raw mulaw 8kHz bytes (from Twilio)
    Audio out: pcm_16000 bytes by default — see audio_utils.py for conversion.

    NOTE: To avoid conversion overhead, set your ElevenLabs Agent's
    output audio format to 'ulaw_8000' in the dashboard (Agent Settings →
    Advanced → Output Audio Format). Then no conversion is needed.
    """
    url = f"{AGENT_WS_URL}?agent_id={agent_id}"

    ws = await ws_connect(
        url,
        additional_headers={"xi-api-key": settings.ELEVENLABS_API_KEY},
        ping_interval=None,   # Agent manages its own keepalive via ping/pong
        ping_timeout=None,
    )

    # ── Wait for session_started ──────────────────────────────────────────────
    raw = await ws.recv()
    data = json.loads(raw)
    conversation_id = None

    if data.get("type") == "conversation_initiation_metadata":
        meta = data.get("conversation_initiation_metadata_event", {})
        conversation_id = meta.get("conversation_id")
        log.info("agent_session_started", conversation_id=conversation_id)

    elif data.get("type") in ("auth_error", "error"):
        raise ConnectionError(f"Agent connection failed: {data}")

    # ── Send optional config overrides ────────────────────────────────────────
    # Use this to inject a dynamic system prompt or opening message
    # without re-creating the agent in the dashboard.
    if system_prompt or first_message:
        override: dict = {
            "type": "conversation_initiation_client_data",
            "conversation_config_override": {
                "agent": {}
            },
        }
        # if system_prompt:
        #     override["conversation_config_override"]["agent"]["prompt"] = {
        #         "prompt": system_prompt
        #     }
        # if first_message:
        #     override["conversation_config_override"]["agent"]["first_message"] = first_message

        await ws.send(json.dumps(override))
        log.info("agent_config_override_sent")

    return ws, conversation_id


# ──────────────────────────────────────────────────────────────────────────────
# Sending audio to agent
# ──────────────────────────────────────────────────────────────────────────────

async def send_audio_chunk(ws, audio_bytes: bytes) -> None:
    """
    Send a raw audio chunk (mulaw 8kHz from Twilio) to the agent.
    The agent handles VAD and STT internally — just stream bytes continuously.
    """
    await ws.send(json.dumps({
        "user_audio_chunk": base64.b64encode(audio_bytes).decode()
    }))


# ──────────────────────────────────────────────────────────────────────────────
# Sending tool results back to agent
# ──────────────────────────────────────────────────────────────────────────────

async def send_tool_result(ws, tool_call_id: str, result: str) -> None:
    """
    Acknowledge a tool_call event back to the agent.
    Must be called after handling every tool_call, or the agent will stall.
    """
    await ws.send(json.dumps({
        "type": "tool_result",
        "tool_call_id": tool_call_id,
        "result": result,
    }))


# ──────────────────────────────────────────────────────────────────────────────
# Event loop (receive from agent)
# ──────────────────────────────────────────────────────────────────────────────

async def receive_agent_events(
    ws,
    on_audio:      Callable[[bytes], Awaitable[None]],
    on_tool_call:  Callable[[dict], Awaitable[None]],
    on_transcript: Callable[[str, str], Awaitable[None]] | None = None,
    call_log=None,
) -> None:
    """
    Continuously receive events from the agent and dispatch to callbacks.

    Events handled:
      audio          → decoded bytes forwarded to on_audio (send to Twilio)
      agent_response → agent's text reply (optional logging)
      user_transcript→ what the user said (optional logging)
      tool_call      → forward to on_tool_call (your business logic)
      interruption   → agent was barged in (no action needed)
      ping           → respond with pong (keepalive)
      error/auth_error → log and exit

    Exits cleanly on WebSocket close or CancelledError.
    """
    _log = call_log or log

    try:
        async for raw in ws:
            data       = json.loads(raw)
            event_type = data.get("type")

            # ── Audio from TTS ────────────────────────────────────────────────
            if event_type == "audio":
                audio_b64 = (
                    data.get("audio_event", {}).get("audio_base_64", "")
                )
                if audio_b64:
                    audio_bytes = base64.b64decode(audio_b64)
                    try:
                        await on_audio(audio_bytes)
                    except Exception as e:
                        _log.warning("agent_on_audio_failed", error=str(e))

            # ── Agent text response ───────────────────────────────────────────
            elif event_type == "agent_response":
                text = (
                    data.get("agent_response_event", {})
                        .get("agent_response", "")
                )
                _log.info("agent_response", preview=text[:120])
                if on_transcript and text:
                    await on_transcript("agent", text)

            # ── User speech transcript ────────────────────────────────────────
            elif event_type == "user_transcript":
                text = (
                    data.get("user_transcription_event", {})
                        .get("user_transcript", "")
                )
                _log.info("user_transcript", preview=text[:120])
                if on_transcript and text:
                    await on_transcript("user", text)

            # ── Tool call (e.g. transfer_to_human) ───────────────────────────
            elif event_type == "tool_call":
                tool_event = data.get("tool_call_event", {})
                _log.info(
                    "tool_call_received",
                    tool=tool_event.get("tool_name"),
                    tool_call_id=tool_event.get("tool_call_id"),
                )
                try:
                    await on_tool_call(tool_event)
                except Exception as e:
                    _log.error("tool_call_handler_failed", error=str(e))

            # ── Barge-in signal ───────────────────────────────────────────────
            elif event_type == "interruption":
                _log.info("agent_interrupted_by_user")

            # ── Keepalive ping → must respond with pong ───────────────────────
            elif event_type == "ping":
                event_id = data.get("ping_event", {}).get("event_id")
                await ws.send(json.dumps({
                    "type": "pong",
                    "event_id": event_id,
                }))

            # ── Errors ────────────────────────────────────────────────────────
            elif event_type in ("error", "auth_error", "quota_exceeded"):
                _log.error("agent_error_event", data=data)
                break

            else:
                _log.debug("agent_unknown_event", event_type=event_type)

    except asyncio.CancelledError:
        _log.info("agent_event_loop_cancelled")
        raise

    except Exception as e:
        _log.warning("agent_event_loop_error", error=str(e))


# ──────────────────────────────────────────────────────────────────────────────
# Teardown
# ──────────────────────────────────────────────────────────────────────────────

async def close_agent(ws) -> None:
    """Gracefully close the agent WebSocket."""
    try:
        await ws.close()
        log.info("agent_ws_closed")
    except Exception as e:
        log.debug("agent_ws_close_failed", error=str(e))