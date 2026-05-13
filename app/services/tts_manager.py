# tts_manager
import json
from typing import Callable, Awaitable

from app.services.tts_service import open_elevenlabs_stream
from store.calling_may_12 import SYSTEM_PROMPT
async def start_tts_stream():
        nonlocal tts_pump_task
        ws = await open_elevenlabs_stream(
            system_prompt=dynamic_one_liner or SYSTEM_PROMPT
        )
        async with tts_lock:
            tts_ws_holder["ws"] = ws

        async def pump_and_reset():
            try:
                await pump_audio(ws, send_audio_to_twilio)
            finally:
                # Always clear the stale ws reference when pump exits
                async with tts_lock:
                    if tts_ws_holder.get("ws") is ws:
                        tts_ws_holder["ws"] = None
                call_log.info("tts_pump_exited")

        tts_pump_task = asyncio.create_task(pump_and_reset())
        set_tts_pump_task(tts_pump_task)



async def restart_tts_stream():
        """Cancel current TTS stream and open a fresh one (barge-in)."""
        nonlocal tts_pump_task
        call_log.info("barge_in_tts_restart")

        old_pump = tts_pump_task
        if old_pump and not old_pump.done():
            old_pump.cancel()
            with suppress(asyncio.CancelledError):
                await old_pump

        async with tts_lock:
            old_ws = tts_ws_holder.get("ws")
            tts_ws_holder["ws"] = None

        if old_ws:
            with suppress(Exception):
                await old_ws.close()

        await start_tts_stream()



async def handle_barge_in():
        """
        Called when VAD detects the user speaking while TTS is playing.
        Cancels TTS immediately so the user's speech is heard.
        """
        if tts_state["speaking"]:
            tts_state["speaking"] = False
            stt_control["paused"] = False
            await restart_tts_stream()        
