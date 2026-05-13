# tts_keepalive
import asyncio
import json

async def tts_keepalive_loop(
    tts_ws_holder,
    tts_lock,
    tts_state,
):

    while tts_state.get("running", True):

        await asyncio.sleep(10)

        if tts_state.get("speaking"):
            continue

        async with tts_lock:
            ws = tts_ws_holder.get("ws")

        if ws is None:
            continue

        try:
            await ws.send(json.dumps({
                "text": " ",
                "try_trigger_generation": False,
            }))

        except Exception:

            async with tts_lock:
                tts_ws_holder["ws"] = None