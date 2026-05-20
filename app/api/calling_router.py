from fastapi import APIRouter, WebSocket

# from app.services.calling import handle_voice_call
from app.services.calling_agent.handle_voice_call import handle_voice_call
router = APIRouter()


@router.websocket("/ws/call")
async def voice_call(websocket: WebSocket):
    await handle_voice_call(websocket)
