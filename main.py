from fastapi import FastAPI, Request
from app.api.twilio_router.call import router as call_router
from app.api.twilio_router.number import router as number_router
from fastapi.middleware.cors import CORSMiddleware
from app.api.kb import router as kb_router
from app.api.calling_router import router as calling_router
from app.api.auth import router as auth_router
from app.api.upload import router as upload_router
from fastapi.responses import JSONResponse
from app.core.exceptions import AppException

app = FastAPI(title="calling genai", version="0.1.0")

from app.db.database import Base, engine

Base.metadata.create_all(bind=engine)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(AppException)
async def app_exception_handler(request: Request, exc: AppException):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.message
        }
    )

app.include_router(call_router)
app.include_router(number_router)
app.include_router(kb_router)
app.include_router(calling_router)
app.include_router(auth_router)
app.include_router(upload_router)

@app.get("/")
def root():
    return {"status": "Running"}