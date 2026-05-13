from aiohttp_retry import List
from fastapi import APIRouter, File, UploadFile, HTTPException, Depends
from app.schema.kb_Schema import (
    KBAddRequest,
    KBSearchRequest,
    KBSearchResponse
)
from app.services.kb_service import add_to_kb, search_kb
from app.services.file_parsing_service import extract_text_from_file, ExtractionError
from app.db.database import get_db
from app.core.deps import get_current_user

router = APIRouter(prefix="/kb", tags=["Knowledge Base"])


@router.post("/upload_document")
def add_data( files: UploadFile = File(... ), user_id: int= Depends(get_current_user)):
    user_id = int(user_id.id)
    try:
        result = extract_text_from_file(files)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except ExtractionError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception:
        raise HTTPException(status_code=500, detail="Failed to process uploaded file")

    result = add_to_kb(result, user_id=user_id)
    
    return {"message": "Document added to KB", "details": result}



# @router.post("/search", response_model=KBSearchResponse)
# def search(payload: KBSearchRequest):
#     results = search_kb(payload.query, payload.top_k)
#     return {"results": results}
