import asyncio
import structlog

from app.services.transcript_db import (
    save_call_transcript_to_db_sync
)

log = structlog.get_logger()


async def transcript_db_worker(queue):

    while True:

        item = await queue.get()

        try:

            await asyncio.to_thread(
                save_call_transcript_to_db_sync,
                item["call_id"],
                item["speaker"],
                item["text"],
                item["is_final"],
            )

        except Exception as e:

            log.error(
                "transcript_save_error",
                error=str(e)
            )

        finally:
            queue.task_done()