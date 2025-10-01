import asyncio
import json
import uuid
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession
from openai import APIConnectionError, APITimeoutError
import httpx

from models.presentation_outline_model import PresentationOutlineModel
from models.sql.presentation import PresentationModel
from models.sse_response import (
    SSECompleteResponse,
    SSEErrorResponse,
    SSEResponse,
    SSEStatusResponse,
)
from services.temp_file_service import TEMP_FILE_SERVICE
from services.database import get_async_session
from services.documents_loader import DocumentsLoader
from services.score_based_chunker import ScoreBasedChunker
from utils.llm_calls.generate_presentation_outlines import generate_ppt_outline

OUTLINES_ROUTER = APIRouter(prefix="/outlines", tags=["Outlines"])


@OUTLINES_ROUTER.get("/stream")
async def stream_outlines(
    presentation_id: uuid.UUID, sql_session: AsyncSession = Depends(get_async_session)
):
    presentation = await sql_session.get(PresentationModel, presentation_id)

    if not presentation:
        raise HTTPException(status_code=404, detail="Presentation not found")

    temp_dir = TEMP_FILE_SERVICE.create_temp_dir()

    async def inner():
        yield SSEStatusResponse(
            status="Generating presentation outlines..."
        ).to_string()

        presentation_outlines = None
        additional_context = ""
        if presentation.file_paths:
            documents_loader = DocumentsLoader(file_paths=presentation.file_paths)
            await documents_loader.load_documents(temp_dir)
            documents = documents_loader.documents
            if documents and len(documents) == 1:
                additional_context = documents[0]
                chunker = ScoreBasedChunker()
                try:
                    chunks = await chunker.get_n_chunks(
                        documents[0], presentation.n_slides
                    )
                    presentation_outlines = PresentationOutlineModel(
                        slides=[chunk.to_slide_outline() for chunk in chunks]
                    )
                except Exception as e:
                    pass

            elif documents:
                additional_context = "\n\n".join(documents)

        if not presentation_outlines:
            presentation_outlines_text = ""
            max_retries = 3
            retry_delay = 2

            for attempt in range(max_retries):
                try:
                    async for chunk in generate_ppt_outline(
                        presentation.content,
                        presentation.n_slides,
                        presentation.language,
                        additional_context,
                        presentation.tone,
                        presentation.verbosity,
                        presentation.instructions,
                        True,
                    ):
                        # Give control to the event loop
                        await asyncio.sleep(0)

                        if isinstance(chunk, HTTPException):
                            yield SSEErrorResponse(detail=chunk.detail).to_string()
                            return

                        yield SSEResponse(
                            event="response",
                            data=json.dumps({"type": "chunk", "chunk": chunk}),
                        ).to_string()

                        presentation_outlines_text += chunk

                    # If we get here, generation was successful
                    break

                except (APIConnectionError, APITimeoutError, httpx.ConnectError, httpx.TimeoutException) as e:
                    print(f"Connection attempt {attempt + 1}/{max_retries} failed: {str(e)}")
                    if attempt == max_retries - 1:
                        yield SSEErrorResponse(
                            detail=f"Failed to connect to the server after {max_retries} attempts. Please try again later."
                        ).to_string()
                        return

                    # Exponential backoff with status update
                    yield SSEStatusResponse(
                        status=f"Connection attempt {attempt + 1} failed, retrying in {retry_delay}s..."
                    ).to_string()

                    await asyncio.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff

                except Exception as e:
                    print(f"Unexpected error during presentation generation: {str(e)}")
                    yield SSEErrorResponse(
                        detail=f"An unexpected error occurred: {str(e)}"
                    ).to_string()
                    return

            try:
                print(f"Trying to parse JSON (length: {len(presentation_outlines_text)}): {presentation_outlines_text[:500]}...")
                if not presentation_outlines_text.strip():
                    raise HTTPException(
                        status_code=400,
                        detail="Empty response from LLM. Please try again."
                    )
                presentation_outlines_json = json.loads(presentation_outlines_text)
            except json.JSONDecodeError as e:
                print(f"JSON parsing error: {str(e)}")
                print(f"Raw response: {presentation_outlines_text}")
                raise HTTPException(
                    status_code=400,
                    detail=f"Failed to parse LLM response as JSON: {str(e)}. Raw response length: {len(presentation_outlines_text)}",
                )
            except Exception as e:
                print(f"Unexpected error during outline generation: {str(e)}")
                print(f"Raw response: {presentation_outlines_text}")
                raise HTTPException(
                    status_code=400,
                    detail=f"Failed to generate presentation outlines: {str(e)}",
                )

            presentation_outlines = PresentationOutlineModel(
                **presentation_outlines_json
            )

        presentation_outlines.slides = presentation_outlines.slides[
            : presentation.n_slides
        ]

        presentation.outlines = presentation_outlines.model_dump()
        presentation.title = (
            presentation_outlines.slides[0]
            .content[:50]
            .replace("#", "")
            .replace("/", "")
            .replace("\\", "")
            .replace("\n", "")
        )

        sql_session.add(presentation)
        await sql_session.commit()

        yield SSECompleteResponse(
            key="presentation", value=presentation.model_dump(mode="json")
        ).to_string()

    return StreamingResponse(inner(), media_type="text/event-stream")
