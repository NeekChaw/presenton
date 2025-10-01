import asyncio
import json
import os
import random
import uuid
from typing import Annotated, List, Optional
from fastapi import APIRouter, Body, Depends, HTTPException
from fastapi.responses import StreamingResponse
from sqlalchemy import delete
from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import select
from models.generate_presentation_request import GeneratePresentationRequest
from models.create_presentation_request import CreatePresentationRequest
from models.presentation_and_path import PresentationPathAndEditPath
from models.presentation_from_template import GetPresentationUsingTemplateRequest
from models.presentation_outline_model import (
    PresentationOutlineModel,
    SlideOutlineModel,
)
from models.pptx_models import PptxPresentationModel
from models.presentation_layout import PresentationLayoutModel
from models.presentation_structure_model import PresentationStructureModel
from models.presentation_with_slides import (
    PresentationWithSlides,
)

from services.documents_loader import DocumentsLoader
from services.score_based_chunker import ScoreBasedChunker
from utils.get_layout_by_name import get_layout_by_name
from services.image_generation_service import ImageGenerationService
from utils.dict_utils import deep_update
from utils.export_utils import export_presentation
from utils.llm_calls.generate_presentation_outlines import generate_ppt_outline
from models.sql.slide import SlideModel
from models.sse_response import SSECompleteResponse, SSEErrorResponse, SSEResponse

from services.database import get_async_session
from services.temp_file_service import TEMP_FILE_SERVICE
from models.sql.presentation import PresentationModel
from services.pptx_presentation_creator import PptxPresentationCreator
from utils.asset_directory_utils import get_exports_directory, get_images_directory
from utils.llm_calls.generate_presentation_structure import (
    generate_presentation_structure,
)
from utils.llm_calls.generate_slide_content import (
    get_slide_content_from_type_and_outline,
)
from utils.process_slides import (
    process_slide_add_placeholder_assets,
    process_slide_and_fetch_assets,
)
import uuid


PRESENTATION_ROUTER = APIRouter(prefix="/presentation", tags=["Presentation"])


@PRESENTATION_ROUTER.get("", response_model=PresentationWithSlides)
async def get_presentation(
    id: uuid.UUID, sql_session: AsyncSession = Depends(get_async_session)
):
    presentation = await sql_session.get(PresentationModel, id)
    if not presentation:
        raise HTTPException(404, "Presentation not found")
    slides = await sql_session.scalars(
        select(SlideModel)
        .where(SlideModel.presentation == id)
        .order_by(SlideModel.index)
    )
    # Convert SQLAlchemy objects to Pydantic models and add required user field
    presentation_data = presentation.model_dump()
    presentation_data["user"] = uuid.uuid4()  # Add default user UUID
    slides_data = [slide.model_dump() for slide in slides]

    return PresentationWithSlides(
        **presentation_data,
        slides=slides_data,
    )


@PRESENTATION_ROUTER.delete("", status_code=204)
async def delete_presentation(
    id: uuid.UUID, sql_session: AsyncSession = Depends(get_async_session)
):
    presentation = await sql_session.get(PresentationModel, id)
    if not presentation:
        raise HTTPException(404, "Presentation not found")

    await sql_session.delete(presentation)
    await sql_session.commit()


@PRESENTATION_ROUTER.get("/all", response_model=List[PresentationWithSlides])
async def get_all_presentations(sql_session: AsyncSession = Depends(get_async_session)):
    try:
        presentations = await sql_session.scalars(select(PresentationModel))
        presentations_with_slides = []

        for presentation in presentations:
            first_slide = await sql_session.scalar(
                select(SlideModel)
                .where(SlideModel.presentation == presentation.id)
                .where(SlideModel.index == 0)
            )
            if first_slide:
                # Convert SQLAlchemy objects to Pydantic models and add required user field
                presentation_data = presentation.model_dump()
                presentation_data["user"] = uuid.uuid4()  # Add default user UUID
                presentation_with_slides = PresentationWithSlides(
                    **presentation_data,
                    slides=[first_slide.model_dump()],
                )
                presentations_with_slides.append(presentation_with_slides)

        return presentations_with_slides
    except Exception as e:
        print(f"Error in get_all_presentations: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")


@PRESENTATION_ROUTER.post("/create", response_model=PresentationModel)
async def create_presentation(
    request: CreatePresentationRequest,
    sql_session: AsyncSession = Depends(get_async_session),
):
    presentation_id = uuid.uuid4()

    presentation = PresentationModel(
        id=presentation_id,
        content=request.content,
        n_slides=request.n_slides,
        language=request.language,
        file_paths=request.file_paths,
        tone=request.tone,
        verbosity=request.verbosity,
        instructions=request.instructions,
    )

    sql_session.add(presentation)
    await sql_session.commit()

    return presentation


@PRESENTATION_ROUTER.post("/prepare", response_model=PresentationModel)
async def prepare_presentation(
    presentation_id: Annotated[uuid.UUID, Body()],
    outlines: Annotated[List[SlideOutlineModel], Body()],
    layout: Annotated[PresentationLayoutModel, Body()],
    title: Annotated[Optional[str], Body()] = None,
    sql_session: AsyncSession = Depends(get_async_session),
):
    if not outlines:
        raise HTTPException(status_code=400, detail="Outlines are required")

    presentation = await sql_session.get(PresentationModel, presentation_id)
    if not presentation:
        raise HTTPException(status_code=404, detail="Presentation not found")

    presentation_outline_model = PresentationOutlineModel(slides=outlines)

    total_slide_layouts = len(layout.slides)
    total_outlines = len(outlines)

    if layout.ordered:
        presentation_structure = layout.to_presentation_structure()
    else:
        presentation_structure: PresentationStructureModel = (
            await generate_presentation_structure(
                presentation_outline=presentation_outline_model,
                presentation_layout=layout,
                instructions=presentation.instructions,
            )
        )

    presentation_structure.slides = presentation_structure.slides[: len(outlines)]
    for index in range(total_outlines):
        random_slide_index = random.randint(0, total_slide_layouts - 1)
        if index >= total_outlines:
            presentation_structure.slides.append(random_slide_index)
            continue
        if presentation_structure.slides[index] >= total_slide_layouts:
            presentation_structure.slides[index] = random_slide_index

    sql_session.add(presentation)
    presentation.outlines = presentation_outline_model.model_dump(mode="json")
    presentation.title = title or presentation.title
    presentation.set_layout(layout)
    presentation.set_structure(presentation_structure)
    await sql_session.commit()

    return presentation


@PRESENTATION_ROUTER.get("/stream", response_model=PresentationWithSlides)
async def stream_presentation(
    presentation_id: uuid.UUID, sql_session: AsyncSession = Depends(get_async_session)
):
    print(f"Starting stream for presentation ID: {presentation_id}")
    presentation = await sql_session.get(PresentationModel, presentation_id)
    if not presentation:
        print(f"Presentation not found: {presentation_id}")
        raise HTTPException(status_code=404, detail="Presentation not found")

    print(f"Presentation found. Structure exists: {bool(presentation.structure)}")
    print(f"Outlines exist: {bool(presentation.outlines)}")

    if not presentation.structure:
        print("ERROR: Presentation not prepared for stream - no structure")
        raise HTTPException(
            status_code=400,
            detail="Presentation not prepared for stream",
        )
    if not presentation.outlines:
        print("ERROR: Outlines can not be empty")
        raise HTTPException(
            status_code=400,
            detail="Outlines can not be empty",
        )

    image_generation_service = ImageGenerationService(get_images_directory())

    async def inner():
        print("Starting inner() generator function")
        try:
            structure = presentation.get_structure()
            layout = presentation.get_layout()
            outline = presentation.get_presentation_outline()

            print(f"Structure loaded with {len(structure.slides)} slides")
            print(f"Layout: {layout.name}")
            print(f"Outline has {len(outline.slides)} slides")

            # These tasks will be gathered and awaited after all slides are generated
            async_assets_generation_tasks = []

            slides: List[SlideModel] = []
            print("Yielding initial chunk")
            yield SSEResponse(
                event="response",
                data=json.dumps({"type": "chunk", "chunk": '{ "slides": [ '}),
            ).to_string()
            for i, slide_layout_index in enumerate(structure.slides):
                print(f"Processing slide {i+1}/{len(structure.slides)}")
                slide_layout = layout.slides[slide_layout_index]

                try:
                    print(f"Generating content for slide {i+1}")
                    slide_content = await get_slide_content_from_type_and_outline(
                        slide_layout,
                        outline.slides[i],
                        presentation.language,
                        presentation.tone,
                        presentation.verbosity,
                        presentation.instructions,
                    )
                    print(f"Content generated for slide {i+1}")
                except HTTPException as e:
                    print(f"HTTPException while generating slide {i+1}: {e.detail}")
                    yield SSEErrorResponse(detail=e.detail).to_string()
                    return
                except Exception as e:
                    print(f"Unexpected error while generating slide {i+1}: {e}")
                    yield SSEErrorResponse(detail=f"Error generating slide {i+1}: {str(e)}").to_string()
                    return

                slide = SlideModel(
                    presentation=presentation_id,
                    layout_group=layout.name,
                    layout=slide_layout.id,
                    index=i,
                    speaker_note=slide_content.get("__speaker_note__", ""),
                    content=slide_content,
                )
                slides.append(slide)

                # This will mutate slide and add placeholder assets
                process_slide_add_placeholder_assets(slide)

                # This will mutate slide
                async_assets_generation_tasks.append(
                    process_slide_and_fetch_assets(image_generation_service, slide)
                )

                yield SSEResponse(
                    event="response",
                    data=json.dumps({"type": "chunk", "chunk": slide.model_dump_json()}),
                ).to_string()

            yield SSEResponse(
                event="response",
                data=json.dumps({"type": "chunk", "chunk": " ] }"}),
            ).to_string()
            print("Finished sending final chunk")

            # Save slides to database immediately
            await sql_session.execute(
                delete(SlideModel).where(SlideModel.presentation == presentation_id)
            )
            sql_session.add(presentation)
            sql_session.add_all(slides)
            await sql_session.commit()
            print("Slides saved to database")

            # Create response and send completion event
            presentation_data = presentation.model_dump()
            # Add default user UUID since PresentationModel doesn't have user field but PresentationWithSlides requires it
            presentation_data["user"] = uuid.uuid4()  # Use a default UUID for now

            # Convert SQLAlchemy objects to Pydantic models
            slides_data = [slide.model_dump() for slide in slides]

            response = PresentationWithSlides(
                **presentation_data,
                slides=slides_data,
            )
            print("About to send complete event...")

            yield SSECompleteResponse(
                key="presentation",
                value=response.model_dump(mode="json"),
            ).to_string()
            print("Complete event sent successfully")

            print("Generator function completed")

        except Exception as e:
            print(f"Error in inner() generator: {e}")
            import traceback
            traceback.print_exc()
            yield SSEErrorResponse(detail=f"Internal error: {str(e)}").to_string()

    return StreamingResponse(inner(), media_type="text/event-stream")


@PRESENTATION_ROUTER.patch("/update", response_model=PresentationWithSlides)
async def update_presentation(
    id: Annotated[uuid.UUID, Body()],
    n_slides: Annotated[Optional[int], Body()] = None,
    title: Annotated[Optional[str], Body()] = None,
    slides: Annotated[Optional[List[SlideModel]], Body()] = None,
    sql_session: AsyncSession = Depends(get_async_session),
):
    presentation = await sql_session.get(PresentationModel, id)
    if not presentation:
        raise HTTPException(status_code=404, detail="Presentation not found")

    presentation_update_dict = {}
    if n_slides:
        presentation_update_dict["n_slides"] = n_slides
    if title:
        presentation_update_dict["title"] = title

    if n_slides or title:
        presentation.sqlmodel_update(presentation_update_dict)

    if slides:
        # Just to make sure id is UUID
        for slide in slides:
            slide.presentation = uuid.UUID(slide.presentation)
            slide.id = uuid.UUID(slide.id)

        await sql_session.execute(
            delete(SlideModel).where(SlideModel.presentation == presentation.id)
        )
        sql_session.add_all(slides)

    await sql_session.commit()

    # Convert SQLAlchemy objects to Pydantic models and add required user field
    presentation_data = presentation.model_dump()
    presentation_data["user"] = uuid.uuid4()  # Add default user UUID
    slides_data = [slide.model_dump() for slide in slides] if slides else []

    return PresentationWithSlides(
        **presentation_data,
        slides=slides_data,
    )


@PRESENTATION_ROUTER.post("/export/pptx", response_model=str)
async def create_pptx(
    pptx_model: Annotated[PptxPresentationModel, Body()],
):
    temp_dir = TEMP_FILE_SERVICE.create_temp_dir()

    pptx_creator = PptxPresentationCreator(pptx_model, temp_dir)
    await pptx_creator.create_ppt()

    export_directory = get_exports_directory()
    pptx_path = os.path.join(
        export_directory, f"{pptx_model.name or uuid.uuid4()}.pptx"
    )
    pptx_creator.save(pptx_path)

    return pptx_path


@PRESENTATION_ROUTER.post("/generate", response_model=PresentationPathAndEditPath)
async def generate_presentation_api(
    request: GeneratePresentationRequest,
    sql_session: AsyncSession = Depends(get_async_session),
):
    presentation_id = uuid.uuid4()

    # 3. Generate Outlines
    presentation_outlines = None
    additional_context = ""

    # Process files
    if request.files:
        documents_loader = DocumentsLoader(file_paths=request.files)
        await documents_loader.load_documents()
        documents = documents_loader.documents
        if documents and len(documents) == 1:
            additional_context = documents[0]
            chunker = ScoreBasedChunker()
            try:
                chunks = await chunker.get_n_chunks(documents[0], request.n_slides)
                presentation_outlines = PresentationOutlineModel(
                    slides=[chunk.to_slide_outline() for chunk in chunks]
                )
            except Exception as e:
                pass

        elif documents:
            additional_context = "\n\n".join(documents)

    if not presentation_outlines:
        presentation_outlines_text = ""
        async for chunk in generate_ppt_outline(
            request.content,
            request.n_slides,
            request.language,
            additional_context,
            request.tone,
            request.verbosity,
            request.instructions,
            request.web_search,
        ):

            if isinstance(chunk, HTTPException):
                raise chunk

            presentation_outlines_text += chunk

        try:
            print(f"Attempting to parse presentation_outlines_text (length: {len(presentation_outlines_text)})")
            print(f"First 500 chars: {presentation_outlines_text[:500]!r}")
            if not presentation_outlines_text.strip():
                raise HTTPException(
                    status_code=400,
                    detail="Empty response from LLM. Please try again."
                )
            presentation_outlines_json = json.loads(presentation_outlines_text)
        except json.JSONDecodeError as e:
            print(f"JSON parsing error: {str(e)}")
            print(f"Raw response: {presentation_outlines_text!r}")
            raise HTTPException(
                status_code=400,
                detail=f"Failed to parse LLM response as JSON: {str(e)}. Raw response length: {len(presentation_outlines_text)}"
            )
        except Exception as e:
            print(f"Unexpected error during presentation generation: {str(e)}")
            print(f"Raw response: {presentation_outlines_text!r}")
            raise HTTPException(
                status_code=400,
                detail=f"Failed to generate presentation outlines: {str(e)}"
            )
        presentation_outlines = PresentationOutlineModel(**presentation_outlines_json)

    outlines = presentation_outlines.slides[: request.n_slides]
    total_outlines = len(outlines)

    print("-" * 40)
    print(f"Generated {total_outlines} outlines for the presentation")

    # 4. Parse Layouts
    layout_model = await get_layout_by_name(request.template)
    total_slide_layouts = len(layout_model.slides)

    # 5. Generate Structure
    if layout_model.ordered:
        presentation_structure = layout_model.to_presentation_structure()
    else:
        presentation_structure: PresentationStructureModel = (
            await generate_presentation_structure(
                presentation_outlines,
                layout_model,
                request.instructions,
            )
        )

    presentation_structure.slides = presentation_structure.slides[:total_outlines]
    for index in range(total_outlines):
        random_slide_index = random.randint(0, total_slide_layouts - 1)
        if index >= total_outlines:
            presentation_structure.slides.append(random_slide_index)
            continue
        if presentation_structure.slides[index] >= total_slide_layouts:
            presentation_structure.slides[index] = random_slide_index

    # 6. Create PresentationModel
    presentation = PresentationModel(
        id=presentation_id,
        content=request.content,
        n_slides=request.n_slides,
        language=request.language,
        outlines=presentation_outlines.model_dump(),
        layout=layout_model.model_dump(),
        structure=presentation_structure.model_dump(),
        tone=request.tone,
        verbosity=request.verbosity,
        instructions=request.instructions,
    )

    image_generation_service = ImageGenerationService(get_images_directory())
    async_assets_generation_tasks = []

    # 7. Generate slide content concurrently (batched), then build slides and fetch assets
    slides: List[SlideModel] = []

    slide_layout_indices = presentation_structure.slides
    slide_layouts = [layout_model.slides[idx] for idx in slide_layout_indices]

    # Schedule slide content generation and asset fetching in batches of 10
    batch_size = 10
    for start in range(0, len(slide_layouts), batch_size):
        end = min(start + batch_size, len(slide_layouts))

        print(f"Generating slides from {start} to {end}")

        # Generate contents for this batch concurrently
        content_tasks = [
            get_slide_content_from_type_and_outline(
                slide_layouts[i],
                outlines[i],
                request.language,
                request.tone,
                request.verbosity,
                request.instructions,
            )
            for i in range(start, end)
        ]
        batch_contents: List[dict] = await asyncio.gather(*content_tasks)

        # Build slides for this batch
        batch_slides: List[SlideModel] = []
        for offset, slide_content in enumerate(batch_contents):
            i = start + offset
            slide_layout = slide_layouts[i]
            slide = SlideModel(
                presentation=presentation_id,
                layout_group=layout_model.name,
                layout=slide_layout.id,
                index=i,
                speaker_note=slide_content.get("__speaker_note__"),
                content=slide_content,
            )
            slides.append(slide)
            batch_slides.append(slide)

        # Start asset fetch tasks for just-generated slides so they run while next batch is processed
        asset_tasks = [
            process_slide_and_fetch_assets(image_generation_service, slide)
            for slide in batch_slides
        ]
        async_assets_generation_tasks.extend(asset_tasks)

    # Run all asset tasks concurrently while batches may still be generating content
    generated_assets_list = await asyncio.gather(*async_assets_generation_tasks)
    generated_assets = []
    for assets_list in generated_assets_list:
        generated_assets.extend(assets_list)

    # 8. Save PresentationModel and Slides
    sql_session.add(presentation)
    sql_session.add_all(slides)
    sql_session.add_all(generated_assets)
    await sql_session.commit()

    # 9. Export
    presentation_and_path = await export_presentation(
        presentation_id, presentation.title or str(uuid.uuid4()), request.export_as
    )

    return PresentationPathAndEditPath(
        **presentation_and_path.model_dump(),
        edit_path=f"/presentation?id={presentation_id}",
    )


@PRESENTATION_ROUTER.post("/from-template", response_model=PresentationPathAndEditPath)
async def from_template(
    data: Annotated[GetPresentationUsingTemplateRequest, Body()],
    sql_session: AsyncSession = Depends(get_async_session),
):
    presentation = await sql_session.get(PresentationModel, data.presentation_id)
    if not presentation:
        raise HTTPException(status_code=404, detail="Presentation not found")
    slides = await sql_session.scalars(
        select(SlideModel).where(SlideModel.presentation == data.presentation_id)
    )

    new_presentation = presentation.get_new_presentation()
    new_slides = []
    for each_slide in slides:
        updated_content = None
        new_slide_data = list(filter(lambda x: x.index == each_slide.index, data.data))
        if new_slide_data:
            updated_content = deep_update(each_slide.content, new_slide_data[0].content)
        new_slides.append(
            each_slide.get_new_slide(new_presentation.id, updated_content)
        )

    sql_session.add(new_presentation)
    sql_session.add_all(new_slides)
    await sql_session.commit()

    presentation_and_path = await export_presentation(
        new_presentation.id, new_presentation.title or str(uuid.uuid4()), data.export_as
    )

    return PresentationPathAndEditPath(
        **presentation_and_path.model_dump(),
        edit_path=f"/presentation?id={new_presentation.id}",
    )
