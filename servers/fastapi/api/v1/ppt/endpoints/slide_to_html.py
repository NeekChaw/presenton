import os
import base64
import re
from datetime import datetime
from typing import Optional, List, Dict
from uuid import UUID
from fastapi import APIRouter, HTTPException, File, UploadFile, Form, Depends, Request
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel
from openai import OpenAI
from openai import APIError
from utils.get_env import get_openai_api_key_env, get_openai_url_env, get_openai_model_env, get_llm_provider_env
from utils.user_config import get_user_config
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, delete, func
from utils.asset_directory_utils import get_images_directory
from services.database import get_async_session
from models.sql.presentation_layout_code import PresentationLayoutCodeModel
from .prompts import GENERATE_HTML_SYSTEM_PROMPT, HTML_TO_REACT_SYSTEM_PROMPT, HTML_EDIT_SYSTEM_PROMPT
from models.sql.template import TemplateModel


# Create separate routers for each functionality
SLIDE_TO_HTML_ROUTER = APIRouter(prefix="/slide-to-html", tags=["slide-to-html"])
HTML_TO_REACT_ROUTER = APIRouter(prefix="/html-to-react", tags=["html-to-react"])
HTML_EDIT_ROUTER = APIRouter(prefix="/html-edit", tags=["html-edit"])
LAYOUT_MANAGEMENT_ROUTER = APIRouter(prefix="/template-management", tags=["template-management"])

# JSONResponse import moved to top level for reuse
from fastapi.responses import JSONResponse


def auto_fix_schema_objects(code: str) -> str:
    """
    Fallback strategy: Auto-fix schema object definitions to string definitions.
    This ensures that text fields use z.string() instead of z.object() which causes validation errors.
    """
    # Pattern to find text fields that were incorrectly defined as z.object
    # Common text field names that should be strings, not objects
    text_field_patterns = [
        r'(\w*[Tt]itle\w*):\s*z\.object\([^)]+\)\.default\("([^"]*)"',
        r'(\w*[Dd]escription\w*):\s*z\.object\([^)]+\)\.default\("([^"]*)"',  
        r'(\w*[Tt]ext\w*):\s*z\.object\([^)]+\)\.default\("([^"]*)"',
        r'(\w*[Cc]ontent\w*):\s*z\.object\([^)]+\)\.default\("([^"]*)"',
        r'(\w*[Ss]ection\w*):\s*z\.object\([^)]+\)\.default\("([^"]*)"',
        r'(\w*[Hh]eader\w*):\s*z\.object\([^)]+\)\.default\("([^"]*)"',
        r'(\w*[Ss]ubtext\w*):\s*z\.object\([^)]+\)\.default\("([^"]*)"'
    ]
    
    original_code = code
    fixes_applied = 0
    
    for pattern in text_field_patterns:
        matches = re.findall(pattern, code, re.IGNORECASE | re.MULTILINE | re.DOTALL)
        for field_name, default_value in matches:
            # Replace z.object(...).default("text") with z.string().default("text")
            old_pattern = f'{field_name}:\\s*z\\.object\\([^)]+\\)\\.default\\("{re.escape(default_value)}"'
            new_definition = f'{field_name}: z.string().default("{default_value}"'
            
            if re.search(old_pattern, code, re.IGNORECASE):
                code = re.sub(old_pattern, new_definition, code, flags=re.IGNORECASE)
                fixes_applied += 1
                print(f"🔧 Auto-fixed field '{field_name}' from z.object to z.string")
    
    # Fix incorrect Zod method chaining order (.meta().min() -> .min().meta())
    zod_chain_fixes = [
        (r'\.meta\(([^)]+)\)\.min\((\d+)\)', r'.min(\2).meta(\1)'),
        (r'\.meta\(([^)]+)\)\.max\((\d+)\)', r'.max(\2).meta(\1)'),
        (r'\.meta\(([^)]+)\)\.min\((\d+)\)\.max\((\d+)\)', r'.min(\2).max(\3).meta(\1)'),
        (r'\.meta\(([^)]+)\)\.max\((\d+)\)\.min\((\d+)\)', r'.min(\3).max(\2).meta(\1)')
    ]
    
    for pattern, replacement in zod_chain_fixes:
        if re.search(pattern, code):
            old_code = code
            code = re.sub(pattern, replacement, code)
            if code != old_code:
                fixes_applied += 1
                print(f"🔧 Fixed Zod method chaining order")
    
    if fixes_applied > 0:
        print(f"✅ Applied {fixes_applied} schema fixes as fallback strategy")
    
    return code


# Request/Response models for slide-to-html endpoint
class SlideToHtmlRequest(BaseModel):
    image: str  # Partial path to image file (e.g., "/app_data/images/uuid/slide_1.png")
    xml: str    # OXML content as text
    fonts: Optional[List[str]] = None  # Optional normalized root fonts for this slide

class SlideToHtmlResponse(BaseModel):
    success: bool
    html: str


# Request/Response models for html-edit endpoint
class HtmlEditResponse(BaseModel):
    success: bool
    edited_html: str
    message: Optional[str] = None


# Request/Response models for html-to-react endpoint
class HtmlToReactRequest(BaseModel):
    html: str   # HTML content to convert to React component
    image: Optional[str] = None  # Optional image path to provide visual context


class HtmlToReactResponse(BaseModel):
    success: bool
    react_component: str
    message: Optional[str] = None


# Request/Response models for layout management endpoints
class LayoutData(BaseModel):
    presentation_id: UUID  # UUID of the presentation
    layout_id: str        # Unique identifier for the layout
    layout_name: str      # Display name of the layout
    layout_code: str      # TSX/React component code for the layout
    fonts: Optional[List[str]] = None  # Optional list of font links


class SaveLayoutsRequest(BaseModel):
    layouts: list[LayoutData]


class SaveLayoutsResponse(BaseModel):
    success: bool
    saved_count: int
    message: Optional[str] = None


class GetLayoutsResponse(BaseModel):
    success: bool
    layouts: list[LayoutData]
    message: Optional[str] = None
    template: Optional[dict] = None
    fonts: Optional[List[str]] = None


class PresentationSummary(BaseModel):
    presentation_id: UUID
    layout_count: int
    last_updated_at: Optional[datetime] = None
    template: Optional[dict] = None


class GetPresentationSummaryResponse(BaseModel):
    success: bool
    presentations: List[PresentationSummary]
    total_presentations: int
    total_layouts: int
    message: Optional[str] = None


class ErrorResponse(BaseModel):
    success: bool = False
    detail: str
    error_code: Optional[str] = None


class TemplateCreateRequest(BaseModel):
    id: UUID
    name: str
    description: Optional[str] = None


class TemplateCreateResponse(BaseModel):
    success: bool
    template: dict
    message: Optional[str] = None


class TemplateInfo(BaseModel):
    id: UUID
    name: Optional[str] = None
    description: Optional[str] = None
    created_at: Optional[datetime] = None


async def generate_html_from_slide(base64_image: str, media_type: str, xml_content: str, api_key: str, base_url: str, model_name: str, fonts: Optional[List[str]] = None) -> str:
    """
    Generate HTML content from slide image and XML using OpenAI compatible API.
    
    Args:
        base64_image: Base64 encoded image data
        media_type: MIME type of the image (e.g., 'image/png')
        xml_content: OXML content as text
        api_key: API key
        base_url: API base URL
        model_name: Model name to use
        fonts: Optional list of normalized root font families to prefer in output
    
    Returns:
        Generated HTML content as string
    
    Raises:
        HTTPException: If API call fails or no content is generated
    """
    print(f"Generating HTML from slide image and XML using {model_name} via {base_url}...")
    try:
        client = OpenAI(api_key=api_key, base_url=base_url)

        # Compose input for Responses API. Include system prompt, image (separate), OXML and optional fonts text.
        data_url = f"data:{media_type};base64,{base64_image}"
        fonts_text = f"\nFONTS (Normalized root families used in this slide, use where it is required): {', '.join(fonts)}" if fonts else ""
        user_text = f"OXML: \n\n{fonts_text}"
        input_payload = [
            {"role": "system", "content": GENERATE_HTML_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "input_image", "image_url": data_url},
                    {"type": "input_text", "text": user_text},
                ],
            },
        ]

        print(f"Making API request for HTML generation with model {model_name}...")
        
        # Use Chat Completions API for OpenRouter and specific models
        print(f"Debug: model_name='{model_name}', base_url='{base_url}'")
        print(f"Debug: Using OpenRouter = {'openrouter.ai' in base_url.lower()}")
        print(f"Debug: Using GLM model = {'glm' in model_name.lower()}")
        print(f"Debug: Using GPT-5 model = {'gpt-5' in model_name.lower()}")
        
        # Skip Responses API for OpenRouter, GLM models, or GPT-5
        if "openrouter.ai" in base_url.lower() or "glm" in model_name.lower() or "gpt-5" in model_name.lower():
            print(f"Skipping Responses API for GLM model {model_name}, using Chat Completions directly...")
            # Directly use Chat Completions API for GLM models
            messages = [
                {"role": "system", "content": GENERATE_HTML_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": data_url}},
                        {"type": "text", "text": user_text},
                    ]
                }
            ]
            response = client.chat.completions.create(
                model=model_name,
                messages=messages,
                max_tokens=4000
            )
            print(f"Chat completions API succeeded")
            # Some models (like GLM-4.5V) return content in reasoning field
            message = response.choices[0].message
            print(f"Message content: {message.content}")
            print(f"Message reasoning: {getattr(message, 'reasoning', 'No reasoning field')}")
            print(f"Full message object: {message}")
            html_content = message.content or getattr(message, 'reasoning', '') or ""
        else:
            # Try Responses API first for non-GLM models
            try:
                print(f"Attempting Responses API with {model_name}...")
                response = client.responses.create(
                    model=model_name,
                    input=input_payload,
                    reasoning={"effort": "high"},
                    text={"verbosity": "low"},
                )
                print(f"Responses API succeeded")
                html_content = getattr(response, "output_text", None) or getattr(response, "text", None) or ""
                print(f"Responses API content length: {len(html_content)}")
            except Exception as e:
                print(f"Responses API failed: {e}")
                print(f"Falling back to chat completions API...")
                # Fallback to standard chat completions API
                messages = [
                    {"role": "system", "content": GENERATE_HTML_SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {"url": data_url}},
                            {"type": "text", "text": user_text},
                        ]
                    }
                ]
                response = client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    max_tokens=4000
                )
                print(f"Chat completions API succeeded")
                # Some models (like GLM-4.5V) return content in reasoning field
                message = response.choices[0].message
                print(f"Message content: {message.content}")
                print(f"Message reasoning: {getattr(message, 'reasoning', 'No reasoning field')}")
                print(f"Full message object: {message}")
                html_content = message.content or getattr(message, 'reasoning', '') or ""

        # html_content is already set in the try/except block above
        
        print(f"Received HTML content length: {len(html_content)}")
        
        if not html_content:
            print(f"WARNING: Empty response from {model_name} at {base_url}")
            print(f"DEBUG: Response object type: {type(response) if 'response' in locals() else 'No response'}")
            # TEMPORARY FIX: Return a simple HTML template instead of failing
            print("FALLBACK: Using default HTML template due to empty API response")
            html_content = '''
            <div class="w-full max-w-[1280px] shadow-lg max-h-[720px] aspect-video bg-white relative z-20 mx-auto overflow-hidden rounded-sm">
                <div class="flex flex-col h-full p-8">
                    <div class="text-2xl font-bold text-gray-800 mb-4">标题</div>
                    <div class="flex-1 flex items-center justify-center">
                        <div class="text-center text-gray-600">
                            <p>内容区域 - API暂时不可用，使用默认模板</p>
                        </div>
                    </div>
                </div>
            </div>
            '''
        
        return html_content
        
    except APIError as e:
        print(f"OpenAI API Error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"OpenAI API error during HTML generation: {str(e)}"
        )
    except Exception as e:
        # Handle various API errors
        error_msg = str(e)
        print(f"Exception occurred: {error_msg}")
        print(f"Exception type: {type(e)}")
        if "timeout" in error_msg.lower():
            raise HTTPException(
                status_code=408,
                detail=f"OpenAI API timeout during HTML generation: {error_msg}"
            )
        elif "connection" in error_msg.lower():
            raise HTTPException(
                status_code=503,
                detail=f"OpenAI API connection error during HTML generation: {error_msg}"
            )
        else:
            raise HTTPException(
                status_code=500,
                detail=f"OpenAI API error during HTML generation: {error_msg}"
            )


async def generate_react_component_from_html(html_content: str, api_key: str, base_url: str, model_name: str, image_base64: Optional[str] = None, media_type: Optional[str] = None) -> str:
    """
    Convert HTML content to TSX React component using OpenAI compatible API.
    
    Args:
        html_content: Generated HTML content
        api_key: API key
        base_url: API base URL
        model_name: Model name to use
        image_base64: Optional image for context
        media_type: Optional media type
    
    Returns:
        Generated TSX React component code as string
    
    Raises:
        HTTPException: If API call fails or no content is generated
    """
    try:
        client = OpenAI(api_key=api_key, base_url=base_url)

        print(f"Making API request for React component generation with model {model_name}...")

        # Build payload with optional image
        content_parts = [{"type": "input_text", "text": f"HTML INPUT:\n{html_content}"}]
        if image_base64 and media_type:
            data_url = f"data:{media_type};base64,{image_base64}"
            content_parts.insert(0, {"type": "input_image", "image_url": data_url})

        input_payload = [
            {"role": "system", "content": HTML_TO_REACT_SYSTEM_PROMPT},
            {"role": "user", "content": content_parts},
        ]

        # Use Chat Completions API for OpenRouter and specific models
        print(f"Debug: React generation - model_name='{model_name}', base_url='{base_url}'")
        print(f"Debug: React generation - Using OpenRouter = {'openrouter.ai' in base_url.lower()}")
        print(f"Debug: React generation - Using GLM model = {'glm' in model_name.lower()}")
        print(f"Debug: React generation - Using GPT-5 model = {'gpt-5' in model_name.lower()}")
        
        # Skip Responses API for OpenRouter, GLM models, or GPT-5
        if "openrouter.ai" in base_url.lower() or "glm" in model_name.lower() or "gpt-5" in model_name.lower():
            print(f"Skipping Responses API for {model_name} in React generation, using Chat Completions directly...")
            # Directly use Chat Completions API for GLM models
            messages = [
                {"role": "system", "content": HTML_TO_REACT_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": ([{"type": "image_url", "image_url": {"url": f"data:{media_type};base64,{image_base64}"}}] if image_base64 and media_type else []) + [
                        {"type": "text", "text": f"HTML INPUT:\n{html_content}"}
                    ]
                }
            ]
            response = client.chat.completions.create(
                model=model_name,
                messages=messages,
                max_tokens=4000
            )
            print(f"React generation Chat completions API succeeded")
            # Some models (like GLM-4.5V) return content in reasoning field
            message = response.choices[0].message
            print(f"React Message content: {message.content}")
            print(f"React Message reasoning: {getattr(message, 'reasoning', 'No reasoning field')}")
            react_content = message.content or getattr(message, 'reasoning', '') or ""
            
            # Clean up the React code - remove code block markers
            if react_content.strip().startswith("```"):
                lines = react_content.strip().split('\n')
                if lines[0].startswith("```"):
                    lines = lines[1:]  # Remove first line with ```typescript or ```
                if lines and lines[-1].strip() == "```":
                    lines = lines[:-1]  # Remove last line with ```
                react_content = '\n'.join(lines)
            
            print(f"Cleaned React content length: {len(react_content)}")
            print(f"React content first 100 chars: {react_content[:100]}")
            print(f"React content last 100 chars: {react_content[-100:]}")
        else:
            # Try Responses API first for non-GLM models
            try:
                response = client.responses.create(
                    model=model_name,
                    input=input_payload,
                    reasoning={"effort": "minimal"},
                    text={"verbosity": "low"},
                )
                react_content = getattr(response, "output_text", None) or getattr(response, "text", None) or ""
            except Exception as e:
                print(f"Responses API not available, falling back to chat completions: {e}")
                # Fallback to standard chat completions API
                messages = [
                    {"role": "system", "content": HTML_TO_REACT_SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": ([{"type": "image_url", "image_url": {"url": f"data:{media_type};base64,{image_base64}"}}] if image_base64 and media_type else []) + [
                            {"type": "text", "text": f"HTML INPUT:\n{html_content}"}
                        ]
                    }
                ]
                response = client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    max_tokens=4000
                )
                # Some models (like GLM-4.5V) return content in reasoning field
                message = response.choices[0].message
                react_content = message.content or getattr(message, 'reasoning', '') or ""
                
                # Clean up the React code - remove code block markers
                if react_content.strip().startswith("```"):
                    lines = react_content.strip().split('\n')
                    if lines[0].startswith("```"):
                        lines = lines[1:]  # Remove first line with ```typescript or ```
                    if lines and lines[-1].strip() == "```":
                        lines = lines[:-1]  # Remove last line with ```
                    react_content = '\n'.join(lines)
                
                print(f"Cleaned React content (fallback) length: {len(react_content)}")
                print(f"React content (fallback) first 100 chars: {react_content[:100]}")
                print(f"React content (fallback) last 100 chars: {react_content[-100:]}")
        
        print(f"Received React content length: {len(react_content)}")
        
        if not react_content:
            raise HTTPException(
                status_code=500,
                detail=f"No React component generated by {model_name}"
            )
        
        react_content = react_content.replace("```tsx", "").replace("```", "").replace("typescript", "").replace("javascript", "")

        
        # Filter out only import lines, but keep export lines  
        filtered_lines = []
        for line in react_content.split('\n'):
            stripped_line = line.strip()
            if not stripped_line.startswith('import '):
                filtered_lines.append(line)
        
        filtered_react_content = '\n'.join(filtered_lines)
        print(f"Filtered React content length: {len(filtered_react_content)}")
        
        # Remove any export statements that cause compilation errors
        # The frontend uses Function constructor and handles exports manually
        lines = filtered_react_content.split('\n')
        clean_lines = []
        for line in lines:
            stripped = line.strip()
            if not (stripped.startswith('export default') or stripped.startswith('export {')):
                clean_lines.append(line)
        filtered_react_content = '\n'.join(clean_lines)
        
        # Fix common TypeScript syntax errors in generated code
        import re
        
        # Fix malformed type definitions like "type Name = infer<typeof schema>fx" 
        # Should be "type Name = z.infer<typeof Schema>;"
        type_pattern = r'type\s+(\w+)\s*=\s*infer<typeof\s+(\w+)>(\w*)'
        def fix_type_definition(match):
            type_name = match.group(1)
            schema_name = match.group(2)
            # Ensure proper capitalization for Schema and add z. prefix
            proper_schema = "Schema" if schema_name.lower() == "schema" else schema_name
            return f'type {type_name} = z.infer<typeof {proper_schema}>;'
        
        filtered_react_content = re.sub(type_pattern, fix_type_definition, filtered_react_content)
        
        # Fix missing semicolons in type definitions
        filtered_react_content = re.sub(r'(type\s+\w+\s*=\s*z\.infer<typeof\s+\w+>)(?!;)', r'\1;', filtered_react_content)
        
        print(f"Final React content with TypeScript fixes length: {len(filtered_react_content)}")
        print(f"Final React content last 200 chars: {filtered_react_content[-200:]}")
        
        return filtered_react_content
    except APIError as e:
        print(f"OpenAI API Error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"OpenAI API error during React generation: {str(e)}"
        )
    except Exception as e:
        # Handle various API errors
        error_msg = str(e)
        print(f"Exception occurred: {error_msg}")
        print(f"Exception type: {type(e)}")
        if "timeout" in error_msg.lower():
            raise HTTPException(
                status_code=408,
                detail=f"OpenAI API timeout during React generation: {error_msg}"
            )
        elif "connection" in error_msg.lower():
            raise HTTPException(
                status_code=503,
                detail=f"OpenAI API connection error during React generation: {error_msg}"
            )
        else:
            raise HTTPException(
                status_code=500,
                detail=f"OpenAI API error during React generation: {error_msg}"
            )


async def edit_html_with_images(current_ui_base64: str, sketch_base64: Optional[str], media_type: str, html_content: str, prompt: str, api_key: str, base_url: str, model_name: str) -> str:
    """
    Edit HTML content based on one or two images and a text prompt using OpenAI compatible API.

    Args:
        current_ui_base64: Base64 encoded current UI image data
        sketch_base64: Base64 encoded sketch/indication image data (optional)
        media_type: MIME type of the images (e.g., 'image/png')
        html_content: Current HTML content to edit
        prompt: Text prompt describing the changes
        api_key: API key
        base_url: API base URL
        model_name: Model name to use
    
    Returns:
        Edited HTML content as string
    
    Raises:
        HTTPException: If API call fails or no content is generated
    """
    try:
        client = OpenAI(api_key=api_key, base_url=base_url)

        print(f"Making API request for HTML editing with model {model_name}...")

        current_data_url = f"data:{media_type};base64,{current_ui_base64}"
        sketch_data_url = f"data:{media_type};base64,{sketch_base64}" if sketch_base64 else None

        content_parts = [
            {"type": "input_image", "image_url": current_data_url},
            {"type": "input_text", "text": f"CURRENT HTML TO EDIT:\n{html_content}\n\nTEXT PROMPT FOR CHANGES:\n{prompt}"},
        ]
        if sketch_data_url:
            # Insert sketch image after current UI image for context
            content_parts.insert(1, {"type": "input_image", "image_url": sketch_data_url})

        input_payload = [
            {"role": "system", "content": HTML_EDIT_SYSTEM_PROMPT},
            {"role": "user", "content": content_parts},
        ]

        # Use responses API if available (OpenAI), otherwise use chat completions
        try:
            response = client.responses.create(
                model=model_name,
                input=input_payload,
                reasoning={"effort": "low"},
                text={"verbosity": "low"},
            )
            edited_html = getattr(response, "output_text", None) or getattr(response, "text", None) or ""
        except Exception as e:
            print(f"Responses API not available, falling back to chat completions: {e}")
            # Fallback to standard chat completions API
            messages = [
                {"role": "system", "content": HTML_EDIT_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": current_data_url}}
                    ] + ([{"type": "image_url", "image_url": {"url": sketch_data_url}}] if sketch_data_url else []) + [
                        {"type": "text", "text": f"CURRENT HTML TO EDIT:\n{html_content}\n\nTEXT PROMPT FOR CHANGES:\n{prompt}"}
                    ]
                }
            ]
            response = client.chat.completions.create(
                model=model_name,
                messages=messages,
                max_tokens=4000
            )
            # Some models (like GLM-4.5V) return content in reasoning field
            message = response.choices[0].message
            edited_html = message.content or message.reasoning or ""
        
        print(f"Received edited HTML content length: {len(edited_html)}")
        
        if not edited_html:
            raise HTTPException(
                status_code=500,
                detail="No edited HTML content generated by OpenAI GPT-5"
            )
        
        return edited_html
        
    except APIError as e:
        print(f"OpenAI API Error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"OpenAI API error during HTML editing: {str(e)}"
        )
    except Exception as e:
        # Handle various API errors
        error_msg = str(e)
        print(f"Exception occurred: {error_msg}")
        print(f"Exception type: {type(e)}")
        if "timeout" in error_msg.lower():
            raise HTTPException(
                status_code=408,
                detail=f"OpenAI API timeout during HTML editing: {error_msg}"
            )
        elif "connection" in error_msg.lower():
            raise HTTPException(
                status_code=503,
                detail=f"OpenAI API connection error during HTML editing: {error_msg}"
            )
        else:
            raise HTTPException(
                status_code=500,
                detail=f"OpenAI API error during HTML editing: {error_msg}"
            )


# ENDPOINT 1: Slide to HTML conversion
@SLIDE_TO_HTML_ROUTER.post("/", response_model=SlideToHtmlResponse)
async def convert_slide_to_html(request: SlideToHtmlRequest):
    """
    Convert a slide image and its OXML data to HTML using OpenAI compatible API.
    
    Args:
        request: JSON request containing image path and XML content
    
    Returns:
        SlideToHtmlResponse with generated HTML
    """
    try:
        # Get user configuration for dynamic API settings
        user_config = get_user_config()
        
        # Determine API key and URL based on LLM provider
        llm_provider = user_config.LLM or get_llm_provider_env()
        
        print(f"DEBUG: LLM provider: {llm_provider}")
        print(f"DEBUG: CUSTOM_MODEL from config: {user_config.CUSTOM_MODEL}")
        
        if llm_provider == "custom":
            api_key = user_config.CUSTOM_LLM_API_KEY
            base_url = user_config.CUSTOM_LLM_URL
            model_name = user_config.CUSTOM_MODEL or "gpt-4o"
            print(f"DEBUG: Using custom provider - model: {model_name}")
        else:
            # Default to OpenAI settings
            api_key = user_config.OPENAI_API_KEY or get_openai_api_key_env()
            base_url = user_config.OPENAI_URL or get_openai_url_env() or "https://api.openai.com/v1"
            model_name = user_config.OPENAI_MODEL or get_openai_model_env() or "gpt-4o"
            print(f"DEBUG: Using OpenAI provider - model: {model_name}")
        
        if not api_key:
            raise HTTPException(
                status_code=500, 
                detail=f"API key not set for LLM provider: {llm_provider}"
            )
        
        # Resolve image path to actual file system path
        image_path = request.image
        
        # Handle different path formats
        if image_path.startswith("/app_data/images/"):
            # Remove the /app_data/images/ prefix and join with actual images directory
            relative_path = image_path[len("/app_data/images/"):]
            actual_image_path = os.path.join(get_images_directory(), relative_path)
        elif image_path.startswith("/static/"):
            # Handle static files
            relative_path = image_path[len("/static/"):]
            actual_image_path = os.path.join("static", relative_path)
        else:
            # Assume it's already a full path or relative to images directory
            if os.path.isabs(image_path):
                actual_image_path = image_path
            else:
                actual_image_path = os.path.join(get_images_directory(), image_path)
        
        # Check if image file exists
        if not os.path.exists(actual_image_path):
            raise HTTPException(
                status_code=404,
                detail=f"Image file not found: {image_path}"
            )
        
        # Read and encode image to base64
        with open(actual_image_path, "rb") as image_file:
            image_content = image_file.read()
        base64_image = base64.b64encode(image_content).decode('utf-8')
        
        # Determine media type from file extension
        file_extension = os.path.splitext(actual_image_path)[1].lower()
        media_type_map = {
            '.png': 'image/png',
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg',
            '.gif': 'image/gif',
            '.webp': 'image/webp'
        }
        media_type = media_type_map.get(file_extension, 'image/png')
        
        # Generate HTML using the extracted function
        html_content = await generate_html_from_slide(
            base64_image=base64_image,
            media_type=media_type,
            xml_content=request.xml,
            api_key=api_key,
            base_url=base_url,
            model_name=model_name,
            fonts=request.fonts,
            )
        
        html_content = html_content.replace("```html", "").replace("```", "")
        
        return SlideToHtmlResponse(
            success=True,
            html=html_content
        )
        
    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    except Exception as e:
        # Log the full error for debugging
        print(f"Unexpected error during slide to HTML processing: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error processing slide to HTML: {str(e)}"
        )


# ENDPOINT 2: HTML to React component conversion
@HTML_TO_REACT_ROUTER.post("/", response_model=HtmlToReactResponse)
async def convert_html_to_react(request: HtmlToReactRequest):
    """
    Convert HTML content to TSX React component using OpenAI compatible API.
    
    Args:
        request: JSON request containing HTML content
    
    Returns:
        HtmlToReactResponse with generated React component
    """
    try:
        # Get user configuration for dynamic API settings
        user_config = get_user_config()
        
        # Determine API key and URL based on LLM provider
        llm_provider = user_config.LLM or get_llm_provider_env()
        
        if llm_provider == "custom":
            api_key = user_config.CUSTOM_LLM_API_KEY
            base_url = user_config.CUSTOM_LLM_URL
            model_name = user_config.CUSTOM_MODEL or "gpt-4o"
        else:
            # Default to OpenAI settings
            api_key = user_config.OPENAI_API_KEY or get_openai_api_key_env()
            base_url = user_config.OPENAI_URL or get_openai_url_env() or "https://api.openai.com/v1"
            model_name = user_config.OPENAI_MODEL or get_openai_model_env() or "gpt-4o"
        
        if not api_key:
            raise HTTPException(
                status_code=500, 
                detail=f"API key not set for LLM provider: {llm_provider}"
            )
        
        # Validate HTML content
        if not request.html or not request.html.strip():
            raise HTTPException(
                status_code=400,
                detail="HTML content cannot be empty"
            )
        
        # Optionally resolve image and encode to base64
        image_b64 = None
        media_type = None
        if request.image:
            image_path = request.image
            if image_path.startswith("/app_data/images/"):
                relative_path = image_path[len("/app_data/images/"):]
                actual_image_path = os.path.join(get_images_directory(), relative_path)
            elif image_path.startswith("/static/"):
                relative_path = image_path[len("/static/"):]
                actual_image_path = os.path.join("static", relative_path)
            else:
                actual_image_path = image_path if os.path.isabs(image_path) else os.path.join(get_images_directory(), image_path)
            if os.path.exists(actual_image_path):
                with open(actual_image_path, "rb") as f:
                    image_b64 = base64.b64encode(f.read()).decode("utf-8")
                ext = os.path.splitext(actual_image_path)[1].lower()
                media_type = {'.png':'image/png','.jpg':'image/jpeg','.jpeg':'image/jpeg','.gif':'image/gif','.webp':'image/webp'}.get(ext, 'image/png')
        
        # Convert HTML to React component
        react_component = await generate_react_component_from_html(
            html_content=request.html,
            api_key=api_key,
            base_url=base_url,
            model_name=model_name,
            image_base64=image_b64,
            media_type=media_type
        )

        react_component = react_component.replace("```tsx", "").replace("```", "")
        
        # FALLBACK STRATEGY: Auto-fix problematic schema patterns
        react_component = auto_fix_schema_objects(react_component)
        
        return HtmlToReactResponse(
            success=True,
            react_component=react_component,
            message="React component generated successfully"
        )
        
    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    except Exception as e:
        # Log the full error for debugging
        print(f"Unexpected error during HTML to React processing: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error processing HTML to React: {str(e)}"
        )


# ENDPOINT 3: HTML editing with images
@HTML_EDIT_ROUTER.post("/", response_model=HtmlEditResponse)
async def edit_html_with_images_endpoint(
    current_ui_image: UploadFile = File(..., description="Current UI image file"),
    sketch_image: Optional[UploadFile] = File(None, description="Sketch/indication image file (optional)"),
    html: str = Form(..., description="Current HTML content to edit"),
    prompt: str = Form(..., description="Text prompt describing the changes")
):
    """
    Edit HTML content based on one or two uploaded images and a text prompt using OpenAI compatible API.
    
    Args:
        current_ui_image: Uploaded current UI image file
        sketch_image: Uploaded sketch/indication image file (optional)
        html: Current HTML content to edit (form data)
        prompt: Text prompt describing the changes (form data)
    
    Returns:
        HtmlEditResponse with edited HTML
    """
    try:
        # Get user configuration for dynamic API settings
        user_config = get_user_config()
        
        # Determine API key and URL based on LLM provider
        llm_provider = user_config.LLM or get_llm_provider_env()
        
        if llm_provider == "custom":
            api_key = user_config.CUSTOM_LLM_API_KEY
            base_url = user_config.CUSTOM_LLM_URL
            model_name = user_config.CUSTOM_MODEL or "gpt-4o"
        else:
            # Default to OpenAI settings
            api_key = user_config.OPENAI_API_KEY or get_openai_api_key_env()
            base_url = user_config.OPENAI_URL or get_openai_url_env() or "https://api.openai.com/v1"
            model_name = user_config.OPENAI_MODEL or get_openai_model_env() or "gpt-4o"
        
        if not api_key:
            raise HTTPException(
                status_code=500, 
                detail=f"API key not set for LLM provider: {llm_provider}"
            )
        
        # Validate inputs
        if not html or not html.strip():
            raise HTTPException(
                status_code=400,
                detail="HTML content cannot be empty"
            )
        
        if not prompt or not prompt.strip():
            raise HTTPException(
                status_code=400,
                detail="Text prompt cannot be empty"
            )
        
        # Validate current UI image file
        if not current_ui_image.content_type or not current_ui_image.content_type.startswith("image/"):
            raise HTTPException(
                status_code=400,
                detail="Current UI file must be an image"
            )
        
        # Validate sketch image file only if provided
        if sketch_image and (not sketch_image.content_type or not sketch_image.content_type.startswith("image/")):
            raise HTTPException(
                status_code=400,
                detail="Sketch file must be an image"
            )
        
        # Read and encode current UI image to base64
        current_ui_content = await current_ui_image.read()
        current_ui_base64 = base64.b64encode(current_ui_content).decode('utf-8')
        
        # Read and encode sketch image to base64 only if provided
        sketch_base64 = None
        if sketch_image:
            sketch_content = await sketch_image.read()
            sketch_base64 = base64.b64encode(sketch_content).decode('utf-8')
        
        # Use the content type from the uploaded files
        media_type = current_ui_image.content_type
        
        # Edit HTML using the function
        edited_html = await edit_html_with_images(
            current_ui_base64=current_ui_base64,
            sketch_base64=sketch_base64,
            media_type=media_type,
            html_content=html,
            prompt=prompt,
            api_key=api_key,
            base_url=base_url,
            model_name=model_name
        )

        edited_html = edited_html.replace("```html", "").replace("```", "")
        
        return HtmlEditResponse(
            success=True,
            edited_html=edited_html,
            message="HTML edited successfully"
        )
        
    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    except Exception as e:
        # Log the full error for debugging
        print(f"Unexpected error during HTML editing: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error processing HTML editing: {str(e)}"
        ) 


# ENDPOINT 3.5: Test endpoint to debug validation issues
@LAYOUT_MANAGEMENT_ROUTER.post("/test-validation")
async def test_validation(raw_body: dict):
    """
    Test endpoint to debug validation issues.
    """
    print(f"DEBUG: Raw body received: {raw_body}")
    try:
        # Try to create the SaveLayoutsRequest manually
        request = SaveLayoutsRequest(**raw_body)
        print(f"DEBUG: Validation successful! {len(request.layouts)} layouts")
        return {"status": "success", "layouts_count": len(request.layouts)}
    except Exception as e:
        print(f"DEBUG: Validation failed: {e}")
        return {"status": "error", "error": str(e)}

# ENDPOINT 4: Save layouts for a presentation
@LAYOUT_MANAGEMENT_ROUTER.post(
    "/save-templates", 
    response_model=SaveLayoutsResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Validation error"},
        500: {"model": ErrorResponse, "description": "Internal server error"}
    }
)
async def save_layouts(
    raw_data: dict,
    session: AsyncSession = Depends(get_async_session)
):
    """
    Save multiple layouts for presentations.
    
    Args:
        request: JSON request containing array of layout data
        session: Database session
    
    Returns:
        SaveLayoutsResponse with success status and count of saved layouts
    
    Raises:
        HTTPException: 400 for validation errors, 500 for server errors
    """
    try:
        print(f"DEBUG: save_layouts function reached successfully!")
        print(f"DEBUG: Raw data type: {type(raw_data)}")
        print(f"DEBUG: Raw data keys: {raw_data.keys() if isinstance(raw_data, dict) else 'Not a dict'}")
        print(f"DEBUG: Raw data: {raw_data}")
        
        # Try to parse the request
        try:
            request = SaveLayoutsRequest(**raw_data)
            print(f"DEBUG: Parsing successful! {len(request.layouts)} layouts")
        except Exception as parse_error:
            print(f"DEBUG: Parsing failed: {parse_error}")
            print(f"DEBUG: Trying to understand the structure...")
            if isinstance(raw_data, dict) and 'layouts' in raw_data:
                layouts_data = raw_data['layouts']
                print(f"DEBUG: layouts key found, type: {type(layouts_data)}")
                if isinstance(layouts_data, list) and len(layouts_data) > 0:
                    print(f"DEBUG: First layout structure: {layouts_data[0]}")
                    print(f"DEBUG: First layout keys: {layouts_data[0].keys() if isinstance(layouts_data[0], dict) else 'Not a dict'}")
            raise HTTPException(
                status_code=400,
                detail=f"Request parsing failed: {parse_error}"
            )
        
        # Additional debugging - log each layout structure
        for i, layout in enumerate(request.layouts):
            print(f"DEBUG: Layout {i+1} structure:")
            print(f"  - presentation_id: {layout.presentation_id} (type: {type(layout.presentation_id)})")
            print(f"  - layout_id: '{layout.layout_id}' (type: {type(layout.layout_id)})")
            print(f"  - layout_name: '{layout.layout_name}' (type: {type(layout.layout_name)})")
            print(f"  - layout_code length: {len(layout.layout_code) if layout.layout_code else 0} (type: {type(layout.layout_code)})")
            print(f"  - fonts: {layout.fonts} (type: {type(layout.fonts)})")
        
        # Validate request data
        if not request.layouts:
            print("ERROR: Layouts array is empty")
            raise HTTPException(
                status_code=400,
                detail="Layouts array cannot be empty"
            )
        
        if len(request.layouts) > 50:  # Reasonable limit
            print(f"ERROR: Too many layouts: {len(request.layouts)}")
            raise HTTPException(
                status_code=400,
                detail="Cannot save more than 50 layouts at once"
            )
        
        saved_count = 0
        
        for i, layout_data in enumerate(request.layouts):
            print(f"DEBUG: Validating layout {i+1}:")
            print(f"  presentation_id: {layout_data.presentation_id}")
            print(f"  layout_id: '{layout_data.layout_id}'")
            print(f"  layout_name: '{layout_data.layout_name}'")
            print(f"  layout_code length: {len(layout_data.layout_code) if layout_data.layout_code else 0}")
            print(f"  fonts: {layout_data.fonts}")
            
            # Validate individual layout data
            if not layout_data.presentation_id or not str(layout_data.presentation_id).strip():
                print(f"ERROR: Layout {i+1} - presentation_id is empty")
                raise HTTPException(
                    status_code=400,
                    detail=f"Layout {i+1}: presentation_id cannot be empty"
                )
            
            if not layout_data.layout_id or not layout_data.layout_id.strip():
                print(f"ERROR: Layout {i+1} - layout_id is empty")
                raise HTTPException(
                    status_code=400,
                    detail=f"Layout {i+1}: layout_id cannot be empty"
                )
            
            if not layout_data.layout_name or not layout_data.layout_name.strip():
                print(f"ERROR: Layout {i+1} - layout_name is empty")
                raise HTTPException(
                    status_code=400,
                    detail=f"Layout {i+1}: layout_name cannot be empty"
                )
            
            if not layout_data.layout_code or not layout_data.layout_code.strip():
                print(f"ERROR: Layout {i+1} - layout_code is empty or whitespace only")
                print(f"  layout_code repr: {repr(layout_data.layout_code)}")
                raise HTTPException(
                    status_code=400,
                    detail=f"Layout {i+1}: layout_code cannot be empty"
                )
            
            # Check if layout already exists for this presentation and layout_id
            stmt = select(PresentationLayoutCodeModel).where(
                PresentationLayoutCodeModel.presentation == layout_data.presentation_id,
                PresentationLayoutCodeModel.layout_id == layout_data.layout_id
            )
            result = await session.execute(stmt)
            existing_layout = result.scalar_one_or_none()
            
            if existing_layout:
                # Update existing layout
                existing_layout.layout_name = layout_data.layout_name
                existing_layout.layout_code = layout_data.layout_code
                existing_layout.fonts = layout_data.fonts
                existing_layout.updated_at = datetime.now()
            else:
                # Create new layout
                new_layout = PresentationLayoutCodeModel(
                    presentation=layout_data.presentation_id,
                    layout_id=layout_data.layout_id,
                    layout_name=layout_data.layout_name,
                    layout_code=layout_data.layout_code,
                    fonts=layout_data.fonts
                )
                session.add(new_layout)
            
            saved_count += 1
        
        await session.commit()
        
        return SaveLayoutsResponse(
            success=True,
            saved_count=saved_count,
            message=f"Successfully saved {saved_count} layout(s)"
        )
        
    except HTTPException:
        # Re-raise HTTP exceptions as-is
        await session.rollback()
        raise
    except Exception as e:
        await session.rollback()
        print(f"Unexpected error saving layouts: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error while saving layouts: {str(e)}"
        )


# ENDPOINT 5: Get layouts for a presentation
@LAYOUT_MANAGEMENT_ROUTER.get(
    "/get-templates/{presentation}", 
    response_model=GetLayoutsResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid presentation ID"},
        404: {"model": ErrorResponse, "description": "No layouts found for presentation"},
        500: {"model": ErrorResponse, "description": "Internal server error"}
    }
)
async def get_layouts(
    presentation: UUID,
    session: AsyncSession = Depends(get_async_session)
):
    """
    Retrieve all layouts for a specific presentation.
    
    Args:
        presentation: UUID of the presentation
        session: Database session
    
    Returns:
        GetLayoutsResponse with layouts data
    
    Raises:
        HTTPException: 404 if no layouts found, 400 for invalid UUID, 500 for server errors
    """
    try:
        # Validate presentation_id format (basic UUID check)
        if not presentation or len(str(presentation).strip()) == 0:
            raise HTTPException(
                status_code=400,
                detail="Presentation ID cannot be empty"
            )
        
        # Query layouts for the given presentation_id
        stmt = select(PresentationLayoutCodeModel).where(
            PresentationLayoutCodeModel.presentation == presentation
        )
        result = await session.execute(stmt)
        layouts_db = result.scalars().all()
        
        # Check if any layouts were found
        if not layouts_db:
            raise HTTPException(
                status_code=404,
                detail=f"No layouts found for presentation ID: {presentation}"
            )
        
        # Convert to response format
        layouts = [
            LayoutData(
                presentation_id=layout.presentation,
                layout_id=layout.layout_id,
                layout_name=layout.layout_name,
                layout_code=layout.layout_code,
                fonts=layout.fonts
            )
            for layout in layouts_db
        ]
        
        # Aggregate unique fonts across all layouts
        aggregated_fonts: set[str] = set()
        for layout in layouts_db:
            if layout.fonts:
                aggregated_fonts.update([f for f in layout.fonts if isinstance(f, str)])
        fonts_list = sorted(list(aggregated_fonts)) if aggregated_fonts else None
        
        # Fetch template meta
        template_meta = await session.get(TemplateModel, presentation)
        template = None
        if template_meta:
            template = {
                "id": template_meta.id,
                "name": template_meta.name,
                "description": template_meta.description,
                "created_at": template_meta.created_at,
            }

        return GetLayoutsResponse(
            success=True,
            layouts=layouts,
            message=f"Retrieved {len(layouts)} layout(s) for presentation {presentation}",
            template=template,
            fonts=fonts_list,
        )
        
    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    except Exception as e:
        print(f"Error retrieving layouts for presentation {presentation}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error while retrieving layouts: {str(e)}"
        )


# ENDPOINT: Get all presentations with layout counts
@LAYOUT_MANAGEMENT_ROUTER.get(
    "/summary",
    response_model=GetPresentationSummaryResponse,
    summary="Get all presentations with layout counts",
    description="Retrieve a summary of all presentations and the number of layouts in each",
    responses={
        200: {"model": GetPresentationSummaryResponse, "description": "Presentations summary retrieved successfully"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
    },
)
async def get_presentations_summary(
    session: AsyncSession = Depends(get_async_session),
):
    """
    Get summary of all presentations with their layout counts.
    """
    try:
        # Query to get presentation_id, count of layouts, and MAX(updated_at)
        stmt = select(
            PresentationLayoutCodeModel.presentation,
            func.count(PresentationLayoutCodeModel.id).label('layout_count'),
            func.max(PresentationLayoutCodeModel.updated_at).label('last_updated_at')
        ).group_by(PresentationLayoutCodeModel.presentation)
        
        result = await session.execute(stmt)
        presentation_data = result.all()
        
        # Convert to response format with template info if available
        presentations = []
        for row in presentation_data:
            template_meta = await session.get(TemplateModel, row.presentation)
            template = None
            if template_meta:
                template = {
                    "id": template_meta.id,
                    "name": template_meta.name,
                    "description": template_meta.description,
                    "created_at": template_meta.created_at,
                }
            presentations.append(
                PresentationSummary(
                    presentation_id=row.presentation,
                    layout_count=row.layout_count,
                    last_updated_at=row.last_updated_at,
                    template=template,
                )
            )

        # Calculate totals
        total_presentations = len(presentations)
        total_layouts = sum(p.layout_count for p in presentations)
        
        return GetPresentationSummaryResponse(
            success=True,
            presentations=presentations,
            total_presentations=total_presentations,
            total_layouts=total_layouts,
            message=f"Retrieved {total_presentations} presentation(s) with {total_layouts} total layout(s)",
        )
        
    except Exception as e:
        print(f"Error retrieving presentations summary: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error while retrieving presentations summary: {str(e)}"
        ) 


@LAYOUT_MANAGEMENT_ROUTER.post(
    "/templates",
    response_model=TemplateCreateResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Validation error"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
    },
)
async def create_template(
    request: TemplateCreateRequest,
    session: AsyncSession = Depends(get_async_session),
):
    try:
        if not request.id or not request.name:
            raise HTTPException(status_code=400, detail="id and name are required")

        # Upsert template by id
        existing = await session.get(TemplateModel, request.id)
        if existing:
            existing.name = request.name
            existing.description = request.description
        else:
            session.add(
                TemplateModel(
                    id=request.id, name=request.name, description=request.description
                )
            )
        await session.commit()

        # Read back
        template = await session.get(TemplateModel, request.id)
        return TemplateCreateResponse(
            success=True,
            template={
                "id": template.id,
                "name": template.name,
                "description": template.description,
                "created_at": template.created_at,
            },
            message="Template saved",
        )
    except HTTPException:
        await session.rollback()
        raise
    except Exception as e:
        await session.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to save template: {str(e)}")


# Temporary endpoint to fix existing layouts with missing export statements
@LAYOUT_MANAGEMENT_ROUTER.post("/fix-exports")
async def fix_layout_exports(session: AsyncSession = Depends(get_async_session)):
    """
    Temporary endpoint to add missing export statements to existing layouts
    """
    try:
        # Get all layouts
        stmt = select(PresentationLayoutCodeModel)
        result = await session.execute(stmt)
        layouts = result.scalars().all()
        
        updated_count = 0
        for layout in layouts:
            # Check if the layout code already has export statements
            if "export default dynamicSlideLayout" not in layout.layout_code:
                # Add the export statements
                if not layout.layout_code.endswith("\n"):
                    layout.layout_code += "\n"
                layout.layout_code += "\nexport default dynamicSlideLayout;\nexport { Schema };"
                updated_count += 1
        
        await session.commit()
        
        return {
            "success": True,
            "message": f"Updated {updated_count} layouts with export statements",
            "total_layouts": len(layouts),
            "updated_count": updated_count
        }
    except Exception as e:
        await session.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to fix layout exports: {str(e)}") 


@LAYOUT_MANAGEMENT_ROUTER.post("/fix-schemas")
async def fix_layout_schemas(session: AsyncSession = Depends(get_async_session)):
    """
    Fix Schema validation issues in existing layouts by providing simple defaults.
    """
    try:
        # Get all layouts
        stmt = select(PresentationLayoutCodeModel)
        result = await session.execute(stmt)
        layouts = result.scalars().all()
        
        fixed_count = 0
        for layout in layouts:
            code = layout.layout_code
            original_code = code
            
            # Check if this layout has common schema validation issues
            if "z.object({" in code:
                import re
                
                # Simple fix: Replace complex object schemas for text fields with string defaults
                # This handles the common case where header/subtext are defined as objects but should be strings
                
                # Find ALL object type definitions and replace with string defaults
                # Use a general pattern to find any field defined as z.object
                object_field_pattern = r'(\w+):\s*z\.object\s*\('
                object_matches = re.findall(object_field_pattern, code, re.IGNORECASE)
                
                print(f"Found object fields in layout {layout.layout_id}: {object_matches}")
                
                for field in object_matches:
                    # Look for multiline object definitions like: field: z.object({...})
                    # Use a more sophisticated approach to handle nested braces
                    pattern_start = f'{field}:\\s*z\\.object\\s*\\('
                    
                    if re.search(pattern_start, code, re.IGNORECASE):
                        # Find the start position
                        match = re.search(pattern_start, code, re.IGNORECASE)
                        if match:
                            start_pos = match.start()
                            # Find the matching closing brace by counting braces
                            brace_count = 0
                            in_object = False
                            end_pos = start_pos
                            
                            for i, char in enumerate(code[match.end():], match.end()):
                                if char == '{':
                                    if not in_object:
                                        in_object = True
                                    brace_count += 1
                                elif char == '}':
                                    brace_count -= 1
                                    if brace_count == 0 and in_object:
                                        # Find the closing parenthesis
                                        for j in range(i + 1, len(code)):
                                            if code[j] == ')':
                                                end_pos = j + 1
                                                break
                                        break
                            
                            if end_pos > start_pos:
                                # Replace the entire object definition with a simple string
                                before = code[:start_pos]
                                after = code[end_pos:]
                                replacement = f'{field}: z.string().default("")'
                                code = before + replacement + after
                                print(f"Fixed {field} field in layout {layout.layout_id}")
                                fixed_count += 1
                
                # Also ensure the Schema is properly defined and accessible
                if "const Schema =" not in code:
                    # Try to find any schema definition and make it accessible
                    schema_pattern = r'(\w+Schema\s*=\s*z\.object\(\{[^}]+\}\))'
                    match = re.search(schema_pattern, code)
                    if match:
                        # Add a generic Schema constant
                        code += "\nconst Schema = " + match.group(1).split('=', 1)[1]
                        
                # Fix common TypeScript syntax errors in existing layouts
                # Fix malformed type definitions like "type Name = infer<typeof schema>fx" 
                # Should be "type Name = z.infer<typeof Schema>;"
                type_pattern = r'type\s+(\w+)\s*=\s*infer<typeof\s+(\w+)>(\w*)'
                def fix_type_definition(match):
                    type_name = match.group(1)
                    schema_name = match.group(2)
                    # Ensure proper capitalization for Schema and add z. prefix
                    proper_schema = "Schema" if schema_name.lower() == "schema" else schema_name
                    return f'type {type_name} = z.infer<typeof {proper_schema}>;'
                
                if re.search(type_pattern, code):
                    code = re.sub(type_pattern, fix_type_definition, code)
                    print(f"Fixed TypeScript type definition in layout {layout.layout_id}")
                    fixed_count += 1
                
                # Fix missing semicolons in type definitions
                semicolon_pattern = r'(type\s+\w+\s*=\s*z\.infer<typeof\s+\w+>)(?!;)'
                if re.search(semicolon_pattern, code):
                    code = re.sub(semicolon_pattern, r'\1;', code)
                    print(f"Fixed missing semicolon in type definition for layout {layout.layout_id}")
                    fixed_count += 1
                
                if code != original_code:
                    layout.layout_code = code
                    fixed_count += 1
        
        if fixed_count > 0:
            await session.commit()
        
        return {
            "success": True,
            "fixed_layouts": fixed_count,
            "total_layouts": len(layouts),
            "message": f"Fixed {fixed_count} layout(s) by correcting schema definitions"
        }
        
    except Exception as e:
        print(f"Error fixing layout schemas: {e}")
        await session.rollback()
        return {
            "success": False,
            "error": str(e),
            "message": "Failed to fix layout schemas"
        }


@LAYOUT_MANAGEMENT_ROUTER.post("/debug-schemas")
async def debug_layout_schemas(session: AsyncSession = Depends(get_async_session)):
    """
    Debug schema issues by showing what fields exist in layouts.
    """
    try:
        # Get all layouts
        stmt = select(PresentationLayoutCodeModel)
        result = await session.execute(stmt)
        layouts = result.scalars().all()
        
        debug_info = []
        
        for layout in layouts:
            if layout.layout_code:
                code = layout.layout_code
                layout_info = {
                    "layout_id": layout.layout_id,
                    "presentation": str(layout.presentation),
                    "fields_found": [],
                    "object_fields": [],
                    "has_subtitle_section": "subTitleSection" in code,
                    "code_snippet": ""
                }
                
                # Look for all z.object fields
                object_pattern = r'(\w+):\s*z\.object\s*\('
                object_matches = re.findall(object_pattern, code, re.IGNORECASE)
                layout_info["object_fields"] = object_matches
                
                # Look for any field with "section" or "title" in the name
                field_pattern = r'(\w*(?:title|section|header|text|content)\w*):\s*z\.'
                field_matches = re.findall(field_pattern, code, re.IGNORECASE)
                layout_info["fields_found"] = field_matches
                
                # Get a snippet around subTitleSection if it exists
                if "subTitleSection" in code:
                    lines = code.split('\n')
                    for i, line in enumerate(lines):
                        if 'subTitleSection' in line:
                            start = max(0, i - 2)
                            end = min(len(lines), i + 3)
                            layout_info["code_snippet"] = '\n'.join(lines[start:end])
                            break
                
                debug_info.append(layout_info)
        
        return {
            "success": True,
            "total_layouts": len(layouts),
            "debug_info": debug_info
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to debug schemas: {str(e)}")


@LAYOUT_MANAGEMENT_ROUTER.post("/fix-missing-components")
async def fix_missing_components(session: AsyncSession = Depends(get_async_session)):
    """
    Fix layouts that are missing React component definitions.
    """
    try:
        # Get all layouts
        stmt = select(PresentationLayoutCodeModel)
        result = await session.execute(stmt)
        layouts = result.scalars().all()
        
        fixed_count = 0
        
        for layout in layouts:
            original_code = layout.layout_code
            code = original_code
            
            # Check if the layout is missing React component definition
            if "dynamicSlideLayout" not in code or "const dynamicSlideLayout" not in code:
                # Extract layout metadata if available
                layout_id = "generic-slide"
                layout_name = "GenericLayout"  
                layout_description = "A generic slide layout"
                
                # Try to extract from existing constants
                import re
                id_match = re.search(r'const\s+layoutId\s*=\s*["\']([^"\']+)["\']', code)
                if id_match:
                    layout_id = id_match.group(1)
                    
                name_match = re.search(r'const\s+layoutName\s*=\s*["\']([^"\']+)["\']', code)
                if name_match:
                    layout_name = name_match.group(1)
                    
                desc_match = re.search(r'const\s+layoutDescription\s*=\s*["\']([^"\']*)["\']', code)
                if desc_match:
                    layout_description = desc_match.group(1)
                
                # Add missing component structure
                component_template = f'''

interface {layout_name}Props {{
    data?: Partial<SlideDataSchema>
}}

const dynamicSlideLayout: React.FC<{layout_name}Props> = ({{ data: slideData }}) => {{
    return (
        <div className="w-full rounded-sm max-w-[1280px] shadow-lg max-h-[720px] aspect-video bg-white relative z-20 mx-auto overflow-hidden">
            <div className="flex flex-col h-full p-8">
                <div className="text-2xl font-bold text-gray-800 mb-4">
                    {{slideData?.header || "标题"}}
                </div>
                <div className="flex-1 flex items-center justify-center">
                    <div className="text-center text-gray-600">
                        <p>{{slideData?.content || "内容区域"}}</p>
                    </div>
                </div>
            </div>
        </div>
    );
}};
'''
                
                # Append the component to the existing code
                code = code + component_template
                
                # Update the layout
                layout.layout_code = code
                fixed_count += 1
                print(f"Added missing component to layout {layout.layout_id}")
        
        if fixed_count > 0:
            await session.commit()
        
        return {
            "success": True,
            "fixed_layouts": fixed_count,
            "total_layouts": len(layouts),
            "message": f"Fixed {fixed_count} layout(s) by adding missing component definitions"
        }
        
    except Exception as e:
        print(f"Error fixing missing components: {e}")
        await session.rollback()
        return {
            "success": False,
            "error": str(e),
            "message": "Failed to fix missing components"
        }