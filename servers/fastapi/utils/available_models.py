import asyncio
import httpx
from anthropic import AsyncAnthropic
from openai import AsyncOpenAI
from google import genai


async def list_available_openai_compatible_models(url: str, api_key: str) -> list[str]:
    # Configure retry and timeout settings
    max_retries = 3
    timeout = 60.0  # 60 seconds timeout

    for attempt in range(max_retries):
        try:
            # Create HTTP client with retry and timeout configuration
            http_client = httpx.AsyncClient(
                timeout=httpx.Timeout(timeout, connect=10.0),
                limits=httpx.Limits(max_connections=10, max_keepalive_connections=5),
                transport=httpx.AsyncHTTPTransport(retries=2)
            )

            client = AsyncOpenAI(
                api_key=api_key,
                base_url=url,
                http_client=http_client,
                timeout=timeout
            )

            models = (await client.models.list()).data
            if models:
                return list(map(lambda x: x.id, models))
            return []

        except Exception as e:
            print(f"Attempt {attempt + 1}/{max_retries} failed: {str(e)}")
            if attempt == max_retries - 1:
                raise e
            # Exponential backoff: wait 1s, 2s, 4s
            await asyncio.sleep(2 ** attempt)

    return []


async def list_available_anthropic_models(api_key: str) -> list[str]:
    client = AsyncAnthropic(api_key=api_key)
    return list(map(lambda x: x.id, (await client.models.list(limit=50)).data))


async def list_available_google_models(api_key: str) -> list[str]:
    client = genai.Client(api_key=api_key)
    return list(map(lambda x: x.name, client.models.list(config={"page_size": 50})))
