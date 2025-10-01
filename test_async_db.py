import asyncio
from sqlalchemy.ext.asyncio import create_async_engine

async def main():
    engine = create_async_engine("sqlite+aiosqlite:///D:/presenton/app_data/fastapi.db")
    async with engine.begin() as conn:
        await conn.run_sync(lambda c: None)
    await engine.dispose()

asyncio.run(main())
print("ok")
