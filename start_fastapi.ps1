Set-Location "D:\presenton\servers\fastapi"
$env:APP_DATA_DIRECTORY = 'D:\presenton\app_data'
$env:USER_CONFIG_PATH = 'D:\presenton\app_data\userConfig.json'
$env:DATABASE_URL = 'sqlite+aiosqlite:///D:/presenton/app_data/fastapi.db'
uv run python server.py --port 8000 --reload true
