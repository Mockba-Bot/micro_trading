import os
import aiohttp
from dotenv import load_dotenv

# Load environment variables from the .env file
load_dotenv(dotenv_path=".env.micro.trading")


MICRO_CENTRAL_URL = os.getenv("MICRO_CENTRAL_URL")  # Your micro central URL

async def send_bot_message(token, message, parse_mode="HTML"):
    url = f"{MICRO_CENTRAL_URL}/send_notification"
    payload = {
        "token": token,
        "message": message
    }
    headers = {
        "Token": token
    }
    async with aiohttp.ClientSession() as session:
        async with session.post(url, json=payload, headers=headers) as response:
            if response.status == 200:
                return await response.json()
            else:
                response.raise_for_status()  