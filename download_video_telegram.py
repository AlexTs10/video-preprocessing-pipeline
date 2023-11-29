import asyncio
from pyrogram import Client
import os

# Replace with your own credentials
api_id = '25016928'
api_hash = '369a11d1cb91fba26edd9d1f0690f1fc'

# Replace 'channel_username' with the username of the Telegram channel
channel_username = 'combat_ftg'

async def download_videos():
    app = Client("my_account", api_id=api_id, api_hash=api_hash)
    
    async with app:
        video_count = 0
        async for message in app.get_chat_history(channel_username):
            if message.video and 5 <= message.video.duration <= 20:
                print(f"Downloading video from Message ID: {message.id} (Duration: {message.video.duration} seconds)")
                await app.download_media(message.video.file_id, file_name=f"downloads/{message.video.file_id}.mp4")
                video_count += 1

                if video_count >= 150:
                    break

if __name__ == "__main__":
    if not os.path.exists('downloads'):
        os.makedirs('downloads')

    loop = asyncio.get_event_loop()
    loop.run_until_complete(download_videos())
