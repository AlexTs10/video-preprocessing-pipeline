import yt_dlp

NUMBER_OF_VIDEOS_TO_DOWNLOAD = 120
 
def download_video(video_url, download_path):
    ydl_opts = {
        'format': 'best',
        'outtmpl': download_path + '/%(id)s.%(ext)s',  # Save video with its YouTube ID

    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([video_url])

def get_channel_videos(channel_url):
    ydl_opts = {
        'quiet': True,
        'extract_flat': True,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        result = ydl.extract_info(channel_url, download=False)
        if 'entries' in result:
            return result['entries']
        else:
            return []

channel_url = 'https://www.youtube.com/c/@thesun'
download_dir = 'videos_from_the_sun_yt'

videos = get_channel_videos(channel_url)

# Filter videos containing 'fpv' or 'drone' in the title
filtered_videos = [video for video in videos[0]['entries'] if 'fpv' in video['title'].lower() or 'drone' in video['title'].lower()]

# Sort videos by duration
sorted_videos = sorted(filtered_videos, key=lambda v: v['duration'])


# Download the videos, limiting to 120
for video in sorted_videos[:NUMBER_OF_VIDEOS_TO_DOWNLOAD]:
    download_video(video['url'], download_dir)
    print(f"Downloaded: {video['title']}")
