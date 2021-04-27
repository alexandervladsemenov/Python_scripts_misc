from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
import os, sys
start_time = 0
end_time = 100
file_name = "-SubsPlease- Horimiya - 07 -1080p- -E8DB4B38-.mp4"
path = r"C:\Users\Alexander Semenov\Downloads"
inp_path = os.path.join(path,file_name)
out_pat = os.path.join(path,"test"+file_name)
ffmpeg_extract_subclip(inp_path, start_time, end_time, targetname=out_pat)
sys.exit()