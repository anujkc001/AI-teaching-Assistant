# convert the video to mp3
import os
import subprocess

files=os.listdir("videos")
#print(files)
for file in files:
   # print(file)
   # tutorial_number=file.split(" [")[0]#.split(" #")#[1]   
    file_name=file.split(" | ")[0]
    print(file_name)#tutorial_number)
    subprocess.run(["ffmpeg","-i",f"videos/{file}",f"audios/{file_name}.mp3"])
