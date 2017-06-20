from pytube import YouTube

# Bruce Lee
#video_list = ["https://www.youtube.com/watch?v=fifUccuJbEE",
#              "https://www.youtube.com/watch?v=J3zYme6oi_I",
#              "https://www.youtube.com/watch?v=nFpWsE6nKnI",
#              "https://www.youtube.com/watch?v=1eDN4nOhCY0"]

# Idol dance
#video_list = ["https://www.youtube.com/watch?v=FA9R7UfWtkY",
#              "https://www.youtube.com/watch?v=M1-e-_QyMp4",
#              "https://www.youtube.com/watch?v=y1uTFsWXuYI",
#              "https://www.youtube.com/watch?v=dNh8ZRCiHTg",
#              "https://www.youtube.com/watch?v=-Q8WW6vVdyA",
#              "https://www.youtube.com/watch?v=LTXJk9CSjJY"]

# Gravure idol
video_list = ["https://www.youtube.com/watch?v=m8fMUiQ2m08",
              "https://www.youtube.com/watch?v=YX-rm9cvdzc",
              "https://www.youtube.com/watch?v=49jAY9dGhiw",
              "https://www.youtube.com/watch?v=ayGdHp7v0oM",
              "https://www.youtube.com/watch?v=AKRkZB6jKqQ",
              "https://www.youtube.com/watch?v=Ku8NFXEqG74",
              "https://www.youtube.com/watch?v=COQRU0bb6-c",
              "https://www.youtube.com/watch?v=1Wjjk5tLzWU",
              "https://www.youtube.com/watch?v=1ZawgNhvZzY"]

files = []
for idx, vd in enumerate(video_list):
    try:
        print("Downloading " + vd)
        yt = YouTube(vd)
        video = yt.get('mp4', '360p')
        filename = 'video_%d' % idx
        yt.set_filename(filename)
        video.download('./')
        files.append(filename + ".mp4")
    except:
        print("Fail")
        pass

train_list_file = "train_list.txt"
print('\nSave %s' % train_list_file)
with open(train_list_file, 'w') as f:
    f.write('\n'.join(files))
