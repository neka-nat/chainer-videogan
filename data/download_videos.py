from pytube import YouTube

# Gravure idol
video_list = ["https://www.youtube.com/watch?v=jCYz5tBR1q0",
              "https://www.youtube.com/watch?v=ix75NgPPptE",
              "https://www.youtube.com/watch?v=1-oHRcTJPns",
              "https://www.youtube.com/watch?v=AvlMPpcdoJE",
              "https://www.youtube.com/watch?v=JymEPvWUi3U"]

files = []
for idx, vd in enumerate(video_list):
    try:
        print("Downloading " + vd)
        yt = YouTube(vd)
        filename = './video_%d' % idx
        yt.streams.filter(mime_type='video/mp4').first().download(filename=filename)
        files.append(filename + ".mp4")
    except:
        print("Fail")
        pass

train_list_file = "train_list.txt"
print('\nSave %s' % train_list_file)
with open(train_list_file, 'w') as f:
    f.write('\n'.join(files))
