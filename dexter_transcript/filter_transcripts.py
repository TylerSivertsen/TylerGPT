import os

filelist = os.listdir("dexter_transcript/season_1")

transcript = []
for file in filelist:
    with open(os.path.join("dexter_transcript/season_1", file), "r") as f:
        content = f.read()

    transcript.append(content)

with open("dexter_transcript/season_1/season_1_transcript_filtered.txt", "w") as f:
    for episode in transcript:
        f.write(episode)