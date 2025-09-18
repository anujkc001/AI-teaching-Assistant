import whisper
import json
import os

# load model once
model = whisper.load_model("large-v2")

# input and output folders
input_folder = "audios"
output_folder = "jsons"

# make sure output folder exists
os.makedirs(output_folder, exist_ok=True)

# loop through all audio files in audios/
for filename in os.listdir(input_folder):
    if filename.lower().endswith((".mp3", ".wav", ".m4a", ".flac")):
        audio_path = os.path.join(input_folder, filename)

        print(f"Processing: {audio_path}")

        # transcribe
        result = model.transcribe(
            audio=audio_path,
            language="hi",
            task="translate",
            word_timestamps=False
        )

        # collect chunks
        chunks = []
        for segment in result["segments"]:
            chunks.append({
                "start": segment["start"],
                "end": segment["end"],
                "text": segment["text"]
            })

        # save JSON file with same name as audio file
        json_filename = os.path.splitext(filename)[0] + ".json"
        json_path = os.path.join(output_folder, json_filename)

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(chunks, f, ensure_ascii=False, indent=2)

        print(f"Saved: {json_path}")
