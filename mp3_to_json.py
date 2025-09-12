import whisper
import json
import os

# Load the Whisper model
model = whisper.load_model("large-v2")

audios = os.listdir("audios")

for audio in audios: 
    if "_" in audio:
        number = audio.split("_")[0]
        # title = audio.split("_")[1][:-4]

        result = model.transcribe(
            audio=f"audios/{audio}",   # use the loop variable instead of hardcoding
            language="hi",
            task="translate",
            word_timestamps=False
        )
        
        chunks = []
        for segment in result["segments"]:
            chunks.append({
                "number": number,
                "start": segment["start"],
                "end": segment["end"],
                "text": segment["text"]
            })
        
        chunks_with_metadata = {
            "chunks": chunks,
            "text": result["text"]
        }
        
        # Ensure output directory exists
        os.makedirs("jsons", exist_ok=True)

        with open(f"jsons/{audio}.json", "w", encoding="utf-8") as f:
            json.dump(chunks_with_metadata, f, ensure_ascii=False, indent=2)
