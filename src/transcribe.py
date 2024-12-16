import time
import whisper

start = time.time()
model = whisper.load_model(
    "large", device="cuda"
)  # CUDA is default if available anyway
result = model.transcribe("audio.mp3")

print(f"Transcription took {time.time() - start:.2f} seconds")
print(result)
