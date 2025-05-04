import whisper

model = whisper.load_model("small")
result = model.transcribe("CPSC_221_L6.mp3")
print(result["text"])