import whisper

whisper_model = whisper.load_model("base")

def transcribe(audio_path):
    audio = whisper.load_audio(audio_path)
    audio = whisper.pad_or_trim(audio)
    mel = whisper.log_mel_spectrogram(audio).to(whisper_model.device)
    options = whisper.DecodingOptions()
    result = whisper.decode(whisper_model, mel, options)
    return result.text
