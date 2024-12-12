import librosa
from transformers import AutoFeatureExtractor

SAMPLING_RATE = 16000

# Load feature extractor
feature_extractor = AutoFeatureExtractor.from_pretrained("team-lucid/hubert-base-korean")

def analyze_emotion(audio_path):
    audio_np, _ = librosa.load(audio_path, sr=SAMPLING_RATE, mono=True)
    inputs = feature_extractor(raw_speech=audio_np, return_tensors="pt", sampling_rate=SAMPLING_RATE)
    audio_values = inputs["input_values"].to("cuda" if torch.cuda.is_available() else "cpu").half()
    audio_attn_mask = inputs.get("attention_mask", None)

    with torch.no_grad():
        if audio_attn_mask is None:
            label2_logits, intensity_preds = hubert_model(audio_values)
        else:
            label2_logits, intensity_preds = hubert_model(audio_values, audio_attn_mask)

    label2 = torch.argmax(label2_logits, dim=-1).item()
    intensity = intensity_preds.item()
    return label2, intensity
