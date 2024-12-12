from fastapi import Request
from app.langchain_prompts import classify_chain, answer_chain
from app.transcriber import transcribe
from app.emotion_analysis import analyze_emotion
from app.google_search import get_google_results

async def analyze_request(request: Request, classify_chain, answer_chain, transcribe, analyze_emotion, get_google_results):
    data = await request.json()
    audio_path = data["audio_path"]
    user_text = transcribe(audio_path)
    emotion_label, emotion_intensity = analyze_emotion(audio_path)

    classify_result = classify_chain.predict(question=user_text)

    if classify_result.lower() == "false":
        additional_info = ""
    else:
        search_keywords = classify_result.split(",")
        if len(search_keywords) <= 2:
            locations = ",".join(search_keywords)
            additional_info = locations
        else:
            search_query = " ".join(search_keywords)
            additional_info = get_google_results(search_query)

    messages = answer_chain.prompt.format_messages(
        question=user_text,
        emotion_label=emotion_label,
        emotion_intensity=emotion_intensity,
        additional_info=additional_info,
        user_text=user_text
    )
    response = answer_chain(messages)
    response_content = response.content

    return {
        "user_text": user_text,
        "emotion_label": emotion_label,
        "emotion_intensity": emotion_intensity,
        "additional_info": additional_info,
        "response": response_content,
    }
