import os
import torch
import librosa
from transformers import AutoFeatureExtractor, AutoConfig
import whisper
from pytorch_lightning import Trainer
import pytorch_lightning as pl
from torch import nn
from transformers import HubertForSequenceClassification
import dotenv
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from langchain.prompts.chat import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferWindowMemory
import urllib.parse
import requests
import logging
from typing import Optional
from langchain.chains import LLMChain
import logging
logger = logging.getLogger(__name__)


# Load environment variables
dotenv_file = dotenv.find_dotenv()
dotenv.load_dotenv(dotenv_file)
openai_api_key = os.getenv("OPENAI_API_KEY")
google_api_key = os.getenv("GOOGLE_API_KEY")
search_engine_id = os.getenv("SEARCH_ENGINE_ID")


# FastAPI Initialization
app = FastAPI(
    title="Cha-Vis",
    version="1.0",
    description="AI Navigation Cha-Vis using FastAPI, Whisper, HuBERT, and LangChain."
)

# CORS setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Model Configurations
audio_model_name = "team-lucid/hubert-base-korean"
NUM_LABELS = 7
SAMPLING_RATE = 16000

# Load Whisper Model
whisper_model = whisper.load_model("base")

# Load HuBERT Model
class MyLitModel(pl.LightningModule):
    def __init__(self, audio_model_name, num_label2s):
        super(MyLitModel, self).__init__()
        self.config = AutoConfig.from_pretrained(audio_model_name)
        self.config.output_hidden_states = True
        self.audio_model = HubertForSequenceClassification.from_pretrained(audio_model_name, config=self.config)
        self.label2_classifier = nn.Linear(self.audio_model.config.hidden_size, num_label2s)
        self.intensity_regressor = nn.Linear(self.audio_model.config.hidden_size, 1)

    def forward(self, audio_values, audio_attn_mask=None):
        outputs = self.audio_model(input_values=audio_values, attention_mask=audio_attn_mask)
        label2_logits = self.label2_classifier(outputs.hidden_states[-1][:, 0, :])
        intensity_preds = self.intensity_regressor(outputs.hidden_states[-1][:, 0, :]).squeeze(-1)
        return label2_logits, intensity_preds

pretrained_model_path = "/home/user/AZ/hubert/model2/fold_idx=11_epoch=28-val_acc_label2=0.4819.ckpt"
hubert_model = MyLitModel.load_from_checkpoint(
    pretrained_model_path,
    audio_model_name=audio_model_name,
    num_label2s=NUM_LABELS,
)
hubert_model.eval()
hubert_model.to("cuda" if torch.cuda.is_available() else "cpu")

# Load Feature Extractor
feature_extractor = AutoFeatureExtractor.from_pretrained(audio_model_name)

# Whisper Transcription
def transcribe(audio_path):
    audio = whisper.load_audio(audio_path)
    audio = whisper.pad_or_trim(audio)
    mel = whisper.log_mel_spectrogram(audio).to(whisper_model.device)
    options = whisper.DecodingOptions()
    result = whisper.decode(whisper_model, mel, options)
    return result.text

# HuBERT Emotion Analysis
def analyze_emotion(audio_path):
    audio_np, _ = librosa.load(audio_path, sr=SAMPLING_RATE, mono=True)
    inputs = feature_extractor(raw_speech=audio_np, return_tensors="pt", sampling_rate=SAMPLING_RATE)
    audio_values = inputs["input_values"].to(hubert_model.device)
    audio_attn_mask = inputs.get("attention_mask", None)
    if audio_attn_mask is not None:
        audio_attn_mask = audio_attn_mask.to(hubert_model.device)

    with torch.no_grad():
        if audio_attn_mask is None:
            label2_logits, intensity_preds = hubert_model(audio_values)
        else:
            label2_logits, intensity_preds = hubert_model(audio_values, audio_attn_mask)

    label2 = torch.argmax(label2_logits, dim=-1).item()
    intensity = intensity_preds.item()
    return label2, intensity

# Google Search Integration
def get_google_results(query, num_results=2):
    google_api_key = os.getenv("GOOGLE_API_KEY")
    search_engine_id = os.getenv("SEARCH_ENGINE_ID")
    
    encoded_query = urllib.parse.quote(query)
    url = f"https://www.googleapis.com/customsearch/v1?key={google_api_key}&cx={search_engine_id}&q={encoded_query}&num={num_results}"
    logger.info(f"Google Search URL: {url}")
    
    try:
        response = requests.get(url)
        response.raise_for_status()
        results = response.json()
        
        search_results = []
        if "items" in results:
            for item in results["items"]:
                search_results.append(f"Title: {item['title']}\nLink: {item['link']}\nSnippet: {item['snippet']}\n")
            
            return "\n".join(search_results)
        else:
            logger.info("No items found in the search results")
            return "검색 결과가 없습니다."
        
    except RequestException as e:
        logger.error(f"Network error occurred during Google search")
        return "네트워크 오류가 발생했습니다. 인터넷 연결을 확인해 주세요."
    
    except ValueError as e:
        logger.error(f"Value error occurred during Google search")
        return "검색어 형식이 올바르지 않습니다. 다시 입력해 주세요."
    
    except Exception as e:
        logger.error(f"An error occurred during Google search")
        return "예기치 않은 오류가 발생했습니다. 관리자에게 문의해주세요"



# LangChain Prompts 날짜, 위치 어케 반영할지 고려 
classify_prompt = ChatPromptTemplate.from_messages(
    [
        SystemMessagePromptTemplate.from_template(
            """ You are an AI assistant that determines whether an input requires a keyword search, navigation routing, or is casual conversation.

            For navigation-related queries:
            - Respond with ONLY 'ROUTE'

            For search-requiring queries:
            - If a search is not needed, respond with ONLY 'False'.
            - Respond with a comma-separated list of up to 3 relevant search keywords in English


            Your response must be either 'ROUTE', 'False', or a list of keywords, without any additional text or punctuation.

            Examples:
            1. Input: What is the stock price of Amazon today?
            Output: Amazon, stock price, today

            2. Input: How do I get to Gangnam Station?
            Output: ROUTE

            3. Input: TWhy is that car interrupting all of a sudden? You didn't turn on the blinkers!
            Output: False

            4. Input: What time does the Lotte World Mall close?
            Output: Lotte World Mall, opening hours, today

            5. Input: My stomach hurts all of a sudden. Is there a place nearby where I can go to the bathroom?
            Output: ROUTE

            6. Input: 여기 근처 맛집 추천해줘
            Output: restaurants, nearby, recommendations

            7. Input: 오늘 기분이 너무 좋아!
            Output: False

            8. Input: 나 오늘 석촌호수에 가기로 했는데, 주변에 아이들과 들를만 한 곳이 있을까?
            Output: Seok-chon lake, kids, visit
           """
        ),
        HumanMessagePromptTemplate.from_template("{question}"),
    ]
)

answer_prompt = ChatPromptTemplate.from_messages(
    [
        SystemMessagePromptTemplate.from_template(
            
"""
You are a smart navigation and voice assistant Cha-Vis that considers user emotions. Follow these instructions carefully:

<Instructions>
당신은 스마트 내비게이션 및 음성 도우미 'Cha-Vis'입니다. 
주 역할은 사용자의 감정을 이해하고 그에 맞춰 길 안내와 정보를 제공하는 것입니다.
아래 <Requirements>를 준수하여 답변하세요.
다음의 추가 정보를 반드시 참고하여 답변하세요 {emotion_label}, {emotion_intensity}, {user_text}
</Instructions>

<Requirements>
1. 사용자의 감정에 공감하며 답변하세요

각 {emotion_label}에 따라 다음과 같이 응답하세요:

HAPPINESS (0):
- 사용자의 긍정적 에너지에 맞춰 밝고 경쾌한 톤 유지

ANGRY (1):
- 자율신경계 안정화를 위한 호흡 유도 등을 통한 안전운전 유도
- 상황 특성에 따른 맞춤 대응
- 예시: "이런 상황에서 화가 나시는 게 당연합니다. 제가 설명하는 동안 천천히 숨 한번 내쉬시면서 들어보세요. 함께 이 상황을 최대한 빨리 해결해드리겠습니다."


DISGUST (2):
- 공감을 표현하되 중립적 입장 유지

FEAR (3):
- 안정감 있는 차분한 목소리로 안내

NEUTRAL (4):
- 명확하고 전문적인 어조 유지
- 간결한 정보 전달

SADNESS (5):
- 따뜻하고 공감적인 톤 사용
- 긍정적인 대안 제시

SURPRISE (6):
- 사용자의 놀람에 공감하는 톤

{emotion_intensity}가 높을수록 해당 감정에 맞는 응답 특성을 강화하여 사용하세요.

</Requirements>

<Note>
부적절하거나 위험한 요청에는 다음과 같이 답변하세요:
"죄송하지만 그런 내용은 답변하기 어렵습니다. 다른 방법으로 도움을 드릴 수 있을까요?"
</Note>

"""
),
        HumanMessagePromptTemplate.from_template("{question}\n\nAdditional info: {additional_info}")
    ]
)


# Conversation Memory
memory = ConversationBufferWindowMemory(k=10, return_messages=True)
# Initialize LangChain models
classify_llm = ChatOpenAI(api_key=openai_api_key, model_name="gpt-4o-mini")
answer_llm = ChatOpenAI(api_key=openai_api_key, model_name="gpt-4o-mini")

# Chains
classify_chain = LLMChain(llm=classify_llm, prompt=classify_prompt)
answer_chain = LLMChain(llm=answer_llm, prompt=answer_prompt, memory=memory)

# API Endpoints
@app.post("/analyze")
async def analyze_request(request: Request):
    data = await request.json()
    audio_path = data["audio_path"]
    user_text = transcribe(audio_path)
    emotion_label, emotion_intensity = analyze_emotion(audio_path)

    classify_result = classify_chain.predict(question=user_text)

    if classify_result.upper() == "ROUTE":
        additional_info = "Route-related logic here."
    elif classify_result.lower() != "false":
        search_keywords = classify_result.split(",")
        search_query = " ".join(search_keywords)
        additional_info = get_google_results(search_query)
    else:
        additional_info = ""

    memory.save_context({"input": user_text}, {"output": additional_info})

    # Corrected code starts here
    messages = answer_prompt.format_messages(
        question=user_text,
        emotion_label=emotion_label,
        emotion_intensity=emotion_intensity,
        additional_info=additional_info,
        user_text=user_text  # 이 줄을 추가합니다.
    )
    response = answer_llm(messages)
    response_content = response.content

    return {
        "user_text": user_text,
        "emotion_label": emotion_label,
        "emotion_intensity": emotion_intensity,
        "additional_info": additional_info,
        "response": response_content,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
