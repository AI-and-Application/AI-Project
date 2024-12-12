import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import VitsModel, AutoTokenizer
from torch import nn, optim
from torch.nn import functional as F

class EmotionalVitsModel(nn.Module):
    def __init__(self, base_model_path="facebook/mms-tts-kor"):
        super().__init__()
        print("기본 VITS 모델 로드 중...")
        self.base_model = VitsModel.from_pretrained(base_model_path)
        print("기존 TTS 가중치 동결 중...")
        # 기존 모델의 가중치를 동결
        for param in self.base_model.parameters():
            param.requires_grad = False
            
        print("감정 레이어 초기화 중...")
        # 감정 관련 레이어만 새로 초기화
        self.emotion_embedding = nn.Embedding(3, 192)
        self.emotion_projection = nn.Sequential(
            nn.Linear(192, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Tanh()
        )
        print("모델 준비 완료")

    def forward(self, input_ids, attention_mask, emotion):
        # 기본 모델로 출력 생성
        with torch.no_grad():
            base_output = self.base_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True
            )
        
        # 감정 임베딩 및 변조 계수 계산
        emotion_emb = self.emotion_embedding(emotion)  # [batch_size, 192]
        emotion_factor = self.emotion_projection(emotion_emb)  # [batch_size, 1]
        
        # 원본 waveform 가져오기
        waveform = base_output.waveform  # [batch_size, 1, time]
        
        # 감정 변조 적용
        modified_waveform = waveform * (1.0 + emotion_factor.unsqueeze(-1))
        
        # 출력 생성
        output = base_output
        output.waveform = modified_waveform
        
        return output

class EmotionalVitsDataset(Dataset):
    def __init__(self, base_path, fixed_text="안녕하세요, 차비스 전용 티티에스입니다", segment_length=68608):  # 모델 출력 길이에 맞춤
        self.base_path = base_path
        self.fixed_text = fixed_text
        self.segment_length = segment_length
        self.emotion_map = {'neutral': 0, 'happy': 1, 'sad': 2}
        self.tokenizer = AutoTokenizer.from_pretrained("facebook/mms-tts-kor")
        self.samples = []
        
        for emotion in self.emotion_map.keys():
            emotion_path = os.path.join(base_path, emotion)
            if os.path.exists(emotion_path):
                for file in os.listdir(emotion_path):
                    if file.endswith('.raw'):
                        self.samples.append({
                            'path': os.path.join(emotion_path, file),
                            'emotion': self.emotion_map[emotion]
                        })
        print(f"총 {len(self.samples)}개의 샘플을 찾았습니다.")

    def pad_or_trim(self, audio):
        """오디오를 모델 출력 길이에 맞게 조정"""
        if len(audio) > self.segment_length:
            # 중앙 부분을 사용
            start = (len(audio) - self.segment_length) // 2
            audio = audio[start:start + self.segment_length]
        else:
            # 패딩
            padding = self.segment_length - len(audio)
            audio = F.pad(audio, (0, padding), 'constant', 0)
        return audio

    def load_raw_audio(self, file_path):
        audio_data = np.fromfile(file_path, dtype=np.int16)
        audio_data = audio_data.astype(np.float32) / 32768.0
        return torch.FloatTensor(audio_data)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        text_tokens = self.tokenizer(self.fixed_text, return_tensors="pt")
        
        audio = self.load_raw_audio(sample['path'])
        audio = self.pad_or_trim(audio)
        
        emotion = torch.tensor(sample['emotion'])
        
        return {
            'input_ids': text_tokens['input_ids'].squeeze(),
            'attention_mask': text_tokens['attention_mask'].squeeze(),
            'audio': audio,
            'emotion': emotion
        }



def train_emotional_tts(base_path, num_epochs=30, batch_size=128, learning_rate=1e-4):
    print("데이터 로딩 중...")
    dataset = EmotionalVitsDataset(base_path)
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=32,
        pin_memory=True
    )
    
    print("감정 TTS 모델 초기화 중...")
    model = EmotionalVitsModel().cuda()
    print("기존 TTS 가중치는 유지된 상태이며, 감정 레이어만 학습됩니다.")
    
    optimizer = optim.AdamW([
        {'params': model.emotion_embedding.parameters()},
        {'params': model.emotion_projection.parameters()}
    ], lr=learning_rate)
    
    print(f"학습 시작 (총 {num_epochs} 에폭)")
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        
        for batch_idx, batch in enumerate(dataloader):
            optimizer.zero_grad()
            
            input_ids = batch['input_ids'].cuda()
            attention_mask = batch['attention_mask'].cuda()
            audio = batch['audio'].cuda()
            emotion = batch['emotion'].cuda()
            
            output = model(input_ids, attention_mask, emotion)
            
            # 오디오 길이 맞추기
            target_audio = audio
            if output.waveform.shape[-1] != target_audio.shape[-1]:
                target_audio = F.pad(target_audio, (0, output.waveform.shape[-1] - target_audio.shape[-1]))
            
            loss = F.l1_loss(output.waveform.mean(dim=1), target_audio)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            if batch_idx % 10 == 0:
                print(f"에폭 {epoch+1}/{num_epochs}, 배치 {batch_idx}/{len(dataloader)}, Loss: {loss.item():.4f}")
        
        avg_loss = total_loss / len(dataloader)
        print(f"에폭 {epoch+1}/{num_epochs} 완료, 평균 Loss: {avg_loss:.4f}")
        
        # 수정된 체크포인트 저장
        save_path = os.path.join(base_path, f'emotional_vits_checkpoint_epoch_{epoch+1}.pth')
        print(f"체크포인트 저장 경로: {save_path}")
        
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss,
        }, save_path)
        print(f"체크포인트 저장됨: {save_path}")


if __name__ == "__main__":
    base_path = "/home/user/AZ/tts_emotion"
    print("감정 TTS 학습 시작")
    train_emotional_tts(base_path)
