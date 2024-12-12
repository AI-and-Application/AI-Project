import os
import numpy as np
import pandas as pd
import librosa
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from tqdm.auto import tqdm
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from transformers.optimization import AdamW, get_constant_schedule_with_warmup
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, StochasticWeightAveraging
from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer, AutoFeatureExtractor, HubertForSequenceClassification, AutoConfig

# 데이터셋 폴더 및 CSV 파일 목록
DATA_DIR = '/home/user/AZ/hubert/dataset'
#csv_files = ['/home/user/AZ/hubert/dataset/4차년도.csv', '/home/user/AZ/hubert/dataset/5차년도.csv', '/home/user/AZ/hubert/dataset/5차년도_2차.csv']
csv_files = '/home/user/AZ/hubert/dataset/data_1000.csv'

import chardet

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"  # 2번 GPU만 사용

# 파일의 인코딩 감지 함수
def detect_file_encoding(file_path):
    with open(file_path, 'rb') as f:
        result = chardet.detect(f.read())
        return result['encoding']

# CSV 파일의 인코딩 확인
detected_encoding = detect_file_encoding(csv_files)
print(f"{csv_files} 인코딩: {detected_encoding}")

# 'label' 열 전처리 (소문자 변환 및 공백 제거)
combined_metadata = pd.read_csv(csv_files, encoding=detected_encoding)
combined_metadata['label'] = combined_metadata['label'].str.strip().str.lower()

# 감정 레이블 매핑
emotion_mapping = {
    'happiness': 0,
    'angry': 1,
    'disgust': 2,
    'fear': 3,
    'neutral': 4,
    'sadness': 5,
    'surprise': 6
}
combined_metadata['label2'] = combined_metadata['label'].map(emotion_mapping)

# 'wav_id'를 기반으로 파일 경로 생성 함수
def generate_file_path(wav_id, subfolders, base_dir):
    for folder in subfolders:
        file_path = os.path.join(base_dir, folder, f"{wav_id}.wav")
        if os.path.exists(file_path):  # 파일이 존재하는 경우 해당 경로 반환
            return file_path
    return None  # 파일이 없는 경우 None 반환

# 파일 경로 생성 및 필터링
combined_metadata['path'] = combined_metadata['wav_id'].apply(
    lambda x: generate_file_path(x, subfolders, DATA_DIR)
)

# 파일 경로가 None인 행 제거 (존재하지 않는 파일)
combined_metadata = combined_metadata.dropna(subset=['path'])

# 유효한 데이터 통계 출력
print(f"유효한 데이터: {len(combined_metadata)}개의 항목")

# 병합된 데이터를 저장 (선택 사항)
output_csv = os.path.join(DATA_DIR, 'combined_metadata.csv')
combined_metadata.to_csv(output_csv, index=False, encoding='utf-8')
print(f"Combined metadata saved to {output_csv}")

# 정확도 계산 함수
def accuracy(preds, labels):
    return (preds == labels).float().mean()

# 오디오 데이터 로드 함수
def getAudios(df):
    audios = []
    for idx, row in tqdm(df.iterrows(),total=len(df)):
        audio,_ = librosa.load(row['path'],sr=SAMPLING_RATE)
        audios.append(audio)
    return audios

# 데이터셋 클래스
class MyDataset(Dataset):
    def __init__(self, audio, audio_feature_extractor, label2=None, intensity=None):
        if label2 is None:
            label2 = [0] * len(audio)
        if intensity is None:
            intensity = [0] * len(audio)
        
        self.label2 = np.array(label2).astype(np.int64)  # label2는 정수형
        self.intensity = np.array(intensity).astype(np.float32)  # intensity는 실수형
        self.audio = audio
        self.audio_feature_extractor = audio_feature_extractor

    def __len__(self):
        return len(self.label2)

    def __getitem__(self, idx):
        label2 = self.label2[idx]
        intensity = self.intensity[idx]
        audio = self.audio[idx]

        # 오디오 특성 추출
        audio_feature = self.audio_feature_extractor(
            raw_speech=audio, return_tensors='np', sampling_rate=SAMPLING_RATE
        )
        audio_values, audio_attn_mask = audio_feature['input_values'][0], audio_feature['attention_mask'][0]

        item = {
            'label2': label2,
            'intensity': intensity,
            'audio_values': audio_values,
            'audio_attn_mask': audio_attn_mask,
        }

        return item

# 배치 처리 함수
def collate_fn(samples):
    batch_labels = []
    batch_intensities = []  
    batch_audio_values = []
    batch_audio_attn_masks = []

    for sample in samples:
        batch_labels.append(sample['label2'])
        batch_intensities.append(sample['intensity'])  # 'intensity' 추가
        batch_audio_values.append(torch.tensor(sample['audio_values']))
        batch_audio_attn_masks.append(torch.tensor(sample['audio_attn_mask']))

    batch_labels = torch.tensor(batch_labels)
    batch_intensities = torch.tensor(batch_intensities)  # intensity 리스트를 텐서로 변환
    batch_audio_values = pad_sequence(batch_audio_values, batch_first=True)
    batch_audio_attn_masks = pad_sequence(batch_audio_attn_masks, batch_first=True)

    batch = {
        'intensity': batch_intensities,
        'label2': batch_labels,
        'audio_values': batch_audio_values,
        'audio_attn_mask': batch_audio_attn_masks,
    }

    return batch

# Pytorch Lightning 모델 클래스
class MyLitModel(pl.LightningModule):
    def __init__(self, audio_model_name, num_label2s, n_layers=1, projector=True, classifier=True, dropout=0.07, lr_decay=1):
        super(MyLitModel, self).__init__()
        self.config = AutoConfig.from_pretrained(audio_model_name)
        self.config.output_hidden_states = True  # hidden_states 반환 활성화
        self.config.activation_dropout = dropout
        self.config.attention_dropout = dropout
        self.config.final_dropout = dropout
        self.config.hidden_dropout = dropout
        self.config.hidden_dropout_prob = dropout

        self.audio_model = HubertForSequenceClassification.from_pretrained(audio_model_name, config=self.config)
        self.lr_decay = lr_decay

        # label2와 intensity를 위한 추가 classifier
        self.label2_classifier = nn.Linear(self.audio_model.config.hidden_size, num_label2s)
        self.intensity_regressor = nn.Linear(self.audio_model.config.hidden_size, 1)  # intensity는 회귀 문제

        self._do_reinit(n_layers, projector, classifier)

    def forward(self, audio_values, audio_attn_mask=None):
        if audio_attn_mask is not None:
            outputs = self.audio_model(input_values=audio_values, attention_mask=audio_attn_mask)
        else:
            outputs = self.audio_model(input_values=audio_values)

        label2_logits = self.label2_classifier(outputs.hidden_states[-1][:, 0, :])  # label2 logits
        intensity_preds = self.intensity_regressor(outputs.hidden_states[-1][:, 0, :]).squeeze(-1)  # intensity 값
        return label2_logits, intensity_preds

    def training_step(self, batch, batch_idx):
        audio_values = batch['audio_values']
        audio_attn_mask = batch['audio_attn_mask']
        label2s = batch['label2']
        intensities = batch['intensity']

        label2_logits, intensity_preds = self(audio_values, audio_attn_mask)

        # 각각의 loss 계산
        loss_label2 = nn.CrossEntropyLoss()(label2_logits, label2s)
        loss_intensity = nn.MSELoss()(intensity_preds, intensities)
        total_loss = loss_label2 + loss_intensity  # 전체 loss

        preds_label2 = torch.argmax(label2_logits, dim=1)
        acc_label2 = accuracy(preds_label2, label2s)

        self.log('train_loss', total_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('train_acc_label2', acc_label2, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('train_loss_intensity', loss_intensity, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return total_loss

    def validation_step(self, batch, batch_idx):
        audio_values = batch['audio_values']
        audio_attn_mask = batch['audio_attn_mask']
        label2s = batch['label2']
        intensities = batch['intensity']

        label2_logits, intensity_preds = self(audio_values, audio_attn_mask)

        # 각각의 loss 계산
        loss_label2 = nn.CrossEntropyLoss()(label2_logits, label2s)
        loss_intensity = nn.MSELoss()(intensity_preds, intensities)
        total_loss = loss_label2 + loss_intensity  # 전체 loss

        preds_label2 = torch.argmax(label2_logits, dim=1)
        acc_label2 = accuracy(preds_label2, label2s)

        self.log('val_loss', total_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_acc_label2', acc_label2, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_loss_intensity', loss_intensity, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        return total_loss

    def configure_optimizers(self):
        lr = 1e-5
        layer_decay = self.lr_decay
        weight_decay = 0.01
        llrd_params = self._get_llrd_params(lr=lr, layer_decay=layer_decay, weight_decay=weight_decay)
        optimizer = AdamW(llrd_params)
        return optimizer

    def _get_llrd_params(self, lr, layer_decay, weight_decay):
        n_layers = self.audio_model.config.num_hidden_layers
        llrd_params = []
        for name, value in list(self.named_parameters()):
            if ('bias' in name) or ('layer_norm' in name):
                llrd_params.append({"params": value, "lr": lr, "weight_decay": 0.0})
            elif ('emb' in name) or ('feature' in name) :
                llrd_params.append({"params": value, "lr": lr * (layer_decay**(n_layers+1)), "weight_decay": weight_decay})
            elif 'encoder.layer' in name:
                for n_layer in range(n_layers):
                    if f'encoder.layer.{n_layer}' in name:
                        llrd_params.append({"params": value, "lr": lr * (layer_decay**(n_layer+1)), "weight_decay": weight_decay})
            else:
                llrd_params.append({"params": value, "lr": lr , "weight_decay": weight_decay})
        return llrd_params

    def _do_reinit(self, n_layers=0, projector=True, classifier=True):
        if projector:
            self.audio_model.projector.apply(self._init_weight_and_bias)
        if classifier:
            self.audio_model.classifier.apply(self._init_weight_and_bias)

        for n in range(n_layers):
            self.audio_model.hubert.encoder.layers[-(n+1)].apply(self._init_weight_and_bias)

    def _init_weight_and_bias(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.audio_model.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

# 데이터 디렉터리 경로들
DATA_DIR = '/home/user/AZ/hubert/dataset'
PREPROC_DIR = '/home/user/AZ/hubert/preproc'
SUBMISSION_DIR = '/home/user/AZ/hubert/submission'
MODEL_DIR = '/home/user/AZ/hubert/model2'
SAMPLING_RATE = 16000
SEED = 0
N_FOLD = 20
BATCH_SIZE = 8
NUM_LABELS = 7  # 데이터셋에 맞춰 7로 설정

# 오디오 모델 이름
audio_model_name = 'team-lucid/hubert-base-korean'
audio_feature_extractor = AutoFeatureExtractor.from_pretrained(audio_model_name)
audio_feature_extractor.return_attention_mask = True

from sklearn.model_selection import train_test_split

# 학습/검증 데이터 분리
train_df, test_df = train_test_split(combined_metadata, test_size=0.2, random_state=42)

# 오디오 데이터 로드
train_audios = getAudios(train_df)
test_audios = getAudios(test_df)

# 레이블 준비 (label2와 intensity)
train_label2 = train_df['label2'].values
test_label2 = test_df['label2'].values

train_intensity = train_df['intensity'].values
test_intensity = test_df['intensity'].values

# K-Fold Cross Validation을 위한 Stratified KFold 설정
skf = StratifiedKFold(n_splits=N_FOLD, shuffle=True, random_state=SEED)

# StratifiedKFold 분리 작업
for fold_idx, (train_indices, val_indices) in enumerate(skf.split(train_label2, train_label2)):
    # 오디오 데이터 분리
    train_fold_audios = [train_audios[train_index] for train_index in train_indices]
    val_fold_audios = [train_audios[val_index] for val_index in val_indices]

    # 레이블 분리
    train_fold_label2 = train_label2[train_indices]
    val_fold_label2 = train_label2[val_indices]

    train_fold_intensity = train_intensity[train_indices]
    val_fold_intensity = train_intensity[val_indices]

    # 데이터셋 생성
    train_fold_ds = MyDataset(train_fold_audios, audio_feature_extractor, label2=train_fold_label2, intensity=train_fold_intensity)
    val_fold_ds = MyDataset(val_fold_audios, audio_feature_extractor, label2=val_fold_label2, intensity=val_fold_intensity)

    # 데이터로더 생성
    train_fold_dl = DataLoader(train_fold_ds, batch_size=BATCH_SIZE, collate_fn=collate_fn, num_workers=4)
    val_fold_dl = DataLoader(val_fold_ds, batch_size=BATCH_SIZE * 2, collate_fn=collate_fn, num_workers=4)

    # 체크포인트 콜백 설정
    checkpoint_acc_callback = ModelCheckpoint(
        monitor='val_acc_label2',  # label2 정확도를 기준으로 체크포인트 저장
        dirpath=MODEL_DIR,
        filename=f'{fold_idx=}' + '_{epoch:02d}-{val_acc_label2:.4f}',
        save_top_k=1,
        mode='max'
    )

    # 모델 생성
    my_lit_model = MyLitModel(
        audio_model_name=audio_model_name,
        num_label2s=NUM_LABELS,  # label2의 클래스 개수
        n_layers=1, projector=True, classifier=True, dropout=0.07, lr_decay=0.8
    )

    # Trainer 설정
    trainer = pl.Trainer(
        accelerator='cuda',
        devices=[2],
        max_epochs=30,
        precision='16-mixed',
        val_check_interval=0.1,
        callbacks=[checkpoint_acc_callback],
    )

    # 모델 학습
    trainer.fit(my_lit_model, train_fold_dl, val_fold_dl,
    ckpt_path="/home/user/AZ/hubert/model2/fold_idx=3_epoch=01-val_acc_label2=0.4375.ckpt"
    )

    # 모델 삭제
    del my_lit_model
