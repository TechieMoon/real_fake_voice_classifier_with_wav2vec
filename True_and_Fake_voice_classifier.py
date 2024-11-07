# True and Fake Voice Classifier

# Cell 1
from google.colab import drive
drive.mount('/content/drive')


# Cell 2
!pip install fsspec==2024.10.0
!pip install transformers datasets

import os
import torch
import torchaudio
import torchaudio.transforms as T
import numpy as np
from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2Processor, DataCollatorWithPadding
from sklearn.model_selection import train_test_split
from datasets import Dataset
!pip install evaluate
from evaluate import load as load_metric


# Cell 3
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Cell 4
real_path = "/content/drive/MyDrive/Colab Notebooks/wav2vec classification/real"  # real 폴더 경로
fake_path = "/content/drive/MyDrive/Colab Notebooks/wav2vec classification/fake"  # fake 폴더 경로

# Cell 5
# 데이터셋 불러오기 및 라벨링
def load_data(folder_path, label, max_files=None):
    files = []
    labels = []
    for i, file in enumerate(os.listdir(folder_path)):
        if file.endswith('.ogg'):
            files.append(os.path.join(folder_path, file))
            labels.append(label)
        if max_files is not None and i + 1 >= max_files:
            break
    return files, labels

# 데이터셋을 적은 수의 샘플로 불러오기 (각 폴더에서 최대 5개 파일만 사용)
real_files, real_labels = load_data(real_path, 1, max_files=5)  # 진짜 음성 라벨: 1, 최대 5개 파일
fake_files, fake_labels = load_data(fake_path, 0, max_files=5)  # 가짜 음성 라벨: 0, 최대 5개 파일

all_files = real_files + fake_files
all_labels = real_labels + fake_labels

# Cell 6
train_files, test_files, train_labels, test_labels = train_test_split(all_files, all_labels, test_size=0.2, random_state=42)


# Cell 7
model_name = "facebook/wav2vec2-base"
processor = Wav2Vec2Processor.from_pretrained(model_name)
model = Wav2Vec2ForSequenceClassification.from_pretrained(model_name, num_labels=2).to(device)


# Cell 8
# Resampler 정의 (32,000 Hz -> 16,000 Hz)
resampler = T.Resample(orig_freq=32000, new_freq=16000)

# 데이터셋 생성 및 처리 - 재샘플링 포함
def preprocess(file_path):
    speech, sample_rate = torchaudio.load(file_path)
    if sample_rate != 16000:
        speech = resampler(speech)
    return {'speech': speech.squeeze().numpy()}

data = {'file_path': train_files, 'label': train_labels}
dataset = Dataset.from_dict(data)
dataset = dataset.map(lambda example: {
    'input_values': processor(preprocess(example['file_path'])['speech'], sampling_rate=16000, return_tensors="pt", padding=True).input_values.squeeze(),
    'label': example['label']
})

# Cell 9
## 양자화는 나중에

# Cell 10
def apply_low_rank_approximation(model, rank=16):
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            # 저랭크 근사 적용
            weight = module.weight.data.cpu().numpy()
            U, S, Vt = np.linalg.svd(weight, full_matrices=False)
            U = U[:, :rank]
            S = np.diag(S[:rank])
            Vt = Vt[:rank, :]
            new_weight = torch.tensor(U @ S @ Vt, dtype=module.weight.dtype)
            module.weight.data = new_weight.to(module.weight.device)

apply_low_rank_approximation(model)


# Cell 11
from torch.utils.data import DataLoader
from transformers import TrainingArguments, Trainer

training_args = TrainingArguments(
    output_dir='./results',
    evaluation_strategy='epoch',
    learning_rate=1e-4,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    save_steps=10,
    save_total_limit=2,
    report_to='none'  # W&B 사용 비활성화
)


# Cell 12
train_dataloader = DataLoader(dataset, batch_size=8, shuffle=True)


# Cell 13
import numpy as np
from transformers import Trainer

metric = load_metric("accuracy")

def compute_metrics(pred):
    labels = pred.label_ids
    preds = np.argmax(pred.predictions, axis=1)
    return metric.compute(predictions=preds, references=labels)

# DataCollatorWithPadding을 사용하여 데이터의 길이를 맞춤
data_collator = DataCollatorWithPadding(tokenizer=processor, padding=True)

# 트레이너 설정
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    eval_dataset=dataset,
    data_collator=data_collator,
    tokenizer=processor,
    compute_metrics=compute_metrics
)



# Cell 14
trainer.train()

# Cell 15
# 테스트 데이터셋 평가함수 생성
def evaluate(model, files, labels):
    model.eval()
    correct = 0
    total = len(files)

    with torch.no_grad():
        for i, file_path in enumerate(files):
            speech, sample_rate = torchaudio.load(file_path)

            # 샘플링 속도가 16,000이 아닌 경우 재샘플링 적용
            if sample_rate != 16000:
                speech = resampler(speech)

            # 모델의 입력 형식에 맞게 전처리
            inputs = processor(speech.squeeze().numpy(), sampling_rate=16000, return_tensors="pt", padding=True).to(device)

            logits = model(**inputs).logits
            prediction = torch.argmax(logits, dim=-1).item()

            if prediction == labels[i]:
                correct += 1

    accuracy = correct / total
    print(f"Accuracy: {accuracy * 100:.2f}%")



# 평가
evaluate(model, test_files, test_labels)


