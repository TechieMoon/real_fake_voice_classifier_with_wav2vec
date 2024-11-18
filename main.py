import os
import torch
import torchaudio
import torchaudio.transforms as T
import numpy as np
from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2Processor, DataCollatorWithPadding, TrainingArguments, Trainer
from sklearn.model_selection import train_test_split
from datasets import Dataset, Features, Value, load_from_disk
from evaluate import load as load_metric

# GPU 사용 가능 여부 확인
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Device:", device)

# 데이터셋 경로 설정 (사용자에 맞게 수정)
real_path = r"C:\Users\bang\Desktop\gpt\Fake Audio/real"  # real 폴더 경로
fake_path = r"C:\Users\bang\Desktop\gpt\Fake Audio/fake"  # fake 폴더 경로

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

# 모든 파일 불러오기
real_files, real_labels = load_data(real_path, 1)  # 진짜 음성 라벨: 1
fake_files, fake_labels = load_data(fake_path, 0)  # 가짜 음성 라벨: 0

all_files = real_files + fake_files
all_labels = real_labels + fake_labels

print("Number of files:", len(all_files))

# 학습용 데이터와 검증용 데이터로 나누기
train_files, test_files, train_labels, test_labels = train_test_split(
    all_files, all_labels, test_size=0.2, random_state=42
)

# Wav2Vec2 모델과 프로세서 불러오기
model_name = "facebook/wav2vec2-base"
processor = Wav2Vec2Processor.from_pretrained(model_name)
model = Wav2Vec2ForSequenceClassification.from_pretrained(
    model_name, num_labels=2
).to(device)

# Resampler 정의 (필요한 경우 사용)
resampler = T.Resample(orig_freq=48000, new_freq=16000)

# 데이터셋 생성 및 처리 - 리샘플링 포함
features = Features({"file_path": Value("string"), "label": Value("int64")})

# 학습 데이터셋 생성
train_data = {"file_path": train_files, "label": train_labels}
train_dataset = Dataset.from_dict(train_data, features=features)

# 테스트 데이터셋 생성
test_data = {"file_path": test_files, "label": test_labels}
test_dataset = Dataset.from_dict(test_data, features=features)

# 전처리된 데이터셋을 저장할 경로
processed_train_dataset_path = "./processed_train_dataset"
processed_test_dataset_path = "./processed_test_dataset"

# 전처리 함수 정의
def preprocess_function(examples):
    speech_list = []
    for file_path in examples["file_path"]:
        speech, sample_rate = torchaudio.load(file_path)
        # 필요하면 리샘플링
        if sample_rate != 16000:
            speech = resampler(speech)
        speech_list.append(speech.squeeze().numpy())
    # 입력 토큰화
    inputs = processor(speech_list, sampling_rate=16000, padding=True)
    # 레이블을 명시적으로 int64로 변환
    inputs["labels"] = [int(label) for label in examples["label"]]
    return inputs

# 학습 데이터셋 처리
if os.path.exists(processed_train_dataset_path):
    train_dataset = load_from_disk(processed_train_dataset_path)
    print("Processed train dataset loaded from disk.")
else:
    train_dataset = train_dataset.map(
        preprocess_function, batched=True, remove_columns=["file_path", "label"]
    )
    train_dataset = train_dataset.cast_column("labels", Value("int64"))
    train_dataset.save_to_disk(processed_train_dataset_path)
    print("Processed train dataset saved to disk.")

# 테스트 데이터셋 처리
if os.path.exists(processed_test_dataset_path):
    test_dataset = load_from_disk(processed_test_dataset_path)
    print("Processed test dataset loaded from disk.")
else:
    test_dataset = test_dataset.map(
        preprocess_function, batched=True, remove_columns=["file_path", "label"]
    )
    test_dataset = test_dataset.cast_column("labels", Value("int64"))
    test_dataset.save_to_disk(processed_test_dataset_path)
    print("Processed test dataset saved to disk.")

# 데이터 타입 확인
print(train_dataset.features)

# 모델 학습을 위한 파라미터 설정
training_args = TrainingArguments(
    output_dir='./results',
    evaluation_strategy='epoch',
    learning_rate=1e-4,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=40,
    save_steps=500,
    save_total_limit=2,
    logging_strategy='steps',     # 로깅 전략을 'steps'로 설정
    logging_steps=50,             # 로깅 빈도를 조정 (예: 50 스텝마다)
    report_to=['tensorboard'],    # TensorBoard에 로그를 기록
)

# 평가 메트릭 로드
metric_accuracy = load_metric("accuracy")
metric_f1 = load_metric("f1")
metric_precision = load_metric("precision")
metric_recall = load_metric("recall")

def compute_metrics(pred):
    labels = pred.label_ids
    preds = np.argmax(pred.predictions, axis=1)
    accuracy = metric_accuracy.compute(predictions=preds, references=labels)
    f1 = metric_f1.compute(predictions=preds, references=labels, average="weighted")
    precision = metric_precision.compute(
        predictions=preds, references=labels, average="weighted"
    )
    recall = metric_recall.compute(predictions=preds, references=labels, average="weighted")
    return {
        "accuracy": accuracy["accuracy"],
        "f1": f1["f1"],
        "precision": precision["precision"],
        "recall": recall["recall"],
    }

# 커스텀 데이터 콜레이터 정의
class CustomDataCollatorWithPadding(DataCollatorWithPadding):
    def __call__(self, features):
        # 기본 데이터 콜레이터 호출
        batch = super().__call__(features)
        # 레이블의 데이터 타입을 torch.long으로 변환
        batch["labels"] = batch["labels"].long()
        return batch

# 데이터 콜레이터 준비
data_collator = CustomDataCollatorWithPadding(tokenizer=processor, padding=True)

# 트레이너 설정
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# 모델 학습
trainer.train()

# 모델 평가
def evaluate(model, files, labels):
    model.eval()
    correct = 0
    total = len(files)

    with torch.no_grad():
        for i, file_path in enumerate(files):
            speech, sample_rate = torchaudio.load(file_path)

            # 샘플링 속도가 16,000이 아닌 경우 리샘플링 적용
            if sample_rate != 16000:
                speech = resampler(speech)

            # 모델의 입력 형식에 맞게 전처리
            inputs = processor(
                speech.squeeze().numpy(),
                sampling_rate=16000,
                return_tensors="pt",
                padding=True,
            ).to(device)

            logits = model(**inputs).logits
            prediction = torch.argmax(logits, dim=-1).item()

            if prediction == labels[i]:
                correct += 1

    accuracy = correct / total
    print(f"Accuracy: {accuracy * 100:.2f}%")

# 테스트 데이터셋 평가
evaluate(model, test_files, test_labels)
