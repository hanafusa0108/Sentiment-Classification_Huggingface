import argparse
import json
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, EarlyStoppingCallback, default_data_collator
from sklearn.metrics import cohen_kappa_score
import torch
from datasets import Dataset, DatasetDict

# コマンドライン引数または設定ファイルのパース
def parse_args():
    parser = argparse.ArgumentParser(description="Train and evaluate model with transformers.")
    
    # 設定ファイルの引数を追加
    parser.add_argument('--config_file', type=str, required=True, help="Path to the JSON configuration file")
    
    return parser.parse_args()

# 設定ファイルの読み込み
def load_config(config_file):
    with open(config_file, 'r') as file:
        config = json.load(file)
    return config

def preprocess_function(data, tokenizer):
    texts = [q.strip() for q in data["text"]]
    inputs = tokenizer(
        texts,
        max_length=180,
        truncation=True,
        padding='max_length',
        add_special_tokens=True
    )

    # ラベルを2増やす
    modified_labels = [label + 2 for label in data['label']]
    inputs['labels'] = torch.tensor(modified_labels)
    
    return inputs

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    qwk = cohen_kappa_score(labels, preds, weights='quadratic')
    return {"eval_qwk": qwk}

def main():
    # コマンドライン引数を解析
    args = parse_args()

    # 設定ファイルを読み込み
    config = load_config(args.config_file)

    # トークナイザーとモデルの読み込み
    tokenizer = AutoTokenizer.from_pretrained(config["model_name"])

    # データの読み込み
    train_data = pd.read_json(config["train_data"])
    valid_data = pd.read_json(config["valid_data"])

    ds_train = Dataset.from_pandas(train_data)
    ds_valid = Dataset.from_pandas(valid_data)

    dataset = DatasetDict({
        "train": ds_train,
        "validation": ds_valid,
    })

    # データの前処理
    tokenized_data = dataset.map(lambda x: preprocess_function(x, tokenizer), batched=True)

    # デバイス設定
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # モデルの読み込み
    model = AutoModelForSequenceClassification.from_pretrained(config["model_name"], num_labels=config["num_labels"]).to(device)

    # トレーニング設定
    training_args = TrainingArguments(
        output_dir=config["output_dir"],
        warmup_steps=config["warmup_steps"],
        per_device_train_batch_size=config["batch_size"],
        learning_rate=config["learning_rate"],
        num_train_epochs=config["epochs"],
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        lr_scheduler_type="linear",
        metric_for_best_model="eval_qwk",
        fp16=config["fp16"],
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
        save_total_limit=1
    )

    # トレーナーの設定
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_data["train"],
        eval_dataset=tokenized_data["validation"],
        data_collator=default_data_collator,
        tokenizer=tokenizer,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=5)],
        compute_metrics=compute_metrics,
    )

    # トレーニング実行
    trainer.train()

if __name__ == "__main__":
    main()
