import torch
from torch import nn
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import argparse
import json

# コマンドライン引数をパース
def parse_args():
    parser = argparse.ArgumentParser(description="Model inference with transformers.")
    
    # 設定ファイルの引数を追加
    parser.add_argument('--config_file', type=str, required=True, help="Path to the JSON configuration file")
    
    return parser.parse_args()

# 設定ファイルの読み込み
def load_config(config_file):
    with open(config_file, 'r') as file:
        config = json.load(file)
    return config

# 予測関数
def predict(text, tokenizer, model, device):
    inputs = tokenizer(text, add_special_tokens=True, return_tensors="pt").to(device)
    outputs = model(**inputs)
    ps = nn.Softmax(dim=1)(outputs.logits)

    max_p = torch.max(ps)
    result = torch.argmax(ps).item()

    return result

def qwk(labels, pred):
    qwk = cohen_kappa_score(labels, preds, weights='quadratic')
    return {"eval_qwk": qwk}

def main():
    # コマンドライン引数を解析
    args = parse_args()

    # 設定ファイルを読み込み
    config = load_config(args.config_file)

    # デバイス判定
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # モデルとトークナイザーの読み込み
    model = AutoModelForSequenceClassification.from_pretrained(config["model_path"]).to(device)
    tokenizer = AutoTokenizer.from_pretrained(config["model_path"])

    print(model.device)

    y_true = list()
    y_pred = list()

    # テスト用データの読み込み
    with open(config["test_text_file"], 'r') as file:
        lines = file.read().splitlines()

    # 予測実行
    for i in range(len(lines)):
        predicted_label = int(predict(lines[i], tokenizer, model, device)) - 2
        y_pred.append(str(predicted_label))

    # 結果をファイルに書き込み
    with open(config["output_file"], 'w') as file:
        file.write('\n'.join(y_pred))
    
    # QWKで評価
    qwk(predicted_label, y_pred)

if __name__ == "__main__":
    main()
