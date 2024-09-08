# Sentiment-Classification
- 日本語のTwitterテキストについての感情極性分類
* タスク
  * 書き手の感情極性を5クラスで分類（-2, -1, 0, 1, 2）
  * 評価指標： Quadratic Weighted Kappa

* データセット
  * データセットは [WRIME](https://github.com/ids-cv/wrime)
  * 訓練用：30,000件、検証用：1,250件、評価用：1,250件

* モデル
  * [studio-ousia/luke-japanese-large](https://huggingface.co/studio-ousia/luke-japanese-large)

- 実行手順
1. trainのフォルダをダウンロードし、以下のように実行<br>
   ```python luke_train.py --config_file train_config.json```

2. testのフォルダをダウンロードし、以下のように実行<br>
   ```python luke_test.py --config_file test_config.json```

