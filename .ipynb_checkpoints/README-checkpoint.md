# transformer-timeseries-lab

人間活動認識（HAR）やセンサデータ分類に対して、BERTやRoBERTa、LinearAttentionTransformer など複数のTransformer系モデルを適用した実験コード群です。

##  プロジェクト構成

* `BERT/`：時系列対応BERTモデルのファインチューニング実験
* `LinearAttentionTransformer/`：線形注意機構を用いた軽量なTransformerによる分類（[lucidrains/linear-attention-transformer](https://github.com/lucidrains/linear-attention-transformer) を参考）
* `Roberta/`：加速度センサーデータを用いたRoBERTaベースの下流タスク

##  タスク概要

このリポジトリは、加速度センサーなどの時系列データを対象とした人間活動認識（HAR）タスクにTransformerベースのモデルを適用することを目的としています。

* **入力**：X, Y, Z の3軸加速度データ（例：WISDM, PAMAP2, RealWorld など）
* **出力**：活動ラベル（例：歩行、座位、立位など）

## 実行方法

### 1. リポジトリのクローン

```bash
git clone https://github.com/Kazuki010101/transformer-timeseries-lab.git
cd transformer-timeseries-lab
```

### 2. モデルの学習・評価

各モデルフォルダ内のノートブックまたはスクリプトを参照：

* `BERT/bert_finetune.ipynb`
* `LinearAttentionTransformer/train.py`
* `Roberta/wisdm_robert.ipynb` など


##  補足事項

* 学習用データは含まれていません。
* 使用データセット：Oxford Wearable SHL Dataset
* モデルは `[バッチサイズ, 時系列長, 3軸 (X, Y, Z)]` のテンソルを前提としています。

## 開発者

Kazuki Okahashi（[@Kazuki010101](https://github.com/Kazuki010101)）

## ライセンス

MIT License
