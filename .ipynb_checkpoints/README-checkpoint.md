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

## LinearAttentionTransformerによる二段階学習

本リポジトリでは、時系列データに対して高効率な処理を実現する LinearAttentionTransformer を用いた二段階学習を実装している。

### 第1段階：自己教師あり事前学習

* **スクリプト**：`LinearAttentionTransformer/pretrain/train.py`
* **使用データ**：Capture-24（約150名の被験者の加速度データ）
* **目的**：人間の活動に関する時系列特徴をラベルなしで事前学習し、Transformerに汎化的な表現力を獲得させること
* **補足**：本リポジトリには、事前学習済みモデルの重みファイルは含まれていない

### 第2段階：下流タスクへのファインチューニング

* **スクリプト**：`LinearAttentionTransformer/pretrain/adl.ipynb`, `pamap.ipynb` など
* **使用データ**：

  * ADL（Activities of Daily Living）タスク
  * PAMAP2 データセット
* **目的**：事前学習済みモデルを初期化に用いることで、少量のラベル付きデータでも高い分類性能を実現する


## 実行方法

### 1. リポジトリのクローン

```bash
git clone https://github.com/Kazuki010101/transformer-timeseries-lab.git
cd transformer-timeseries-lab
```

### 2. モデルの学習・評価

各モデルフォルダ内のノートブックまたはスクリプトを参照：

* `BERT/adl_bert.ipynb`
* `LinearAttentionTransformer/train.py`
* `Roberta/wisdm_roberta.ipynb` など


##  補足事項

* 学習用データは含まれていません。
* 使用データセット：Oxford Wearable SHL Dataset
* モデルは `[バッチサイズ, 時系列長, 3軸 (X, Y, Z)]` のテンソルを前提としています。

## 開発者

Kazuki Okahashi（[@Kazuki010101](https://github.com/Kazuki010101)）

## ライセンス

MIT License
