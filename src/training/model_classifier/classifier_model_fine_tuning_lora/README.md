# Intent Classification 学習スクリプト

MMLU-Pro ベースのカテゴリ分類（Intent Classification）向け学習スクリプト群。
LoRA（パラメータ効率型）とフルファインチューニング（全パラメータ学習）の 2 方式を提供する。

## スクリプト一覧

| ファイル | 方式 | 説明 |
|---------|------|------|
| `ft_linear_lora.py` | LoRA | PEFT + LoRA アダプターで効率的に学習（既存・推奨 for 省メモリ） |
| `ft_linear_full.py` | フル FT | `peft` なしで全パラメータを `Trainer` で学習 |
| `ft_linear_lora_verifier.go` | — | Go 推論検証（LoRA マージ済みモデル向け） |
| `train_cpu_optimized.sh` | LoRA | CPU 向け LoRA 一括学習シェル |

詳細な比較・使い分けは本 README の [LoRA とフルファインチューニングの比較](#lora-とフルファインチューニングの比較) を参照。

## フルファインチューニング（`ft_linear_full.py`）

### 背景

公開リポジトリの Intent Classification 学習は LoRA 向けに整理されているが、
フルファインチューニングも既存コードを少し改修するだけで実現できる。

**要点:** `peft` を使う部分を外し、`AutoModelForSequenceClassification` をそのまま `Trainer` に渡す。

`ft_linear_full.py` は `ft_linear_lora.py` をベースに、LoRA 固有の import・設定・保存処理を除いた派生スクリプトである。

### LoRA 版からの主な変更点

| 項目 | LoRA (`ft_linear_lora.py`) | フル FT (`ft_linear_full.py`) |
|------|---------------------------|-------------------------------|
| モデル作成 | `get_peft_model()` + `LoraConfig` | `AutoModelForSequenceClassification.from_pretrained()` |
| 学習対象 | LoRA アダプターのみ（~0.02%） | 全パラメータ（100%） |
| デフォルト学習率 | `3e-5` | `2e-5`（1e-5〜3e-5 推奨） |
| デフォルトバッチサイズ | `8` | `4`（メモリ不足時は `2`） |
| メモリ節約 | LoRA 自体が省メモリ | `gradient_checkpointing_enable()` |
| Trainer | `EnhancedLoRATrainer`（LoRA 正則化付き） | 標準 `Trainer` |
| 保存物 | LoRA アダプター（マージは別処理） | 完全なモデル一式 |
| 出力ディレクトリ | `lora_intent_classifier_{model}_r{rank}` | `full_intent_classifier_{model}` |
| 依存 | `peft` 必須 | `peft` 不要 |

### 実装の要点

**削除したもの（LoRA 固有）:**

```python
from peft import LoraConfig, get_peft_model, TaskType, PeftModel

lora_config = create_lora_config(...)
model = create_lora_model(model_path, num_labels, lora_config)
```

**追加したもの（フル FT）:**

```python
model, tokenizer = create_full_finetune_model(
    model_path,
    len(category_to_idx),
    gradient_checkpointing=True,
)
```

`create_full_finetune_model` の処理内容:

- `AutoModelForSequenceClassification.from_pretrained(..., problem_type="single_label_classification")`
- 全パラメータの `requires_grad = True`
- 任意で `gradient_checkpointing_enable()`（デフォルト有効）
- `pad_token` / `pad_token_id` の設定

### 使い方

```bash
cd src/training/model_classifier/classifier_model_fine_tuning_lora

# 学習
python ft_linear_full.py \
  --mode train \
  --model modernbert-base \
  --epochs 3 \
  --max-samples 5000

# 推論テスト
python ft_linear_full.py \
  --mode test \
  --model-path full_intent_classifier_modernbert-base
```

### 主要オプション

| オプション | デフォルト | 説明 |
|-----------|-----------|------|
| `--model` | `mmbert-32k` | ベースモデル（`bert-base-uncased`, `roberta-base`, `modernbert-base` 等） |
| `--epochs` | `3` | エポック数 |
| `--batch-size` | `4` | デバイスあたりバッチサイズ |
| `--learning-rate` | `2e-5` | 学習率（フル FT では 1e-5〜3e-5） |
| `--max-samples` | `5000` | MMLU-Pro から取得する最大サンプル数 |
| `--output-dir` | `full_intent_classifier_{model}` | 保存先 |
| `--gpu-id` | 自動選択 | 使用 GPU ID |
| `--no-gradient-checkpointing` | — | メモリに余裕がある場合に checkpointing を無効化 |

### 保存されるファイル

```
full_intent_classifier_{model}/
├── config.json              # id2label / label2id 付き（Rust/Go 互換）
├── model.safetensors        # 完全なモデル重み
├── tokenizer files...
├── label_mapping.json
├── category_mapping.json    # Go verifier 互換
└── logs/                    # TensorBoard ログ
```

### 学習設定（フル FT 向けデフォルト）

```python
TrainingArguments(
    learning_rate=2e-5,
    weight_decay=0.05,
    warmup_ratio=0.06,
    lr_scheduler_type="cosine",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=2,
    max_grad_norm=1.0,
    fp16=True,                  # CUDA 利用時のみ（CPU では False）
    eval_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="eval_f1",
)
```

## LoRA 学習（`ft_linear_lora.py`）

```bash
# 手動学習
python ft_linear_lora.py \
  --mode train \
  --model bert-base-uncased \
  --epochs 3 \
  --lora-rank 16 \
  --max-samples 5000

# CPU 一括学習
./train_cpu_optimized.sh
```

## LoRA とフルファインチューニングの比較

| 項目 | LoRA | フルファインチューニング |
|------|------|------------------------|
| 学習率 | `2e-4` 前後も可 | `1e-5`〜`3e-5` |
| バッチサイズ | やや大きめ可 | 小さめから開始 |
| 学習対象 | 追加パラメータ中心 | 全パラメータ |
| メモリ | 少ない | 多い |
| 保存物 | adapter 中心（マージ別途） | 完全なモデル |
| 過学習リスク | 比較的低い | 高め |
| 推奨用途 | CPU / 単一 GPU / 試行錯誤 | GPU メモリに余裕があり最高精度を狙う場合 |

### 使い分けの目安

- **LoRA を選ぶ:** メモリが限られる、複数モデルを試したい、既存の LoRA 推論パイプラインを使う
- **フル FT を選ぶ:** GPU メモリに余裕がある、マージ不要の完全モデルが欲しい、LoRA より高い適応精度を狙う

### 注意点

フルファインチューニングは LoRA よりメモリ消費が大きい。
同じバッチサイズ・学習率のまま置き換えると OOM や学習不安定が起きやすい。

メモリ不足時の対処:

1. `--batch-size 2` に下げる
2. `--no-gradient-checkpointing` は付けない（デフォルトの checkpointing を維持）
3. それでも足りなければ LoRA 版（`ft_linear_lora.py`）を使用

## データセット

両スクリプト共通:

- **主データ:** [TIGER-Lab/MMLU-Pro](https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro)（14 カテゴリ）
- **補助データ:** `LLM-Semantic-Router/category-classifier-supplement`（`other` カテゴリ強化）

## 対応モデル

- `mmbert-32k` — mmBERT-32K YaRN（32K コンテキスト、推奨）
- `mmbert-base` — 多言語 mmBERT（1800+ 言語）
- `modernbert-base` — ModernBERT
- `bert-base-uncased` — BERT（CPU 向け・最も安定）
- `roberta-base` — RoBERTa

## 参考

- LoRA 論文: [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
- Hugging Face Trainer: [Fine-tuning with the Trainer API](https://huggingface.co/learn/llm-course/en/chapter3/3)
- 親ディレクトリ README: [`../README.md`](../README.md)
