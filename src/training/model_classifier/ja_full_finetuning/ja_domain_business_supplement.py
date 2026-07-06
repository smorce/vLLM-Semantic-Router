"""
business / economics 混同を改善するための追加ファインチューニング。

既存の ja-full-domain-classifier チェックポイントを読み込み、
企業経営・法務・M&A 系の business サンプルと economics 対照サンプルで
短い追加学習を行う。

Usage:
    python ja_domain_business_supplement.py \
      --model-path ../../../../models/ja-full-domain-classifier-ruri-v3-30m \
      --epochs 3
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common_lora_utils import clear_gpu_memory, log_memory_usage, setup_logging

logger = setup_logging()

# verify_domain で誤分類されやすい business 文と economics 対照文
BUSINESS_SUPPLEMENT: list[tuple[str, str]] = [
    ("企業合併における最良の戦略は何ですか？", "business"),
    ("独占禁止法は企業間の競争にどのような影響を与えますか？", "business"),
    ("B2B SaaS の価格戦略と顧客獲得コストのトレードオフを整理してください。", "business"),
    ("小売チェーン展開における事業戦略とマーケティング戦略の関係を説明してください。", "business"),
    ("MBAケーススタディとして、製造業サプライチェーンの在庫最適化を経営視点で分析してください。", "business"),
    ("新規事業の市場参入戦略と競合分析のフレームワークを教えてください。", "business"),
    ("組織再編における人材配置とコスト削減のバランスをどう取るべきですか？", "business"),
    ("フランチャイズビジネスの収益モデルとロイヤリティ設計について説明してください。", "business"),
    ("企業のブランド戦略と顧客ロイヤルティ向上施策を整理してください。", "business"),
    ("M&A後のPMI（統合）で経営シナジーを最大化する方法は？", "business"),
    ("スタートアップの資金調達ラウンドとバリュエーション交渉の要点は？", "business"),
    ("サプライチェーンリスク管理を経営会議向けに要約してください。", "business"),
    ("デジタルトランスフォーメーション投資のROIを経営指標で評価する方法は？", "business"),
    ("従業員エンゲージメント向上のための経営施策を提案してください。", "business"),
    ("グローバル展開における現地法人のガバナンス体制を設計するポイントは？", "business"),
]

ECONOMICS_CONTRAST: list[tuple[str, str]] = [
    ("需要と供給の経済原則を説明してください。", "economics"),
    ("GDP成長率とインフレ率の関係について教えてください。", "economics"),
    ("完全競争市場と独占市場の経済学的な違いは何ですか？", "economics"),
    ("ケインズ経済学と古典派経済学の主な相違点を説明してください。", "economics"),
    ("金融政策が金利と物価に与える影響を経済理論で整理してください。", "economics"),
    ("限界効用逓減の法則を具体例で説明してください。", "economics"),
    ("為替レート変動が輸出入に与えるマクロ経済的影響は？", "economics"),
    ("失業率とフィリップス曲線の関係を説明してください。", "economics"),
]

# business サンプルを economics 対照よりやや多めにオーバーサンプリング
REPEAT_BUSINESS = 2

# business 以外のカテゴリも少量混ぜて過学習を防ぐ
OTHER_CONTRAST: list[tuple[str, str]] = [
    ("消費者行動に影響を与える心理的要因は何ですか？", "psychology"),
    ("契約成立の法的要件を説明してください。", "law"),
    ("光合成の仕組みを説明してください。", "biology"),
    ("eのx乗の微分は何ですか？", "math"),
    ("コンピュータのトランジスタの仕組みを説明してください。", "computer science"),
    ("星が瞬いて見えるのはなぜですか？", "physics"),
    ("ローマ帝国の歴史的意義を説明してください。", "history"),
    ("フランスの首都はどこですか？", "other"),
]


def load_label_mapping(model_path: str) -> tuple[dict[str, int], dict[int, str]]:
    mapping_path = Path(model_path) / "category_mapping.json"
    with open(mapping_path, encoding="utf-8") as f:
        data = json.load(f)
    label2id = data["category_to_idx"]
    id2label = {int(k): v for k, v in data["idx_to_category"].items()}
    return label2id, id2label


def build_samples(label2id: dict[str, int]) -> list[dict]:
    samples: list[dict] = []
    for text, label in BUSINESS_SUPPLEMENT * REPEAT_BUSINESS + ECONOMICS_CONTRAST + OTHER_CONTRAST:
        if label not in label2id:
            raise ValueError(f"unknown label {label!r} in mapping")
        samples.append({"text": text, "label": label2id[label]})
    return samples


def tokenize_data(samples: list[dict], tokenizer, max_length: int = 512) -> Dataset:
    texts = [s["text"] for s in samples]
    labels = [s["label"] for s in samples]
    encodings = tokenizer(
        texts,
        truncation=True,
        padding=True,
        max_length=max_length,
        return_tensors="pt",
    )
    return Dataset.from_dict(
        {
            "input_ids": encodings["input_ids"],
            "attention_mask": encodings["attention_mask"],
            "labels": labels,
        }
    )


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = torch.argmax(torch.tensor(predictions), dim=1)
    return {
        "accuracy": accuracy_score(labels, predictions),
        "f1": f1_score(labels, predictions, average="weighted"),
    }


def main(
    model_path: str,
    epochs: int = 2,
    batch_size: int = 4,
    learning_rate: float = 5e-6,
    max_length: int = 512,
):
    logger.info("business/economics 追加ファインチューニングを開始")
    clear_gpu_memory()
    log_memory_usage("Pre-training")

    label2id, id2label = load_label_mapping(model_path)
    all_samples = build_samples(label2id)
    train_data, val_data = train_test_split(all_samples, test_size=0.15, random_state=42)
    logger.info(f"学習: {len(train_data)}, 検証: {len(val_data)}")

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForSequenceClassification.from_pretrained(
        model_path,
        num_labels=len(label2id),
    )

    train_ds = tokenize_data(train_data, tokenizer, max_length)
    val_ds = tokenize_data(val_data, tokenizer, max_length)

    use_fp16 = torch.cuda.is_available()
    training_args = TrainingArguments(
        output_dir=f"{model_path}/business_supplement_checkpoints",
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_f1",
        greater_is_better=True,
        logging_steps=5,
        report_to="none",
        fp16=use_fp16,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    trainer.save_model(model_path)
    tokenizer.save_pretrained(model_path)

    val_results = trainer.evaluate()
    logger.info(f"Validation Accuracy: {val_results['eval_accuracy']:.4f}")
    logger.info(f"Validation F1: {val_results['eval_f1']:.4f}")
    logger.info(f"モデルを上書き保存: {model_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="business/economics 追加学習")
    parser.add_argument(
        "--model-path",
        default="../../../../models/ja-full-domain-classifier-ruri-v3-30m",
        help="既存 domain 分類器のパス",
    )
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=5e-6)
    args = parser.parse_args()
    main(
        model_path=args.model_path,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
    )
