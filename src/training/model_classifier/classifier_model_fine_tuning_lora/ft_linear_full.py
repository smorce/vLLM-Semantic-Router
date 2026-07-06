"""
MMLU-Pro Category Classification — Full Fine-tuning
全パラメータを学習するフルファインチューニング版（LoRA 版 ft_linear_lora.py の派生）。

Usage:
    # 推奨パラメータで学習（CPU 向け）
    python ft_linear_full.py --mode train --model bert-base-uncased --epochs 3 --max-samples 2000

    # カスタム学習率・バッチサイズ
    python ft_linear_full.py --mode train --model modernbert-base --epochs 3 --learning-rate 2e-5 --batch-size 4

    # 学習済みモデルで推論テスト
    python ft_linear_full.py --mode test --model-path full_intent_classifier_modernbert-base

    # デバッグ用クイック学習
    python ft_linear_full.py --mode train --model bert-base-uncased --epochs 1 --max-samples 50

Supported models:
    - mmbert-base: mmBERT base（149M パラメータ、1800+ 言語）
    - bert-base-uncased: BERT base（110M パラメータ、最も安定）
    - roberta-base: RoBERTa base（125M パラメータ）
    - modernbert-base: ModernBERT base（149M パラメータ）

Dataset:
    - TIGER-Lab/MMLU-Pro: 学術問題のカテゴリ分類データセット
    - LLM-Semantic-Router/category-classifier-supplement: 補助データ（"other" カテゴリ強化）

Note:
    フルファインチューニングは LoRA よりメモリ消費が大きい。
    学習率は 1e-5〜3e-5、バッチサイズは小さめから始めることを推奨。
"""

import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

import torch
from datasets import Dataset, load_dataset
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

# 共通ユーティリティ（LoRA 固有関数は使用しない）
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common_lora_utils import (
    clear_gpu_memory,
    get_all_gpu_info,
    log_memory_usage,
    resolve_model_path,
    set_gpu_device,
    setup_logging,
)

logger = setup_logging()

# レガシーモデルと一致させる 14 カテゴリ
REQUIRED_CATEGORIES = [
    "biology",
    "business",
    "chemistry",
    "computer science",
    "economics",
    "engineering",
    "health",
    "history",
    "law",
    "math",
    "other",
    "philosophy",
    "physics",
    "psychology",
]


def create_tokenizer_for_model(model_path: str, base_model_name: str = None):
    """モデル固有の設定でトークナイザーを作成する。"""
    model_identifier = base_model_name or model_path

    if "roberta" in model_identifier.lower():
        logger.info("RoBERTa トークナイザーを add_prefix_space=True で使用")
        return AutoTokenizer.from_pretrained(model_path, add_prefix_space=True)
    return AutoTokenizer.from_pretrained(model_path)


DEFAULT_SUPPLEMENT_DATASET = "LLM-Semantic-Router/category-classifier-supplement"


class MMLU_Dataset:
    """MMLU-Pro カテゴリ分類用データセットローダー（補助データ対応）。"""

    def __init__(
        self,
        dataset_name="TIGER-Lab/MMLU-Pro",
        supplement_dataset: str = DEFAULT_SUPPLEMENT_DATASET,
    ):
        self.dataset_name = dataset_name
        self.supplement_dataset = supplement_dataset
        self.label2id = {}
        self.id2label = {}

    def _load_supplement_data(self) -> list:
        """HuggingFace Hub から補助学習データを読み込む。"""
        if not self.supplement_dataset:
            return []

        try:
            logger.info(f"補助データを読み込み中: {self.supplement_dataset}")
            supplement = load_dataset(self.supplement_dataset)
            data = (
                supplement["train"]
                if "train" in supplement
                else supplement[list(supplement.keys())[0]]
            )
            samples = [(item["text"], item["label"]) for item in data]
            logger.info(f"補助サンプル {len(samples)} 件を読み込み")
            return samples
        except Exception as e:
            logger.warning(f"Failed to load supplement dataset: {e}")
            return []

    def load_huggingface_dataset(self, max_samples=1000):
        """MMLU-Pro を HuggingFace から読み込み、カテゴリバランスを取る。"""
        logger.info(f"データセットを読み込み中: {self.dataset_name}")

        try:
            dataset = load_dataset(self.dataset_name)
            logger.info(f"Dataset splits: {dataset.keys()}")

            all_texts = list(dataset["test"]["question"])
            all_labels = list(dataset["test"]["category"])
            logger.info(f"MMLU-Pro ベースサンプル数: {len(all_texts)}")

            supplement_samples = self._load_supplement_data()
            if supplement_samples:
                supp_texts, supp_labels = zip(*supplement_samples)
                all_texts.extend(supp_texts)
                all_labels.extend(supp_labels)
                logger.info(f"補助サンプル {len(supplement_samples)} 件を追加")

            logger.info(f"合計サンプル数: {len(all_texts)}")

            category_samples = {}
            for text, label in zip(all_texts, all_labels):
                if label not in category_samples:
                    category_samples[label] = []
                category_samples[label].append(text)

            logger.info(
                f"利用可能カテゴリ: {sorted(category_samples.keys())}"
            )
            logger.info(f"必須カテゴリ: {REQUIRED_CATEGORIES}")

            missing_categories = set(REQUIRED_CATEGORIES) - set(category_samples.keys())
            if missing_categories:
                logger.warning(f"データセットに存在しないカテゴリ: {missing_categories}")

            available_required_categories = [
                cat for cat in REQUIRED_CATEGORIES if cat in category_samples
            ]

            min_samples_per_category = max(
                50, max_samples // (len(available_required_categories) * 2)
            )
            target_samples_per_category = max_samples // len(
                available_required_categories
            )

            logger.info(f"利用可能カテゴリ数: {len(available_required_categories)}")
            logger.info(f"カテゴリあたり最小サンプル数: {min_samples_per_category}")
            logger.info(f"カテゴリあたり目標サンプル数: {target_samples_per_category}")

            filtered_texts = []
            filtered_labels = []
            category_counts = {}
            insufficient_categories = []

            for category in available_required_categories:
                if category in category_samples:
                    available_samples = len(category_samples[category])

                    if available_samples < min_samples_per_category:
                        insufficient_categories.append(category)
                        samples_to_take = available_samples
                    else:
                        samples_to_take = min(
                            target_samples_per_category, available_samples
                        )

                    category_texts = category_samples[category][:samples_to_take]
                    filtered_texts.extend(category_texts)
                    filtered_labels.extend([category] * len(category_texts))
                    category_counts[category] = len(category_texts)

            if insufficient_categories:
                logger.warning(
                    f"サンプル不足カテゴリ: {insufficient_categories}"
                )
                for cat in insufficient_categories:
                    logger.warning(
                        f"  {cat}: {category_counts.get(cat, 0)} サンプルのみ"
                    )

            logger.info(f"最終カテゴリ分布: {category_counts}")
            logger.info(f"フィルタ後サンプル数: {len(filtered_texts)}")

            missing_categories = set(available_required_categories) - set(
                category_counts.keys()
            )
            if missing_categories:
                logger.error(
                    f"CRITICAL: サンプルが存在しないカテゴリ: {missing_categories}"
                )

            if len(category_counts) < len(REQUIRED_CATEGORIES) * 0.8:
                logger.error(
                    f"CRITICAL: {len(category_counts)}/{len(REQUIRED_CATEGORIES)} カテゴリのみサンプルあり"
                )
                logger.error(
                    "max_samples を増やすか、別データセットの使用を検討してください。"
                )

            return filtered_texts, filtered_labels

        except Exception as e:
            logger.error(f"Error loading dataset: {e}")
            raise

    def prepare_datasets(self, max_samples=1000):
        """MMLU-Pro から train/validation/test データセットを準備する。"""
        texts, labels = self.load_huggingface_dataset(max_samples)

        unique_labels = sorted(list(set(labels)))
        ordered_labels = [cat for cat in REQUIRED_CATEGORIES if cat in unique_labels]
        extra_labels = [cat for cat in unique_labels if cat not in REQUIRED_CATEGORIES]
        final_labels = ordered_labels + sorted(extra_labels)

        self.label2id = {label: idx for idx, label in enumerate(final_labels)}
        self.id2label = {idx: label for label, idx in self.label2id.items()}

        logger.info(f"カテゴリ数: {len(final_labels)} — {final_labels}")
        logger.info(f"ラベルマッピング: {self.label2id}")

        label_ids = [self.label2id[label] for label in labels]

        train_texts, temp_texts, train_labels, temp_labels = train_test_split(
            texts, label_ids, test_size=0.4, random_state=42, stratify=label_ids
        )

        val_texts, test_texts, val_labels, test_labels = train_test_split(
            temp_texts,
            temp_labels,
            test_size=0.5,
            random_state=42,
            stratify=temp_labels,
        )

        logger.info("データセットサイズ:")
        logger.info(f"  Train: {len(train_texts)}")
        logger.info(f"  Validation: {len(val_texts)}")
        logger.info(f"  Test: {len(test_texts)}")

        return {
            "train": (train_texts, train_labels),
            "validation": (val_texts, val_labels),
            "test": (test_texts, test_labels),
        }


def create_mmlu_dataset(max_samples=1000):
    """MMLU-Pro データセットを作成する。"""
    dataset_loader = MMLU_Dataset()
    datasets = dataset_loader.prepare_datasets(max_samples)

    train_texts, train_labels = datasets["train"]
    val_texts, val_labels = datasets["validation"]

    sample_data = []
    for text, label in zip(train_texts + val_texts, train_labels + val_labels):
        sample_data.append({"text": text, "label": label})

    logger.info(f"データセット作成完了: {len(sample_data)} サンプル")
    logger.info(f"ラベルマッピング: {dataset_loader.label2id}")

    return sample_data, dataset_loader.label2id, dataset_loader.id2label


def create_full_finetune_model(
    model_name: str,
    num_labels: int,
    gradient_checkpointing: bool = True,
):
    """
    フルファインチューニング用モデルとトークナイザーを作成する。
    peft / LoRA は使用せず、全パラメータを学習対象とする。
    """
    logger.info(f"フルファインチューニングモデルを作成: {model_name}")

    tokenizer = create_tokenizer_for_model(model_name, model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        torch_dtype=torch.float32,
        problem_type="single_label_classification",
    )

    if tokenizer.pad_token_id is not None:
        model.config.pad_token_id = tokenizer.pad_token_id

    # 全パラメータを学習対象にする
    trainable_params = 0
    total_params = 0
    for param in model.parameters():
        total_params += param.numel()
        param.requires_grad = True
        trainable_params += param.numel()

    logger.info(
        f"学習可能パラメータ: {trainable_params:,} / {total_params:,} "
        f"({100 * trainable_params / total_params:.1f}%)"
    )

    if gradient_checkpointing:
        model.gradient_checkpointing_enable()
        logger.info("Gradient checkpointing を有効化（メモリ節約）")

    return model, tokenizer


def tokenize_data(data, tokenizer, max_length=512):
    """データをトークナイズする。"""
    texts = [item["text"] for item in data]
    labels = [item["label"] for item in data]

    encodings = tokenizer(
        texts, truncation=True, padding=True, max_length=max_length, return_tensors="pt"
    )

    return Dataset.from_dict(
        {
            "input_ids": encodings["input_ids"],
            "attention_mask": encodings["attention_mask"],
            "labels": labels,
        }
    )


def compute_metrics(eval_pred):
    """評価メトリクスを計算する。"""
    predictions, labels = eval_pred
    predictions = torch.argmax(torch.tensor(predictions), dim=1)

    accuracy = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average="weighted")

    return {"accuracy": accuracy, "f1": f1}


def _update_config_labels(output_dir: str, category_to_idx: Dict, idx_to_category: Dict):
    """config.json にラベルマッピングを書き込み、Rust/Go 互換性を確保する。"""
    config_path = os.path.join(output_dir, "config.json")
    if not os.path.exists(config_path):
        return

    with open(config_path, "r") as f:
        config = json.load(f)

    config["id2label"] = {str(k): v for k, v in idx_to_category.items()}
    config["label2id"] = category_to_idx

    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    logger.info("config.json にラベルマッピングを更新")


def main(
    model_name: str = "modernbert-base",
    num_epochs: int = 3,
    batch_size: int = 4,
    learning_rate: float = 2e-5,
    max_samples: int = 1000,
    output_dir: str = None,
    gradient_checkpointing: bool = True,
    gpu_id: int = None,
):
    """フルファインチューニングのメイン学習関数。"""
    logger.info("フルファインチューニング Intent Classification 学習を開始")

    # GPU 選択
    if gpu_id is not None:
        logger.info(f"指定 GPU を使用: {gpu_id}")
        device_str, selected_gpu = set_gpu_device(gpu_id=gpu_id, auto_select=False)
    else:
        logger.info("空きメモリが最大の GPU を自動選択...")
        device_str, selected_gpu = set_gpu_device(gpu_id=None, auto_select=True)

    all_gpus = get_all_gpu_info()
    if all_gpus:
        logger.info(f"利用可能 GPU 数: {len(all_gpus)}")
        for gpu in all_gpus:
            status = "SELECTED" if gpu["id"] == selected_gpu else "available"
            logger.info(
                f"  GPU {gpu['id']} ({status}): {gpu['name']} - "
                f"{gpu['free_memory_gb']:.2f}GB free / {gpu['total_memory_gb']:.2f}GB total"
            )

    clear_gpu_memory()
    log_memory_usage("Pre-training")

    model_path = resolve_model_path(model_name)
    logger.info(f"使用モデル: {model_name} -> {model_path}")

    # データセット読み込み
    all_data, category_to_idx, idx_to_category = create_mmlu_dataset(max_samples)
    train_data, val_data = train_test_split(all_data, test_size=0.2, random_state=42)

    logger.info(f"学習サンプル数: {len(train_data)}")
    logger.info(f"検証サンプル数: {len(val_data)}")
    logger.info(f"カテゴリ数: {len(category_to_idx)}")

    # フルファインチューニングモデル作成
    model, tokenizer = create_full_finetune_model(
        model_path,
        len(category_to_idx),
        gradient_checkpointing=gradient_checkpointing,
    )

    train_dataset = tokenize_data(train_data, tokenizer)
    val_dataset = tokenize_data(val_data, tokenizer)

    if output_dir is None:
        output_dir = f"full_intent_classifier_{model_name}"
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"モデル保存先: {output_dir}")

    use_fp16 = torch.cuda.is_available()

    # フルファインチューニング向け学習設定（LoRA より低い学習率・小さめバッチ）
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=2,
        learning_rate=learning_rate,
        weight_decay=0.05,
        warmup_ratio=0.06,
        lr_scheduler_type="cosine",
        max_grad_norm=1.0,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_f1",
        greater_is_better=True,
        logging_dir=f"{output_dir}/logs",
        logging_steps=10,
        report_to="tensorboard",
        fp16=use_fp16,
        dataloader_drop_last=False,
        eval_accumulation_steps=1,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )

    logger.info("学習を開始...")
    trainer.train()

    # 完全なモデルとトークナイザーを保存
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    label_mapping = {
        "category_to_idx": category_to_idx,
        "idx_to_category": idx_to_category,
    }
    with open(os.path.join(output_dir, "label_mapping.json"), "w") as f:
        json.dump(label_mapping, f, indent=2)

    with open(os.path.join(output_dir, "category_mapping.json"), "w") as f:
        json.dump(label_mapping, f, indent=2)

    _update_config_labels(output_dir, category_to_idx, idx_to_category)

    logger.info(f"フルファインチューニングモデルを保存: {output_dir}")
    logger.info("label_mapping.json と category_mapping.json を保存")

    logger.info("検証セットで最終評価...")
    val_results = trainer.evaluate()
    logger.info("Validation Results:")
    logger.info(f"  Accuracy: {val_results['eval_accuracy']:.4f}")
    logger.info(f"  F1: {val_results['eval_f1']:.4f}")


def demo_inference(model_path: str, model_name: str = "modernbert-base"):
    """学習済みフルファインチューニングモデルで推論デモ。"""
    logger.info(f"モデルを読み込み中: {model_path}")

    try:
        with open(os.path.join(model_path, "label_mapping.json"), "r") as f:
            mapping_data = json.load(f)
        idx_to_category = {
            int(k): v for k, v in mapping_data["idx_to_category"].items()
        }
        num_labels = len(idx_to_category)

        logger.info(f"ラベル数: {num_labels} — {list(idx_to_category.values())}")

        model = AutoModelForSequenceClassification.from_pretrained(
            model_path, num_labels=num_labels
        )
        tokenizer = AutoTokenizer.from_pretrained(model_path)

        test_examples = [
            "What is the best strategy for corporate mergers and acquisitions?",
            "How do antitrust laws affect business competition?",
            "What are the psychological factors that influence consumer behavior?",
            "Explain the legal requirements for contract formation",
            "What is the difference between civil and criminal law?",
            "How does cognitive bias affect decision making?",
        ]

        logger.info("推論を実行...")
        for example in test_examples:
            inputs = tokenizer(
                example, return_tensors="pt", truncation=True, padding=True
            )

            with torch.no_grad():
                outputs = model(**inputs)
                predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
                predicted_class_id = predictions.argmax().item()
                confidence = predictions[0][predicted_class_id].item()

            predicted_category = idx_to_category[predicted_class_id]
            print(f"Input: {example}")
            print(f"Predicted: {predicted_category} (confidence: {confidence:.4f})")
            print("-" * 50)

    except Exception as e:
        logger.error(f"Error during inference: {e}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Full Fine-tuning Intent Classification"
    )
    parser.add_argument("--mode", choices=["train", "test"], default="train")
    parser.add_argument(
        "--model",
        choices=[
            "mmbert-32k",
            "mmbert-base",
            "modernbert-base",
            "bert-base-uncased",
            "roberta-base",
        ],
        default="mmbert-32k",
    )
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="デバイスあたりのバッチサイズ（フル FT では小さめ推奨）",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=2e-5,
        help="学習率（フル FT では 1e-5〜3e-5 推奨）",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=5000,
        help="MMLU-Pro から取得する最大サンプル数",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="モデル保存先（デフォルト: full_intent_classifier_${model}）",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="full_intent_classifier_modernbert-base",
        help="推論用モデルパス",
    )
    parser.add_argument(
        "--gpu-id",
        type=int,
        default=None,
        help="使用する GPU ID（未指定時は空きメモリ最大の GPU を自動選択）",
    )
    parser.add_argument(
        "--no-gradient-checkpointing",
        action="store_true",
        help="Gradient checkpointing を無効化（メモリに余裕がある場合）",
    )

    args = parser.parse_args()

    if args.mode == "train":
        main(
            model_name=args.model,
            num_epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            max_samples=args.max_samples,
            output_dir=args.output_dir,
            gradient_checkpointing=not args.no_gradient_checkpointing,
            gpu_id=args.gpu_id,
        )
    elif args.mode == "test":
        demo_inference(args.model_path, args.model)
