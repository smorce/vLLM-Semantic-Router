"""
JMMLU 分野分類（ドメイン分類）— 日本語フルファインチューニング

`classifier_model_fine_tuning_lora/ft_linear_full.py`（MMLU-Pro 英語版）の日本語派生版。
ベースモデルは ModernBERT-Ja 系の `cl-nagoya/ruri-v3-30m` を想定し、全パラメータを学習する。

Usage:
    # 学習（推奨パラメータ）
    python ja_domain_classifier_full.py --mode train --model ruri-v3-30m --epochs 3 --max-samples 3000

    # 推論テスト
    python ja_domain_classifier_full.py --mode test --model-path ja_full_domain_classifier_ruri-v3-30m

Dataset:
    - nlp-waseda/JMMLU: 日本語 MMLU（56 科目、7,536 問の四択問題）
      各科目（config）を読み込み、英語版 MMLU-Pro と同じ 14 カテゴリへマッピングする。

Note:
    ラベル体系は既存の英語版（REQUIRED_CATEGORIES, 14 カテゴリ）とそろえてあるため、
    将来的に英語モデルと日本語モデルを同じ Go/Rust 推論パイプラインで扱いやすい。
"""

import argparse
import csv
import io
import json
import os
import sys
import zipfile
from pathlib import Path
from typing import Dict, List, Tuple

import requests
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
from common_lora_utils import (
    clear_gpu_memory,
    get_all_gpu_info,
    log_memory_usage,
    resolve_model_path,
    set_gpu_device,
    setup_logging,
)

logger = setup_logging()

# 英語版（ft_linear_full.py）と同じ 14 カテゴリに揃える
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

JMMLU_DATASET_NAME = "nlp-waseda/JMMLU"
JMMLU_ZIP_URL = (
    "https://huggingface.co/datasets/nlp-waseda/JMMLU/resolve/main/JMMLU.zip"
)
JMMLU_ZIP_CACHE_PATH = "jmmlu_dataset.zip"

# JMMLU の 56 科目（HuggingFace の config 名）から 14 カテゴリへのマッピング。
# 分類粒度は TIGER-Lab/MMLU-Pro が採用した MMLU 科目 → カテゴリの標準的な括り方に準拠。
JMMLU_SUBJECT_TO_CATEGORY: Dict[str, str] = {
    "japanese_history": "history",
    "miscellaneous": "other",
    "security_studies": "other",
    "virology": "biology",
    "nutrition": "health",
    "human_sexuality": "psychology",
    "college_mathematics": "math",
    "japanese_civics": "other",
    "econometrics": "economics",
    "computer_security": "computer science",
    "clinical_knowledge": "health",
    "machine_learning": "computer science",
    "high_school_chemistry": "chemistry",
    "human_aging": "health",
    "logical_fallacies": "philosophy",
    "sociology": "other",
    "high_school_european_history": "history",
    "high_school_statistics": "math",
    "high_school_physics": "physics",
    "high_school_microeconomics": "economics",
    "college_physics": "physics",
    "anatomy": "health",
    "high_school_psychology": "psychology",
    "business_ethics": "business",
    "professional_psychology": "psychology",
    "college_medicine": "health",
    "elementary_mathematics": "math",
    "moral_disputes": "philosophy",
    "marketing": "business",
    "high_school_macroeconomics": "economics",
    "world_religions": "philosophy",
    "conceptual_physics": "physics",
    "professional_medicine": "health",
    "prehistory": "history",
    "high_school_mathematics": "math",
    "international_law": "law",
    "philosophy": "philosophy",
    "japanese_idiom": "other",
    "japanese_geography": "other",
    "management": "business",
    "high_school_computer_science": "computer science",
    "medical_genetics": "health",
    "college_computer_science": "computer science",
    "public_relations": "business",
    "professional_accounting": "business",
    "abstract_algebra": "math",
    "global_facts": "other",
    "college_biology": "biology",
    "high_school_geography": "other",
    "world_history": "history",
    "high_school_biology": "biology",
    "college_chemistry": "chemistry",
    "electrical_engineering": "engineering",
    "astronomy": "physics",
    "jurisprudence": "law",
    "formal_logic": "philosophy",
}

CHOICE_LABELS = ["A", "B", "C", "D"]


def create_tokenizer_for_model(model_path: str, base_model_name: str = None):
    """モデル固有の設定でトークナイザーを作成する。"""
    model_identifier = base_model_name or model_path
    if "roberta" in model_identifier.lower():
        return AutoTokenizer.from_pretrained(model_path, add_prefix_space=True)
    return AutoTokenizer.from_pretrained(model_path)


def _format_question(question: str, choices: Dict[str, str]) -> str:
    """四択問題を分類器向けの1本のテキストに整形する。"""
    choice_lines = "\n".join(
        f"{label}. {choices[label]}" for label in CHOICE_LABELS if choices.get(label)
    )
    return f"{question}\n{choice_lines}"


def download_jmmlu_zip() -> Path:
    """JMMLU.zip を Hugging Face から直接ダウンロードする。

    `datasets` ライブラリはカスタムローディングスクリプト（JMMLU.py）を
    セキュリティ上の理由でサポートしなくなったため、Presidio データセットと
    同様に生の zip を直接取得し、CSV を手動でパースする。
    """
    zip_path = Path(JMMLU_ZIP_CACHE_PATH)
    if not zip_path.exists():
        logger.info(f"JMMLU.zip をダウンロード中: {JMMLU_ZIP_URL}")
        response = requests.get(JMMLU_ZIP_URL, timeout=60)
        response.raise_for_status()
        zip_path.write_bytes(response.content)
        logger.info(f"ダウンロード完了: {zip_path}")
    else:
        logger.info(f"JMMLU.zip は既に存在: {zip_path}")
    return zip_path


def load_jmmlu_dataset(max_samples: int = 3000) -> Tuple[List[str], List[str]]:
    """JMMLU の全 56 科目を読み込み、14 カテゴリへマッピングする。"""
    logger.info(
        f"データセットを読み込み中: {JMMLU_DATASET_NAME}（全 {len(JMMLU_SUBJECT_TO_CATEGORY)} 科目）"
    )

    zip_path = download_jmmlu_zip()

    category_samples: Dict[str, List[str]] = {}
    loaded_subjects = 0
    failed_subjects: List[str] = []

    with zipfile.ZipFile(zip_path) as archive:
        for subject, category in JMMLU_SUBJECT_TO_CATEGORY.items():
            member_name = f"JMMLU/test/{subject}.csv"
            try:
                with archive.open(member_name) as raw_file:
                    text_stream = io.TextIOWrapper(raw_file, encoding="utf-8-sig")
                    reader = csv.DictReader(text_stream)
                    rows = list(reader)
            except KeyError as e:
                logger.warning(f"科目 {subject} の読み込みに失敗: {e}")
                failed_subjects.append(subject)
                continue

            loaded_subjects += 1
            category_samples.setdefault(category, [])
            for row in rows:
                choices = {label: row.get(label, "") for label in CHOICE_LABELS}
                text = _format_question(row["question"], choices)
                category_samples[category].append(text)

    logger.info(
        f"読み込み成功科目数: {loaded_subjects}/{len(JMMLU_SUBJECT_TO_CATEGORY)}"
    )
    if failed_subjects:
        logger.warning(f"読み込み失敗科目: {failed_subjects}")

    category_raw_counts = {k: len(v) for k, v in category_samples.items()}
    logger.info(f"カテゴリ別サンプル数（フィルタ前）: {category_raw_counts}")

    available_categories = [
        cat for cat in REQUIRED_CATEGORIES if cat in category_samples
    ]
    min_samples_per_category = max(30, max_samples // (len(available_categories) * 2))
    target_samples_per_category = max_samples // len(available_categories)

    filtered_texts: List[str] = []
    filtered_labels: List[str] = []
    category_counts: Dict[str, int] = {}
    insufficient_categories: List[str] = []

    for category in available_categories:
        samples = category_samples.get(category, [])
        if len(samples) < min_samples_per_category:
            insufficient_categories.append(category)
            samples_to_take = len(samples)
        else:
            samples_to_take = min(target_samples_per_category, len(samples))

        filtered_texts.extend(samples[:samples_to_take])
        filtered_labels.extend([category] * samples_to_take)
        category_counts[category] = samples_to_take

    if insufficient_categories:
        logger.warning(f"サンプル不足カテゴリ: {insufficient_categories}")

    logger.info(f"最終カテゴリ分布: {category_counts}")
    logger.info(f"フィルタ後サンプル数: {len(filtered_texts)}")

    if len(category_counts) < len(REQUIRED_CATEGORIES) * 0.8:
        logger.error(
            f"CRITICAL: {len(category_counts)}/{len(REQUIRED_CATEGORIES)} カテゴリのみサンプルあり"
        )

    return filtered_texts, filtered_labels


def prepare_datasets(
    max_samples: int = 3000,
) -> Tuple[List[Dict], Dict[str, int], Dict[int, str]]:
    """JMMLU から学習用データセットを構築する。"""
    texts, labels = load_jmmlu_dataset(max_samples)

    unique_labels = sorted(set(labels))
    ordered_labels = [cat for cat in REQUIRED_CATEGORIES if cat in unique_labels]
    extra_labels = [cat for cat in unique_labels if cat not in REQUIRED_CATEGORIES]
    final_labels = ordered_labels + sorted(extra_labels)

    label2id = {label: idx for idx, label in enumerate(final_labels)}
    id2label = {idx: label for label, idx in label2id.items()}

    logger.info(f"カテゴリ数: {len(final_labels)} — {final_labels}")

    sample_data = [
        {"text": text, "label": label2id[label]} for text, label in zip(texts, labels)
    ]
    return sample_data, label2id, id2label


def create_full_finetune_model(
    model_name: str, num_labels: int, gradient_checkpointing: bool = True
):
    """フルファインチューニング用モデルとトークナイザーを作成する。"""
    logger.info(f"フルファインチューニングモデルを作成: {model_name}")

    tokenizer = create_tokenizer_for_model(model_name, model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        dtype=torch.float32,
        problem_type="single_label_classification",
    )

    if tokenizer.pad_token_id is not None:
        model.config.pad_token_id = tokenizer.pad_token_id

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


def tokenize_data(data: List[Dict], tokenizer, max_length: int = 512) -> Dataset:
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


def _update_config_labels(output_dir: str, label2id: Dict, id2label: Dict):
    """config.json にラベルマッピングを書き込み、Rust/Go 互換性を確保する。"""
    config_path = os.path.join(output_dir, "config.json")
    if not os.path.exists(config_path):
        return

    with open(config_path, "r") as f:
        config = json.load(f)

    config["id2label"] = {str(k): v for k, v in id2label.items()}
    config["label2id"] = label2id

    with open(config_path, "w") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    logger.info("config.json にラベルマッピングを更新")


def main(
    model_name: str = "ruri-v3-30m",
    num_epochs: int = 3,
    batch_size: int = 8,
    learning_rate: float = 2e-5,
    max_samples: int = 3000,
    output_dir: str = None,
    gradient_checkpointing: bool = True,
    gpu_id: int = None,
    max_length: int = 512,
):
    """フルファインチューニングのメイン学習関数。"""
    logger.info("JMMLU 分野分類（日本語）フルファインチューニング学習を開始")

    if gpu_id is not None:
        device_str, selected_gpu = set_gpu_device(gpu_id=gpu_id, auto_select=False)
    else:
        device_str, selected_gpu = set_gpu_device(gpu_id=None, auto_select=True)

    all_gpus = get_all_gpu_info()
    if all_gpus:
        logger.info(f"利用可能 GPU 数: {len(all_gpus)}")
        for gpu in all_gpus:
            status = "SELECTED" if gpu["id"] == selected_gpu else "available"
            logger.info(f"  GPU {gpu['id']} ({status}): {gpu['name']}")

    clear_gpu_memory()
    log_memory_usage("Pre-training")

    model_path = resolve_model_path(model_name)
    logger.info(f"使用モデル: {model_name} -> {model_path}")

    all_data, label2id, id2label = prepare_datasets(max_samples)
    train_data, val_data = train_test_split(all_data, test_size=0.2, random_state=42)

    logger.info(f"学習サンプル数: {len(train_data)}")
    logger.info(f"検証サンプル数: {len(val_data)}")
    logger.info(f"カテゴリ数: {len(label2id)}")

    model, tokenizer = create_full_finetune_model(
        model_path, len(label2id), gradient_checkpointing=gradient_checkpointing
    )

    train_dataset = tokenize_data(train_data, tokenizer, max_length=max_length)
    val_dataset = tokenize_data(val_data, tokenizer, max_length=max_length)

    if output_dir is None:
        output_dir = f"ja_full_domain_classifier_{model_name}"
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"モデル保存先: {output_dir}")

    use_fp16 = torch.cuda.is_available()

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
        report_to="none",
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

    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    label_mapping = {"category_to_idx": label2id, "idx_to_category": id2label}
    with open(os.path.join(output_dir, "label_mapping.json"), "w") as f:
        json.dump(label_mapping, f, indent=2, ensure_ascii=False)
    with open(os.path.join(output_dir, "category_mapping.json"), "w") as f:
        json.dump(label_mapping, f, indent=2, ensure_ascii=False)

    _update_config_labels(output_dir, label2id, id2label)

    logger.info(f"モデルを保存: {output_dir}")

    logger.info("検証セットで最終評価...")
    val_results = trainer.evaluate()
    logger.info("Validation Results:")
    logger.info(f"  Accuracy: {val_results['eval_accuracy']:.4f}")
    logger.info(f"  F1: {val_results['eval_f1']:.4f}")


def demo_inference(model_path: str):
    """学習済みモデルで推論デモ。"""
    logger.info(f"モデルを読み込み中: {model_path}")

    try:
        with open(os.path.join(model_path, "label_mapping.json"), "r") as f:
            mapping_data = json.load(f)
        idx_to_category = {
            int(k): v for k, v in mapping_data["idx_to_category"].items()
        }
        num_labels = len(idx_to_category)

        model = AutoModelForSequenceClassification.from_pretrained(
            model_path, num_labels=num_labels
        )
        tokenizer = AutoTokenizer.from_pretrained(model_path)

        test_examples = [
            "企業合併における最良の戦略は何ですか？",
            "独占禁止法は企業間の競争にどのような影響を与えますか？",
            "消費者行動に影響を与える心理的要因は何ですか？",
            "契約成立の法的要件を説明してください。",
            "民法と刑法の違いは何ですか？",
            "光合成の仕組みを説明してください。",
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
    parser = argparse.ArgumentParser(
        description="JMMLU 分野分類 フルファインチューニング"
    )
    parser.add_argument("--mode", choices=["train", "test"], default="train")
    parser.add_argument(
        "--model",
        choices=["ruri-v3-30m", "modernbert-base", "bert-base-uncased", "roberta-base"],
        default="ruri-v3-30m",
    )
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument(
        "--max-samples", type=int, default=3000, help="JMMLU から取得する最大サンプル数"
    )
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument(
        "--model-path", type=str, default="ja_full_domain_classifier_ruri-v3-30m"
    )
    parser.add_argument("--gpu-id", type=int, default=None)
    parser.add_argument("--no-gradient-checkpointing", action="store_true")

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
            max_length=args.max_length,
        )
    elif args.mode == "test":
        demo_inference(args.model_path)
