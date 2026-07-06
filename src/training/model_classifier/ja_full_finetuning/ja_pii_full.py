"""
akiFQC PII（社外秘情報）検出 — 日本語トークン分類 フルファインチューニング

`pii_model_fine_tuning_lora/pii_bert_finetuning_lora.py`（Presidio + AI4Privacy, 英語）の
日本語派生版。ベースモデルは ModernBERT-Ja 系の `cl-nagoya/ruri-v3-30m` を想定し、
`peft` を使わず全パラメータを学習する。

Usage:
    # 学習（推奨パラメータ）
    python ja_pii_full.py --mode train --model ruri-v3-30m --epochs 5 --max-samples 8000

    # 推論テスト
    python ja_pii_full.py --mode test --model-path ja_full_pii_detector_ruri-v3-30m

Dataset:
    - akiFQC/japanese-confidential-information-extraction-sft
      チャット形式（system: 抽出指示 / user: 本文 / assistant: 11カテゴリの JSON 抽出結果）。
      文字オフセット（span）は含まれないため、抽出値を本文中から検索して
      BIO タグへ変換する（`extract_spans_from_sample` 参照）。

Entity categories (11):
    ADDRESS, COMPANY_NAME, EMAIL_ADDRESS, HUMAN_NAME, PHONE_NUMBER,
    ACCOUNT_IDENTIFIER, NETWORK_IDENTIFIER, SYSTEM_CONFIG, PROJECT_INFO,
    FINANCIAL_INFO, TRANSACTION_ID

Note:
    日本語には単語区切り（スペース）が無いため、既存の LoRA 版で mmbert-32k 向けに
    実装されていた「文字オフセットベースの BIO 整列」（tokenize_with_char_offsets 相当）を
    全サンプルに対して採用する。
"""

import argparse
import json
import os
import random
import re
import sys
from collections import defaultdict
from typing import Dict, List, Tuple

import torch
from datasets import Dataset, load_dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common_lora_utils import (
    clear_gpu_memory,
    log_memory_usage,
    resolve_model_path,
    set_gpu_device,
    setup_logging,
)

logger = setup_logging()

DATASET_NAME = "akiFQC/japanese-confidential-information-extraction-sft"

# akiFQC の 11 カテゴリキー（小文字）→ BIO エンティティ名（大文字）
CATEGORY_KEY_TO_ENTITY = {
    "address": "ADDRESS",
    "company_name": "COMPANY_NAME",
    "email_address": "EMAIL_ADDRESS",
    "human_name": "HUMAN_NAME",
    "phone_number": "PHONE_NUMBER",
    "account_identifier": "ACCOUNT_IDENTIFIER",
    "network_identifier": "NETWORK_IDENTIFIER",
    "system_config": "SYSTEM_CONFIG",
    "project_info": "PROJECT_INFO",
    "financial_info": "FINANCIAL_INFO",
    "transaction_id": "TRANSACTION_ID",
}

ENTITY_TYPES = sorted(set(CATEGORY_KEY_TO_ENTITY.values()))


def create_tokenizer_for_model(model_path: str, base_model_name: str = None):
    """モデル固有の設定でトークナイザーを作成する。"""
    model_identifier = base_model_name or model_path
    if "roberta" in model_identifier.lower():
        return AutoTokenizer.from_pretrained(model_path, add_prefix_space=True)
    return AutoTokenizer.from_pretrained(model_path)


def _extract_assistant_json(assistant_content: str) -> Dict[str, List[str]]:
    """assistant メッセージの JSON 文字列をパースする。壊れている場合は空辞書を返す。"""
    try:
        parsed = json.loads(assistant_content)
        if isinstance(parsed, dict):
            return parsed
    except (json.JSONDecodeError, TypeError):
        pass
    return {}


def extract_spans_from_sample(user_text: str, assistant_content: str) -> List[Dict]:
    """assistant が抽出した値を本文中から検索し、文字オフセット付き span に変換する。

    akiFQC データセットには span の位置情報が含まれないため、抽出された文字列を
    `user_text` 内で完全一致検索することで位置を復元する（複数回出現する場合は全て採用）。
    """
    extracted = _extract_assistant_json(assistant_content)
    spans: List[Dict] = []

    for category_key, values in extracted.items():
        entity_type = CATEGORY_KEY_TO_ENTITY.get(category_key)
        if entity_type is None or not isinstance(values, list):
            continue

        for value in values:
            if not value or not isinstance(value, str):
                continue
            for match in re.finditer(re.escape(value), user_text):
                spans.append(
                    {
                        "entity_type": entity_type,
                        "start_position": match.start(),
                        "end_position": match.end(),
                        "entity_value": value,
                    }
                )

    return spans


def load_akifqc_raw_samples(max_raw_samples: int) -> List[Dict]:
    """akiFQC データセットを読み込み、(text, spans) のリストへ変換する。"""
    logger.info(f"データセットを読み込み中: {DATASET_NAME}")
    dataset = load_dataset(DATASET_NAME, split="train")

    raw_samples: List[Dict] = []
    unmatched_values = 0
    total_values = 0

    for row in dataset:
        messages = row.get("messages", [])
        user_text = None
        assistant_content = None
        for message in messages:
            if message.get("role") == "user":
                user_text = message.get("content")
            elif message.get("role") == "assistant":
                assistant_content = message.get("content")

        if not user_text or not assistant_content:
            continue

        extracted = _extract_assistant_json(assistant_content)
        for values in extracted.values():
            if isinstance(values, list):
                total_values += len(values)

        spans = extract_spans_from_sample(user_text, assistant_content)
        matched_values = len({(s["entity_type"], s["entity_value"]) for s in spans})
        unmatched_values += max(
            0,
            sum(len(v) for v in extracted.values() if isinstance(v, list))
            - matched_values,
        )

        raw_samples.append({"full_text": user_text, "spans": spans})

        if len(raw_samples) >= max_raw_samples:
            break

    if total_values > 0:
        logger.info(
            f"抽出値の本文内一致率（概算）: {(total_values - unmatched_values)}/{total_values}"
        )
    logger.info(f"読み込んだ生サンプル数: {len(raw_samples)}")
    return raw_samples


def balance_pii_samples(raw_samples: List[Dict], max_samples: int) -> List[Dict]:
    """エンティティタイプ別にバランスを取りつつ、無エンティティサンプルも一部残す。"""
    entity_samples: Dict[str, List[Dict]] = defaultdict(list)
    samples_without_entities: List[Dict] = []

    for sample in raw_samples:
        if not sample["spans"]:
            samples_without_entities.append(sample)
            continue
        sample_types = {span["entity_type"] for span in sample["spans"]}
        for entity_type in sample_types:
            entity_samples[entity_type].append(sample)

    if not entity_samples:
        logger.warning(
            "エンティティを含むサンプルが見つからないため、先頭から単純に切り出す"
        )
        return raw_samples[:max_samples]

    samples_per_type = max_samples // (len(entity_samples) + 1)
    balanced: List[Dict] = []
    seen_ids = set()

    for entity_type, samples in entity_samples.items():
        random.shuffle(samples)
        added = 0
        for sample in samples:
            if added >= samples_per_type:
                break
            sample_id = id(sample)
            if sample_id in seen_ids:
                continue
            seen_ids.add(sample_id)
            balanced.append(sample)
            added += 1
        logger.info(f"エンティティタイプ {entity_type}: {added} サンプルを選択")

    remaining = max_samples - len(balanced)
    if remaining > 0 and samples_without_entities:
        random.shuffle(samples_without_entities)
        negative_samples = samples_without_entities[:remaining]
        balanced.extend(negative_samples)
        logger.info(f"無エンティティサンプルを {len(negative_samples)} 件追加（負例）")

    random.shuffle(balanced)
    logger.info(f"バランス調整後サンプル数: {len(balanced)}")
    return balanced


def build_label_mapping() -> Tuple[Dict[str, int], Dict[int, str]]:
    """BIO ラベルの label2id / id2label を構築する。"""
    all_labels = ["O"]
    for entity_type in ENTITY_TYPES:
        all_labels.extend([f"B-{entity_type}", f"I-{entity_type}"])

    label_to_id = {label: idx for idx, label in enumerate(all_labels)}
    id_to_label = {idx: label for label, idx in label_to_id.items()}
    return label_to_id, id_to_label


def tokenize_with_char_offsets(
    text: str,
    spans: List[Dict],
    tokenizer,
    label_to_id: Dict[str, int],
    max_length: int = 384,
) -> Dict:
    """文字オフセットを用いてトークンへ BIO ラベルを整列する（日本語は単語区切りが無いため必須）。"""
    encoding = tokenizer(
        text,
        truncation=True,
        max_length=max_length,
        padding="max_length",
        return_offsets_mapping=True,
    )

    labels = []
    for start, end in encoding["offset_mapping"]:
        if start == 0 and end == 0:
            labels.append(-100)
        else:
            labels.append(label_to_id.get("O", 0))

    sorted_spans = sorted(spans, key=lambda x: (x["start_position"], x["end_position"]))

    for span in sorted_spans:
        entity_type = span["entity_type"]
        span_start = span["start_position"]
        span_end = span["end_position"]

        b_label = f"B-{entity_type}"
        i_label = f"I-{entity_type}"
        if b_label not in label_to_id:
            continue

        first_token = True
        for i, (tok_start, tok_end) in enumerate(encoding["offset_mapping"]):
            if tok_start == 0 and tok_end == 0:
                continue
            if tok_start < span_end and tok_end > span_start:
                # 既に別エンティティでラベル済みのトークンは上書きしない（先勝ち）
                if labels[i] != label_to_id.get("O", 0):
                    continue
                if first_token:
                    labels[i] = label_to_id[b_label]
                    first_token = False
                else:
                    labels[i] = label_to_id.get(i_label, label_to_id[b_label])

    return {
        "input_ids": encoding["input_ids"],
        "attention_mask": encoding["attention_mask"],
        "labels": labels,
    }


def prepare_token_dataset(
    raw_samples: List[Dict],
    tokenizer,
    label_to_id: Dict[str, int],
    max_length: int = 384,
) -> Dataset:
    """トークン分類用データセットを構築する。"""
    all_input_ids, all_attention_masks, all_labels = [], [], []

    for sample in raw_samples:
        result = tokenize_with_char_offsets(
            sample["full_text"], sample["spans"], tokenizer, label_to_id, max_length
        )
        all_input_ids.append(result["input_ids"])
        all_attention_masks.append(result["attention_mask"])
        all_labels.append(result["labels"])

    return Dataset.from_dict(
        {
            "input_ids": all_input_ids,
            "attention_mask": all_attention_masks,
            "labels": all_labels,
        }
    )


def create_full_finetune_token_model(model_name: str, num_labels: int):
    """フルファインチューニング用のトークン分類モデルを作成する。"""
    logger.info(f"フルファインチューニング（トークン分類）モデルを作成: {model_name}")

    tokenizer = create_tokenizer_for_model(model_name, model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForTokenClassification.from_pretrained(
        model_name, num_labels=num_labels, dtype=torch.float32
    )

    if tokenizer.pad_token_id is not None:
        model.config.pad_token_id = tokenizer.pad_token_id

    trainable_params = sum(p.numel() for p in model.parameters())
    for param in model.parameters():
        param.requires_grad = True

    logger.info(f"学習可能パラメータ: {trainable_params:,} (100.0%)")

    return model, tokenizer


def compute_token_metrics(eval_pred):
    """トークン分類の評価メトリクスを計算する。"""
    predictions, labels = eval_pred
    predictions = torch.argmax(torch.tensor(predictions), dim=2)

    true_predictions = [
        [p for p, lbl in zip(prediction, label) if lbl != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [lbl for p, lbl in zip(prediction, label) if lbl != -100]
        for prediction, label in zip(predictions, labels)
    ]

    flat_predictions = [item for sublist in true_predictions for item in sublist]
    flat_labels = [item for sublist in true_labels for item in sublist]

    accuracy = accuracy_score(flat_labels, flat_predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        flat_labels, flat_predictions, average="weighted", zero_division=0
    )

    return {"accuracy": accuracy, "f1": f1, "precision": precision, "recall": recall}


def main(
    model_name: str = "ruri-v3-30m",
    num_epochs: int = 5,
    batch_size: int = 8,
    learning_rate: float = 2e-5,
    max_samples: int = 8000,
    output_dir: str = None,
    max_length: int = 384,
    seed: int = 42,
):
    """フルファインチューニングのメイン学習関数。"""
    logger.info("akiFQC PII 検出（日本語）フルファインチューニング学習を開始")
    random.seed(seed)

    set_gpu_device(gpu_id=None, auto_select=True)
    clear_gpu_memory()
    log_memory_usage("Pre-training")

    model_path = resolve_model_path(model_name)
    logger.info(f"使用モデル: {model_name} -> {model_path}")

    raw_samples = load_akifqc_raw_samples(max_raw_samples=max_samples * 3)
    balanced_samples = balance_pii_samples(raw_samples, max_samples)

    label_to_id, id_to_label = build_label_mapping()
    logger.info(f"ラベル数: {len(label_to_id)}")

    train_size = int(0.85 * len(balanced_samples))
    train_samples = balanced_samples[:train_size]
    val_samples = balanced_samples[train_size:]
    logger.info(f"学習サンプル数: {len(train_samples)}")
    logger.info(f"検証サンプル数: {len(val_samples)}")

    model, tokenizer = create_full_finetune_token_model(model_path, len(label_to_id))

    train_dataset = prepare_token_dataset(
        train_samples, tokenizer, label_to_id, max_length
    )
    val_dataset = prepare_token_dataset(val_samples, tokenizer, label_to_id, max_length)

    if output_dir is None:
        output_dir = f"ja_full_pii_detector_{model_name}"
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"モデル保存先: {output_dir}")

    use_fp16 = torch.cuda.is_available()

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=0.05,
        warmup_ratio=0.06,
        lr_scheduler_type="cosine",
        max_grad_norm=1.0,
        logging_dir=f"{output_dir}/logs",
        logging_steps=20,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        save_total_limit=2,
        dataloader_drop_last=False,
        eval_accumulation_steps=1,
        report_to="none",
        fp16=use_fp16,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_token_metrics,
    )

    logger.info("学習を開始...")
    trainer.train()

    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    with open(os.path.join(output_dir, "label_mapping.json"), "w") as f:
        json.dump(
            {
                "label_to_id": label_to_id,
                "id_to_label": {str(k): v for k, v in id_to_label.items()},
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    # Go 側 classification.PIIMapping のスキーマ（label_to_idx/idx_to_label）に合わせたファイル。
    # config.yaml の pii_mapping_path はこのファイルを指す。
    with open(os.path.join(output_dir, "pii_type_mapping.json"), "w") as f:
        json.dump(
            {
                "label_to_idx": label_to_id,
                "idx_to_label": {str(k): v for k, v in id_to_label.items()},
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    config_path = os.path.join(output_dir, "config.json")
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            config = json.load(f)
        config["id2label"] = {str(k): v for k, v in id_to_label.items()}
        config["label2id"] = label_to_id
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        logger.info("config.json にラベルマッピングを更新")

    eval_results = trainer.evaluate()
    logger.info("Validation Results:")
    logger.info(f"  Accuracy: {eval_results['eval_accuracy']:.4f}")
    logger.info(f"  F1: {eval_results['eval_f1']:.4f}")
    logger.info(f"  Precision: {eval_results['eval_precision']:.4f}")
    logger.info(f"  Recall: {eval_results['eval_recall']:.4f}")
    logger.info(f"PII 検出モデルを保存: {output_dir}")


def demo_inference(model_path: str):
    """学習済みモデルで推論デモ。"""
    logger.info(f"モデルを読み込み中: {model_path}")

    try:
        with open(os.path.join(model_path, "label_mapping.json"), "r") as f:
            mapping_data = json.load(f)
        id_to_label = {int(k): v for k, v in mapping_data["id_to_label"].items()}
        num_labels = len(id_to_label)

        model = AutoModelForTokenClassification.from_pretrained(
            model_path, num_labels=num_labels
        )
        tokenizer = AutoTokenizer.from_pretrained(model_path)

        test_examples = [
            "山田太郎（yamada@example.co.jp）から請求書番号 INV-2024-0042 で見積書が届いた。",
            "東京都千代田区丸の内1-1-1にお住まいの佐藤花子様、電話番号は03-1234-5678です。",
            "これは個人情報を含まない普通の文章です。",
        ]

        logger.info("推論を実行...")
        for example in test_examples:
            encoding = tokenizer(
                example,
                return_tensors="pt",
                truncation=True,
                padding=True,
                return_offsets_mapping=True,
            )
            offset_mapping = encoding.pop("offset_mapping")[0].tolist()

            with torch.no_grad():
                outputs = model(**encoding)
                predictions = torch.argmax(outputs.logits, dim=2)[0].tolist()

            print(f"\nInput: {example}")
            found = False
            for (start, end), label_id in zip(offset_mapping, predictions):
                if start == 0 and end == 0:
                    continue
                label = id_to_label.get(label_id, "O")
                if label != "O":
                    print(f"  {example[start:end]!r}: {label}")
                    found = True
            if not found:
                print("  No PII detected")
            print("-" * 50)

    except Exception as e:
        logger.error(f"Error during inference: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="akiFQC PII 検出 フルファインチューニング"
    )
    parser.add_argument("--mode", choices=["train", "test"], default="train")
    parser.add_argument(
        "--model",
        choices=["ruri-v3-30m", "modernbert-base", "bert-base-uncased", "roberta-base"],
        default="ruri-v3-30m",
    )
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument(
        "--max-samples",
        type=int,
        default=8000,
        help="akiFQC から取得する最大サンプル数",
    )
    parser.add_argument("--max-length", type=int, default=384)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument(
        "--model-path", type=str, default="ja_full_pii_detector_ruri-v3-30m"
    )
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    if args.mode == "train":
        main(
            model_name=args.model,
            num_epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            max_samples=args.max_samples,
            output_dir=args.output_dir,
            max_length=args.max_length,
            seed=args.seed,
        )
    elif args.mode == "test":
        demo_inference(args.model_path)
