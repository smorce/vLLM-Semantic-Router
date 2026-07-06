"""
日本語 意図分類（Intent Classification）— フルファインチューニング

`nappa0326/glaive-function-calling-v2-sharegpt-japanese`（Glaive Function Calling v2 の
日本語 ShareGPT 版）を使用し、ユーザー発話 → 8カテゴリの意図分類器をフルファインチューニングする。
ベースモデルは ModernBERT-Ja 系の `cl-nagoya/ruri-v3-30m` を想定し、`peft` を使わず
全パラメータを学習する。

意図カテゴリ（8種）:
    - information_retrieval: 検索・照会・情報取得系
    - calculation:            計算・数値変換系
    - scheduling:             カレンダー・リマインダー・予約系
    - communication:          メール・通知・SNS投稿系
    - file_operations:        ノート・ファイル・アカウント/連絡先のCRUD系
    - data_transformation:    暗号化・翻訳・生成（QR/パスワード等）系
    - analysis:                感情分析・文章分析・レビュー分析系
    - no_function_needed:     関数呼び出しが不要な一般会話

ラベル付けの方法:
    データセット中の各ユーザー発話 (human) に対して、直後の assistant 応答に
    `<functioncall>` が含まれていればその関数名を抽出し、キーワードベースの
    マッピングテーブルで8カテゴリのいずれかに変換する。関数呼び出しが存在しない
    場合は "no_function_needed" とする。マッピングテーブルに一致しない関数名は
    ラベル品質を保つため学習データから除外する（無理に分類しない）。

Usage:
    # 学習（推奨パラメータ）
    python ja_intent_classifier_full.py --mode train --model ruri-v3-30m --epochs 5 --max-samples 6000

    # 推論テスト
    python ja_intent_classifier_full.py --mode test --model-path ja_full_intent_classifier_ruri-v3-30m

Note:
    本プロジェクトでは学術分野分類（JMMLU, ja_domain_classifier_full.py）は使用せず、
    ルーターの config.yaml `global.model_catalog.system.domain_classifier` /
    `global.model_catalog.modules.classifier.domain` スロットをこの関数呼び出し意図分類器で
    代用する。そのため `category_mapping.json`（category_to_idx/idx_to_category）も出力し、
    Go 側 `classification.CategoryMapping` のスキーマに対応させている。
"""

import argparse
import json
import os
import random
import re
import sys
from collections import Counter
from typing import Dict, List, Optional, Tuple

import torch
from datasets import load_dataset, Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
from transformers import (
    AutoModelForSequenceClassification,
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

GLAIVE_JA_DATASET_NAME = "nappa0326/glaive-function-calling-v2-sharegpt-japanese"
FUNCTIONCALL_NAME_RE = re.compile(r'"name"\s*:\s*"([^"]+)"')

# 関数名 -> 8カテゴリのキーワードベースマッピング。
# 上から順に部分一致を評価し、最初に一致したカテゴリを採用する（先勝ち）。
CATEGORY_KEYWORD_RULES: List[Tuple[str, List[str]]] = [
    (
        "scheduling",
        [
            "calendar",
            "remind",
            "schedule",
            "todo",
            "book_flight",
            "book_hotel",
            "appointment",
            "task",
            "event",
        ],
    ),
    (
        "communication",
        [
            "email",
            "sms",
            "notification",
            "notify",
            "social_media",
            "poll",
            "push_",
            "send_message",
        ],
    ),
    (
        "file_operations",
        [
            "note",
            "upload",
            "survey",
            "create_user",
            "create_account",
            "create_new_user",
            "contact",
            "shopping_cart",
            "shopping_list",
            "playlist",
            "user_profile",
            "create_invoice",
            "execute_program",
        ],
    ),
    (
        "analysis",
        [
            "analyze",
            "sentiment",
            "spelling",
            "word_count",
            "count_words",
            "count_occurrences",
            "review",
        ],
    ),
    (
        "data_transformation",
        [
            "generate_qr",
            "generate_password",
            "generate_barcode",
            "generate_invoice",
            "generate_username",
            "generate_uuid",
            "generate_random_color",
            "generate_random_string",
            "generate_anagram",
            "generate_meme",
            "generate_sudoku",
            "generate_unique_id",
            "generate_report",
            "encrypt",
            "translate",
            "convert_length",
            "reverse_string",
            "validate_credit_card",
            "detect_language",
            "generate_random_name",
            "generate_random_username",
            "password_strength",
            "password_reset",
            "convert_units",
        ],
    ),
    (
        "calculation",
        [
            "calculate",
            "convert_currency",
            "convert_temperature",
            "add_numbers",
            "fibonacci",
            "gcd",
            "square_root",
            "factorial",
            "solve_equation",
            "hypotenuse",
            "circumference",
            "prime",
            "palindrome",
            "anagram",
            "shuffle_array",
            "sort_numbers",
            "common_elements",
            "bmi",
            "sum",
            "median",
            "perimeter",
        ],
    ),
    (
        "information_retrieval",
        [
            "search",
            "get_",
            "find_",
            "check_",
            "track_",
            "recommend",
            "fetch",
            "lookup",
            "order_food",
            "horoscope",
            "generate_random",
        ],
    ),
]

INTENT_CATEGORIES = sorted(
    {rule[0] for rule in CATEGORY_KEYWORD_RULES} | {"no_function_needed"}
)


def map_function_name_to_category(function_name: str) -> Optional[str]:
    """関数名をキーワードベースで8カテゴリのいずれかにマッピングする。"""
    name_lower = function_name.lower()
    for category, keywords in CATEGORY_KEYWORD_RULES:
        for keyword in keywords:
            if keyword in name_lower:
                return category
    return None


def extract_intent_samples(
    dataset, max_no_function_ratio: float = 1.5
) -> Tuple[List[str], List[str]]:
    """
    データセットの各会話から (ユーザー発話, 意図カテゴリ) のペアを抽出する。

    各 human ターンに対し、次の human ターンまでの間に functioncall が
    含まれていればその関数名からカテゴリを決定し、なければ no_function_needed とする。
    """
    texts: List[str] = []
    labels: List[str] = []
    unmapped_counter: Counter = Counter()

    for row in dataset:
        conversations = row.get("conversations", [])
        i = 0
        while i < len(conversations):
            msg = conversations[i]
            if msg.get("from") != "human":
                i += 1
                continue

            user_text = msg.get("value", "").strip()
            j = i + 1
            function_name = None
            while j < len(conversations) and conversations[j].get("from") != "human":
                turn = conversations[j]
                if turn.get("from") == "gpt" and "<functioncall>" in turn.get(
                    "value", ""
                ):
                    match = FUNCTIONCALL_NAME_RE.search(turn["value"])
                    if match:
                        function_name = match.group(1)
                        break
                j += 1

            if user_text:
                if function_name is None:
                    texts.append(user_text)
                    labels.append("no_function_needed")
                else:
                    category = map_function_name_to_category(function_name)
                    if category is not None:
                        texts.append(user_text)
                        labels.append(category)
                    else:
                        unmapped_counter[function_name] += 1

            i = j

    if unmapped_counter:
        top_unmapped = unmapped_counter.most_common(10)
        logger.info(f"未マッピングの関数名（上位10件、除外済み）: {top_unmapped}")
        logger.info(f"未マッピング総数: {sum(unmapped_counter.values())}")

    return texts, labels


def _balance_samples(
    texts: List[str], labels: List[str], max_samples: int
) -> Tuple[List[str], List[str]]:
    """カテゴリごとのサンプル数を均等化する（少数派カテゴリの件数に極力寄せる）。"""
    by_category: Dict[str, List[str]] = {}
    for text, label in zip(texts, labels):
        by_category.setdefault(label, []).append(text)

    counts = {k: len(v) for k, v in by_category.items()}
    logger.info(f"カテゴリ別 raw サンプル数: {counts}")

    num_categories = len(by_category)
    target_per_category = max(1, max_samples // num_categories)

    balanced_texts: List[str] = []
    balanced_labels: List[str] = []
    for category, category_texts in by_category.items():
        random.shuffle(category_texts)
        selected = category_texts[:target_per_category]
        balanced_texts.extend(selected)
        balanced_labels.extend([category] * len(selected))

    combined = list(zip(balanced_texts, balanced_labels))
    random.shuffle(combined)
    balanced_texts, balanced_labels = zip(*combined)
    return list(balanced_texts), list(balanced_labels)


def create_intent_dataset(max_samples: int = 6000, seed: int = 42):
    """glaive-function-calling-v2-sharegpt-japanese から意図分類データセットを構築する。"""
    random.seed(seed)

    logger.info(f"データセットを読み込み中: {GLAIVE_JA_DATASET_NAME}")
    dataset = load_dataset(GLAIVE_JA_DATASET_NAME, split="train")

    texts, labels = extract_intent_samples(dataset)
    logger.info(f"抽出した生サンプル数: {len(texts)}")

    final_counts = Counter(labels)
    logger.info(f"抽出後のカテゴリ分布: {dict(final_counts)}")

    balanced_texts, balanced_labels = _balance_samples(texts, labels, max_samples)

    balanced_counts = Counter(balanced_labels)
    logger.info(f"バランス調整後のカテゴリ分布: {dict(balanced_counts)}")
    logger.info(f"バランス調整後サンプル数: {len(balanced_texts)}")

    label2id = {label: idx for idx, label in enumerate(INTENT_CATEGORIES)}
    id2label = {idx: label for label, idx in label2id.items()}

    sample_data = [
        {"text": text, "label": label2id[label]}
        for text, label in zip(balanced_texts, balanced_labels)
    ]
    return sample_data, label2id, id2label


def create_full_finetune_model(model_name: str, num_labels: int):
    """フルファインチューニング用のシーケンス分類モデルを作成する。"""
    logger.info(f"フルファインチューニングモデルを作成: {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=num_labels, dtype=torch.float32
    )

    if tokenizer.pad_token_id is not None:
        model.config.pad_token_id = tokenizer.pad_token_id

    trainable_params = sum(p.numel() for p in model.parameters())
    for param in model.parameters():
        param.requires_grad = True
    logger.info(f"学習可能パラメータ: {trainable_params:,} (100.0%)")

    return model, tokenizer


def tokenize_data(data: List[Dict], tokenizer, max_length: int = 256) -> Dataset:
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
    precision, recall, _, _ = precision_recall_fscore_support(
        labels, predictions, average="weighted", zero_division=0
    )

    return {"accuracy": accuracy, "f1": f1, "precision": precision, "recall": recall}


def main(
    model_name: str = "ruri-v3-30m",
    num_epochs: int = 5,
    batch_size: int = 16,
    learning_rate: float = 2e-5,
    max_samples: int = 6000,
    output_dir: str = None,
    max_length: int = 256,
    seed: int = 42,
):
    """フルファインチューニングのメイン学習関数。"""
    logger.info("日本語 意図分類 フルファインチューニング学習を開始")

    set_gpu_device(gpu_id=None, auto_select=True)
    clear_gpu_memory()
    log_memory_usage("Pre-training")

    model_path = resolve_model_path(model_name)
    logger.info(f"使用モデル: {model_name} -> {model_path}")

    sample_data, label2id, id2label = create_intent_dataset(max_samples, seed=seed)
    label_ids = [item["label"] for item in sample_data]
    train_data, val_data = train_test_split(
        sample_data, test_size=0.2, random_state=seed, stratify=label_ids
    )
    logger.info(f"学習サンプル数: {len(train_data)}")
    logger.info(f"検証サンプル数: {len(val_data)}")
    logger.info(f"ラベル: {label2id}")

    model, tokenizer = create_full_finetune_model(model_path, len(label2id))

    train_dataset = tokenize_data(train_data, tokenizer, max_length=max_length)
    val_dataset = tokenize_data(val_data, tokenizer, max_length=max_length)

    if output_dir is None:
        output_dir = f"ja_full_intent_classifier_{model_name}"
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
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
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
        compute_metrics=compute_metrics,
    )

    logger.info("学習を開始...")
    trainer.train()

    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    label_mapping = {
        "label_to_id": label2id,
        "id_to_label": {str(k): v for k, v in id2label.items()},
    }
    with open(os.path.join(output_dir, "label_mapping.json"), "w") as f:
        json.dump(label_mapping, f, indent=2, ensure_ascii=False)

    # Go 側 classification.CategoryMapping のスキーマ（category_to_idx/idx_to_category）に
    # 合わせたファイル。config.yaml の classifier.domain スロットをこのモデルで代用する場合、
    # category_mapping_path はこのファイルを指す。
    category_mapping = {
        "category_to_idx": label2id,
        "idx_to_category": {str(k): v for k, v in id2label.items()},
    }
    with open(os.path.join(output_dir, "category_mapping.json"), "w") as f:
        json.dump(category_mapping, f, indent=2, ensure_ascii=False)

    config_path = os.path.join(output_dir, "config.json")
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            config = json.load(f)
        config["id2label"] = {str(k): v for k, v in id2label.items()}
        config["label2id"] = label2id
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        logger.info("config.json にラベルマッピングを更新")

    eval_results = trainer.evaluate()
    logger.info("Validation Results:")
    logger.info(f"  Accuracy: {eval_results['eval_accuracy']:.4f}")
    logger.info(f"  F1: {eval_results['eval_f1']:.4f}")
    logger.info(f"意図分類モデルを保存: {output_dir}")


def demo_inference(model_path: str):
    """学習済みモデルで推論デモ。"""
    logger.info(f"モデルを読み込み中: {model_path}")

    try:
        with open(os.path.join(model_path, "label_mapping.json"), "r") as f:
            mapping_data = json.load(f)
        id_to_label = mapping_data["id_to_label"]
        num_labels = len(id_to_label)

        model = AutoModelForSequenceClassification.from_pretrained(
            model_path, num_labels=num_labels
        )
        tokenizer = AutoTokenizer.from_pretrained(model_path)

        test_examples = [
            "東京の明日の天気を教えてください。",
            "100ドルを日本円に換算するといくらですか？",
            "来週の月曜日にチームミーティングの予定を入れてください。",
            "山田さんにお礼のメールを送ってください。",
            "新しいメモを作成して、買い物リストを保存してください。",
            "パスワードをランダムに生成してください。",
            "このレビューの感情を分析してください。",
            "こんにちは、調子はどうですか？",
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

            predicted_label = id_to_label[str(predicted_class_id)]
            print(f"\nInput: {example}")
            print(f"Prediction: {predicted_label} (confidence: {confidence:.4f})")

    except Exception as e:
        logger.error(f"Error during inference: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="日本語 意図分類 フルファインチューニング"
    )
    parser.add_argument("--mode", choices=["train", "test"], default="train")
    parser.add_argument(
        "--model",
        choices=["ruri-v3-30m", "modernbert-base", "bert-base-uncased", "roberta-base"],
        default="ruri-v3-30m",
    )
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument(
        "--max-samples", type=int, default=6000, help="学習に使用する最大サンプル数"
    )
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument(
        "--model-path", type=str, default="ja_full_intent_classifier_ruri-v3-30m"
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
