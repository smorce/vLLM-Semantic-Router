# 日本語 分類器 フルファインチューニング

ModernBERT-Ja ベースの [`cl-nagoya/ruri-v3-30m`](https://huggingface.co/cl-nagoya/ruri-v3-30m) を
共通ベースモデルとして、複数の分類タスクをそれぞれ独立にフルファインチューニング（`peft` 不使用、全パラメータ学習）する
学習スクリプト群。英語版の `classifier_model_fine_tuning_lora/ft_linear_full.py` を参考にした日本語派生実装。

> **本プロジェクトでの採用状況**: 4モデルすべてをルーターで使用する。
> - **学術分野分類**（JMMLU）→ `domain_classifier` / `classifier.domain` / `routing.signals.domains`
> - **関数呼び出し意図分類**（8カテゴリ）→ `intent_classifier` / `classifier.intent` / `routing.signals.intents`
> - **PII検出** → `pii_classifier` / `classifier.pii`
> - **脱獄検出** → `prompt_guard`

汎用的な GPU 前提の学習・推論手順（他プロジェクトへの流用を想定した最小限のデータ要件を含む）は
[`TRAINING_GUIDE.md`](TRAINING_GUIDE.md) にまとめてある。

## スクリプト一覧

| ファイル | タスク | 分類方式 | 主データセット | 本プロジェクトでの採用 |
|---------|--------|----------|----------------|----------------------|
| `ja_domain_classifier_full.py` | 学術分野分類 | Sequence Classification | [`nlp-waseda/JMMLU`](https://huggingface.co/datasets/nlp-waseda/JMMLU) | 採用（`domain_classifier`） |
| `ja_pii_full.py` | 個人情報(PII)検出 | Token Classification (BIO) | [`akiFQC/japanese-confidential-information-extraction-sft`](https://huggingface.co/datasets/akiFQC/japanese-confidential-information-extraction-sft) | 採用（`pii_classifier`） |
| `ja_jailbreak_full.py` | 脱獄(Jailbreak)検出 | Sequence Classification | [`APTO-001/ja-safety-sft-dataset`](https://huggingface.co/datasets/APTO-001/ja-safety-sft-dataset) + [`kunishou/oasst1-89k-ja`](https://huggingface.co/datasets/kunishou/oasst1-89k-ja) | 採用（`prompt_guard`） |
| `ja_intent_classifier_full.py` | 関数呼び出し意図分類(8カテゴリ) | Sequence Classification | [`nappa0326/glaive-function-calling-v2-sharegpt-japanese`](https://huggingface.co/datasets/nappa0326/glaive-function-calling-v2-sharegpt-japanese) | 採用（`intent_classifier`） |

Go 推論検証スクリプト（各タスクごとに独立したサブディレクトリ / `package main`）:

| ディレクトリ | 対応スクリプト |
|-------------|----------------|
| `verify_domain/` | `ja_domain_classifier_full.py` |
| `verify_pii/` | `ja_pii_full.py` |
| `verify_jailbreak/` | `ja_jailbreak_full.py` |
| `verify_intent/` | `ja_intent_classifier_full.py` |

すべて共通の `go.mod`（module `semantic-router/ja_full_finetuning`）を使用する。

## ベースモデルについて

`cl-nagoya/ruri-v3-30m` は ModernBERT-Ja アーキテクチャの日本語埋め込みモデルで、
`AutoModelForSequenceClassification` / `AutoModelForTokenClassification` にそのまま利用できる。
`common_lora_utils.py` の `get_model_mapping()` / `get_max_length_for_model()` に
`ruri-v3-30m`（8K コンテキスト対応）としてマッピング済み。

学習後の `config.json` の `architectures` は `ModernBertForSequenceClassification` /
`ModernBertForTokenClassification` になるため、Go 側では `InitModernBertClassifier` 系の
API（`use_modernbert: true`, `use_mmbert_32k: false`）で読み込む。

## 各タスクの詳細

### 1. 学術分野分類（`ja_domain_classifier_full.py`）

- **データセット**: `nlp-waseda/JMMLU`（日本語 MMLU、56 科目・約 7,500 問の四択問題）。
  `datasets` ライブラリのデータセットスクリプトが使えなくなったため、`JMMLU.zip` を直接
  ダウンロードして `csv.DictReader` でパースする独自ローダーを実装している。
- **ラベル体系**: 英語版（`ft_linear_full.py`）と同じ 14 の MMLU-Pro カテゴリに揃えてある
  （`biology`, `business`, `chemistry`, `computer science`, `economics`, `engineering`,
  `health`, `history`, `law`, `math`, `philosophy`, `physics`, `psychology`, `other`）。
  `JMMLU_SUBJECT_TO_CATEGORY` で 56 科目 → 14 カテゴリへマッピングする。

```bash
cd src/training/model_classifier/ja_full_finetuning

# 学習（推奨パラメータ）
python ja_domain_classifier_full.py --mode train --model ruri-v3-30m --epochs 3 --max-samples 3000

# 推論テスト
python ja_domain_classifier_full.py --mode test --model-path ja_full_domain_classifier_ruri-v3-30m
```

### 2. 個人情報(PII)検出（`ja_pii_full.py`）

- **データセット**: `akiFQC/japanese-confidential-information-extraction-sft`
  （チャット形式のJSON抽出データ、11カテゴリ: 住所・会社名・メール・氏名・電話番号・
  アカウント識別子・金融情報・システム設定・ネットワーク識別子・プロジェクト情報・取引ID）。
- **ラベル化の方法**: assistant 応答の JSON 抽出値を、正規表現でユーザー本文内から検索し
  文字オフセットを特定した上で BIO タグ（`B-XXX` / `I-XXX` / `O`）に変換する
  （`extract_spans_from_sample` / `tokenize_with_char_offsets`）。抽出値の本文内一致率は
  スモークテストで約 99.9%（4837/4844）。
- **出力ファイル**: `label_mapping.json`（`label_to_id`/`id_to_label`）に加え、
  Go 側 `classification.PIIMapping` のスキーマ（`label_to_idx`/`idx_to_label`）に合わせた
  `pii_type_mapping.json` も出力する。`config.yaml` の `pii_mapping_path` はこちらを参照する。

```bash
cd src/training/model_classifier/ja_full_finetuning

# 学習（推奨パラメータ）
python ja_pii_full.py --mode train --model ruri-v3-30m --epochs 3 --max-samples 3000

# 推論テスト
python ja_pii_full.py --mode test --model-path ja_full_pii_detector_ruri-v3-30m
```

### 3. 脱獄(Jailbreak)検出（`ja_jailbreak_full.py`）

- **データセット**:
  - `APTO-001/ja-safety-sft-dataset`（公開サンプル 500 件、`has_jailbreak` フィールドを
    そのままラベルとして使用）
  - `kunishou/oasst1-89k-ja`（`prompter` 発話を benign 補強データとして使用）
  - 英語版 `jailbreak_bert_finetuning_lora.py` の `SHORT_JAILBREAK_PATTERNS` /
    `LONG_JAILBREAK_PATTERNS` を日本語の同等表現として書き起こしたパターン集
    （`JA_SHORT_JAILBREAK_PATTERNS` / `JA_LONG_JAILBREAK_PATTERNS`）を oversampling して補強する。
    **翻訳APIやLLM呼び出しは一切使用していない**（スクリプト内に直接ハードコードした日本語表現）。
- **注意**: APTO データセットのみでは jailbreak サンプルが少数派かつ総数も少ないため、
  上記の補強を組み合わせてクラスバランスを取っている。スモークテスト（1エポック・300サンプル）
  では検証精度 98.3% だが、パターン oversampling の影響で楽観的な値である可能性が高い。
  実際の学習では `--max-samples` を増やして評価すること。

```bash
cd src/training/model_classifier/ja_full_finetuning

# 学習（推奨パラメータ）
python ja_jailbreak_full.py --mode train --model ruri-v3-30m --epochs 5 --max-samples 4000

# 推論テスト
python ja_jailbreak_full.py --mode test --model-path ja_full_jailbreak_classifier_ruri-v3-30m
```

### 4. 関数呼び出し意図分類（`ja_intent_classifier_full.py`）

- **データセット**: `nappa0326/glaive-function-calling-v2-sharegpt-japanese`
  （Glaive Function Calling v2 の日本語 ShareGPT 版、約 113K 行）。
- **8カテゴリ**: `information_retrieval`, `calculation`, `scheduling`, `communication`,
  `file_operations`, `data_transformation`, `analysis`, `no_function_needed`。
- **ラベル化の方法**: 各ユーザー発話に対し、直後の assistant 応答に含まれる
  `<functioncall>` の関数名をキーワードベースのルール（`CATEGORY_KEYWORD_RULES`）で
  8カテゴリにマッピングする。
- **本番ルーティングとの関係**: `config.yaml` の `global.model_catalog.system.intent_classifier` /
  `global.model_catalog.modules.classifier.intent` スロットおよび
  `routing.signals.intents`（`type: intent`）で使用する。
  Go 互換の `category_mapping.json` を出力する。

```bash
cd src/training/model_classifier/ja_full_finetuning

# 学習（推奨パラメータ）
python ja_intent_classifier_full.py --mode train --model ruri-v3-30m --epochs 5 --max-samples 6000

# 推論テスト
python ja_intent_classifier_full.py --mode test --model-path ja_full_intent_classifier_ruri-v3-30m
```

## Go 検証スクリプトの使い方

```bash
# Rust ライブラリを事前にビルド
make rust   # または cd candle-binding && cargo build --release

export LD_LIBRARY_PATH=$(pwd)/candle-binding/target/release
cd src/training/model_classifier/ja_full_finetuning

CGO_ENABLED=1 go run ./verify_domain    --model ja_full_domain_classifier_ruri-v3-30m    # 不採用スクリプト向け（他プロジェクト用）
CGO_ENABLED=1 go run ./verify_pii       --model ja_full_pii_detector_ruri-v3-30m
CGO_ENABLED=1 go run ./verify_jailbreak --model ja_full_jailbreak_classifier_ruri-v3-30m
CGO_ENABLED=1 go run ./verify_intent    --model ja_full_intent_classifier_ruri-v3-30m     # domain_classifier スロット代用モデル
```

各コマンドはモデルの `config.json` から `architectures` を読み取り
`ModernBertForSequenceClassification` / `ModernBertForTokenClassification` であることを確認した上で、
固定テスト文で推論して正解率サマリを出力する。

## config.yaml への反映

`config/config.yaml` の `global.model_catalog` は Go 側で `decoder.KnownFields(true)`
（厳格スキーマ）でパースされるため（`src/semantic-router/pkg/config/reference_config_contract_test.go`）、
任意のキーを追加することはできない。`model_catalog.system` は
`prompt_guard` / `domain_classifier` / `pii_classifier` / `fact_check_classifier` /
`hallucination_detector` / `hallucination_explainer` / `feedback_detector` の固定スロットのみを持つ
（`src/semantic-router/pkg/config/canonical_global.go`）。

このため、本プロジェクトでは **4モデルすべて** を `config.yaml` に反映している:

| 設定パス | 変更後の値 |
|---------|-----------|
| `global.model_catalog.system.prompt_guard` | `models/ja-full-jailbreak-detector-ruri-v3-30m` |
| `global.model_catalog.system.domain_classifier` | `models/ja-full-domain-classifier-ruri-v3-30m`（学術分野・JMMLU） |
| `global.model_catalog.system.intent_classifier` | `models/ja-full-intent-classifier-ruri-v3-30m`（関数呼び出し意図・8カテゴリ） |
| `global.model_catalog.system.pii_classifier` | `models/ja-full-pii-detector-ruri-v3-30m` |
| `global.model_catalog.modules.classifier.domain.*` | 上記 domain モデル + `use_modernbert: true` |
| `global.model_catalog.modules.classifier.intent.*` | 上記 intent モデル + `use_modernbert: true` |
| `routing.signals.intents` | 8カテゴリ定義（`type: intent` でルーティング可能） |

| タスク | `config.yaml` が参照するパス |
|--------|------------------------------|
| 学術分野分類 | `models/ja-full-domain-classifier-ruri-v3-30m` |
| 関数呼び出し意図分類 | `models/ja-full-intent-classifier-ruri-v3-30m` |
| PII検出 | `models/ja-full-pii-detector-ruri-v3-30m` |
| 脱獄検出 | `models/ja-full-jailbreak-detector-ruri-v3-30m` |

```bash
# 例: 意図分類をフル学習してそのまま config.yaml のパスへ出力する場合
python ja_intent_classifier_full.py \
  --mode train --model ruri-v3-30m --epochs 5 --max-samples 6000 \
  --output-dir ../../../../models/ja-full-intent-classifier-ruri-v3-30m
```

### 既知の制約・未対応事項

- **本段階ではスモークテストのみ**: 各スクリプトは `--max-samples` を数百に絞った
  1エポックの学習で疎通確認済みだが、実際のフル学習（数千〜数万サンプル・複数エポック）は
  未実施。`config.yaml` 上書きは、学習完了後にモデルを配置する前提で先行反映している。
- **意図分類のルーティング例**: `routing.decisions` で `type: intent` / `name: calculation` のように
  意図シグナルを使う場合は、`routing.signals.intents` に同名のルールが定義されている必要がある
  （`config.yaml` に8カテゴリを追加済み）。

## 依存パッケージ

`torch`, `transformers`, `accelerate`, `datasets`, `scikit-learn`, `requests` が必要。
`uv` を使う場合:

```bash
uv pip install torch transformers accelerate datasets scikit-learn requests
```

## 参考

- 英語版フルFT実装: [`../classifier_model_fine_tuning_lora/ft_linear_full.py`](../classifier_model_fine_tuning_lora/README.md)
- ベースモデル: [`cl-nagoya/ruri-v3-30m`](https://huggingface.co/cl-nagoya/ruri-v3-30m)
- 親ディレクトリ README: [`../README.md`](../README.md)
