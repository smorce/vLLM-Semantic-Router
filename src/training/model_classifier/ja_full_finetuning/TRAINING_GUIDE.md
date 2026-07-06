# 日本語分類モデル フルファインチューニング GPU学習ガイド

`cl-nagoya/ruri-v3-30m`（ModernBERT-Ja）をベースにした4つの日本語分類タスク
（学術分野分類・PII検出・脱獄検出・関数呼び出し意図分類）について、GPU環境での
フルファインチューニング手順をまとめた汎用ガイド。本ドキュメントは他プロジェクトへ
モデルを移植・再利用する際に必要な最低限の情報（データ要件・学習方法・推論方法）を
中心にまとめてあり、本リポジトリでの `config.yaml` 統合・採用状況は
[README.md](README.md) を参照すること。

本リポジトリでは **4モデルすべて** をルーターで使用する:

| タスク | スクリプト | `config.yaml` スロット |
|--------|-----------|------------------------|
| 学術分野分類（JMMLU, 14カテゴリ） | `ja_domain_classifier_full.py` | `domain_classifier` / `routing.signals.domains` |
| PII検出（BIO トークン分類） | `ja_pii_full.py` | `pii_classifier` |
| 脱獄検出（二値分類） | `ja_jailbreak_full.py` | `prompt_guard` |
| 関数呼び出し意図分類（8カテゴリ） | `ja_intent_classifier_full.py` | `intent_classifier` / `routing.signals.intents` |

## 前提条件

- **GPU**: NVIDIA GPU 1枚（VRAM 8GB 以上を推奨。`ruri-v3-30m` は 30M パラメータと
  小型なため、フルファインチューニングでも要求 VRAM は小さい）。
- **CUDA 対応 PyTorch** がインストール済みであること。
- Python 3.10〜3.11。

```bash
# GPU が認識されているか確認
nvidia-smi
python3 -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

### 依存パッケージのインストール（`uv` 使用）

```bash
cd src/training/model_classifier/ja_full_finetuning

uv venv --python 3.11
source .venv/bin/activate

# CUDA 版 PyTorch + 学習に必要なライブラリ
uv pip install torch --extra-index-url https://download.pytorch.org/whl/cu121
uv pip install transformers accelerate datasets scikit-learn requests
```

`--extra-index-url` を使うこと（`--index-url` にすると PyTorch 以外のパッケージが
見つからなくなる）。

## 共通ワークフロー

どのタスクも次の3ステップで完結する。

1. **学習**: `python <script>.py --mode train --model ruri-v3-30m ...`
   データセットの自動ダウンロード → 前処理 → `Trainer` によるフルファインチューニング →
   `<output-dir>/` にモデル一式（`model.safetensors`, `config.json`, `tokenizer*`,
   ラベルマッピング JSON）を保存。
2. **Python 推論テスト**: `python <script>.py --mode test --model-path <output-dir>`
   固定サンプル文で推論し、予測ラベルと確信度を表示。
3. **Go 推論検証**（`candle-binding` 経由、他言語サービスへの組み込み確認用）:
   `verify_<task>/main.go` を参照（後述）。

学習は GPU 1枚を自動選択する（`common_lora_utils.set_gpu_device(auto_select=True)`）。
複数 GPU 環境で明示的に指定したい場合は `ja_domain_classifier_full.py` のみ
`--gpu-id <N>` に対応している（他スクリプトは自動選択のみ）。

本リポジトリへそのまま配置する場合は、`--output-dir` を `config.yaml` が参照する
`models/ja-full-*` パスに向ける（例は [README.md](README.md#configyaml-への反映) 参照）。

### 出力物の共通フォーマット

学習後の `<output-dir>/` には概ね以下が生成される。

| ファイル | 内容 |
|---------|------|
| `model.safetensors` / `config.json` | Hugging Face 標準のモデル重み・設定（`architectures` は `ModernBertFor*`） |
| `tokenizer.json` 等 | トークナイザー一式 |
| `label_mapping.json` | Python 推論・デバッグ用のラベルマッピング |
| Go 互換マッピング（タスク別） | 下表を参照 |

| タスク | Go 互換マッピングファイル | `config.yaml` 参照キー例 |
|--------|--------------------------|-------------------------|
| 学術分野分類 | `category_mapping.json` | `category_mapping_path`（domain） |
| 関数呼び出し意図分類 | `category_mapping.json` | `category_mapping_path`（intent） |
| PII検出 | `pii_type_mapping.json` | `pii_mapping_path` |
| 脱獄検出 | `jailbreak_type_mapping.json` | `jailbreak_mapping_path` |

他プロジェクトで Python のみ使う場合は `label_mapping.json` だけ見れば十分。
Go/Rust 経由で推論する場合は上記の Go 互換マッピングファイルを使うこと。
Go 側では `use_modernbert: true`, `use_mmbert_32k: false` で読み込む。

## 1. PII（個人情報）検出 — `ja_pii_full.py`

トークン分類（BIO タグ）で、文中の個人情報・機密情報スパンを検出する。

### 最低限必要なデータ

- **形式**: 「本文（自然文）」＋「本文中に含まれる機密情報エンティティのリスト
  （カテゴリ名 + 実際の文字列値）」。文字オフセット（span）は不要 — スクリプト側で
  本文中を検索して自動的に文字オフセットへ変換する（`extract_spans_from_sample`）。
- **最低件数の目安**: エンティティカテゴリ数 × 200〜300 件程度（本実装は11カテゴリ、
  合計 3,000〜8,000 サンプル程度を推奨）。カテゴリごとの偏りが大きいと精度が落ちるため、
  `balance_pii_samples` のようなアンダーサンプリング/オーバーサンプリングでカテゴリ間の
  件数を揃えることを推奨。
- **他プロジェクトで自前データを使う場合**: `akiFQC/...` の代わりに
  `load_akifqc_raw_samples()` 相当の関数を差し替え、各サンプルを
  `{"text": str, "entities": {"category_key": ["値1", "値2", ...], ...}}` の形へ整形して
  `extract_spans_from_sample` に渡せばよい。

### 学習方法

```bash
python ja_pii_full.py \
  --mode train --model ruri-v3-30m \
  --epochs 5 --batch-size 8 --learning-rate 2e-5 \
  --max-samples 8000 --max-length 384 \
  --output-dir ja_full_pii_detector_ruri-v3-30m
```

主な引数（デフォルト値）: `--epochs 5`, `--batch-size 8`, `--learning-rate 2e-5`,
`--max-samples 8000`, `--max-length 384`。

### 推論方法

Python:

```bash
python ja_pii_full.py --mode test --model-path ja_full_pii_detector_ruri-v3-30m
```

```python
from transformers import AutoModelForTokenClassification, AutoTokenizer
import json, torch

model_path = "ja_full_pii_detector_ruri-v3-30m"
with open(f"{model_path}/label_mapping.json") as f:
    id_to_label = json.load(f)["id_to_label"]

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForTokenClassification.from_pretrained(model_path)

text = "田中太郎さんの電話番号は090-1234-5678です。"
inputs = tokenizer(text, return_tensors="pt", return_offsets_mapping=True, truncation=True)
offsets = inputs.pop("offset_mapping")[0]
with torch.no_grad():
    logits = model(**inputs).logits
pred_ids = logits.argmax(dim=-1)[0].tolist()
labels = [id_to_label[str(i)] for i in pred_ids]
# BIO タグと offsets を突き合わせてエンティティスパンへ復元する
```

Go（`candle-binding` 経由、BIO エンティティ結合ロジック込み）:

```bash
export LD_LIBRARY_PATH=$(pwd)/../../../../candle-binding/target/release
CGO_ENABLED=1 go run ./verify_pii --model ja_full_pii_detector_ruri-v3-30m
```

## 2. 脱獄（Jailbreak）検出 — `ja_jailbreak_full.py`

二値分類（`benign` / `jailbreak`）。

### 最低限必要なデータ

- **形式**: 「テキスト」＋「`benign`/`jailbreak` の二値ラベル」の単純な分類データ。
- **最低件数の目安**: 各クラス 500 件以上（合計 1,000 件以上）。ただし脱獄サンプルは
  実データが希少なため、本実装のように少数の実データ（`APTO-001`, 500件）を
  パターン集（`JA_SHORT_JAILBREAK_PATTERNS` / `JA_LONG_JAILBREAK_PATTERNS`）の
  オーバーサンプリングで補強するアプローチが現実的（翻訳 API や LLM は不使用）。
  クラス不均衡が大きいと再現率が落ちるため、`_balance_samples` のように
  多数派クラスをダウンサンプリングしてバランスを取ること。
- **他プロジェクトで自前データを使う場合**: `create_jailbreak_dataset()` の代わりに
  `(text: str, label: "benign"|"jailbreak")` のペア列を用意し、同じ
  `train_test_split` 以降のパイプラインに渡せばよい。

### 学習方法

```bash
python ja_jailbreak_full.py \
  --mode train --model ruri-v3-30m \
  --epochs 5 --batch-size 8 --learning-rate 2e-5 \
  --max-samples 4000 --max-length 512 \
  --output-dir ja_full_jailbreak_classifier_ruri-v3-30m
```

### 推論方法

Python:

```bash
python ja_jailbreak_full.py --mode test --model-path ja_full_jailbreak_classifier_ruri-v3-30m
```

Go:

```bash
export LD_LIBRARY_PATH=$(pwd)/../../../../candle-binding/target/release
CGO_ENABLED=1 go run ./verify_jailbreak --model ja_full_jailbreak_classifier_ruri-v3-30m
```

## 3. 関数呼び出し意図分類 — `ja_intent_classifier_full.py`

8カテゴリのマルチクラス分類（`information_retrieval`, `calculation`, `scheduling`,
`communication`, `file_operations`, `data_transformation`, `analysis`,
`no_function_needed`）。

### 最低限必要なデータ

- **形式**: 「ユーザー発話（1文〜数文）」＋「意図カテゴリラベル」。
  関数呼び出しログ（`<functioncall>{"name": "...", ...}</functioncall>` 形式）が
  ある場合は、関数名をキーワードルールでカテゴリへマッピングすることで
  ラベル付けコストを大幅に削減できる（`CATEGORY_KEYWORD_RULES` 参照）。
- **最低件数の目安**: カテゴリあたり 300〜500 件、合計 3,000〜6,000 件程度
  （本実装は `_balance_samples` でカテゴリごとに均等化）。
- **他プロジェクトでカテゴリ体系を変える場合**: `CATEGORY_KEYWORD_RULES` の
  キー（関数名の部分文字列）→カテゴリ名の対応表を書き換えるだけで、
  学習パイプライン自体は変更不要。関数呼び出しログがない場合は、
  ユーザー発話に直接ラベルを付与したデータでも同じ形式
  （`(text: str, label: str)`）で学習可能。

### 学習方法

```bash
python ja_intent_classifier_full.py \
  --mode train --model ruri-v3-30m \
  --epochs 5 --batch-size 16 --learning-rate 2e-5 \
  --max-samples 6000 --max-length 256 \
  --output-dir ja_full_intent_classifier_ruri-v3-30m
```

### 推論方法

Python:

```bash
python ja_intent_classifier_full.py --mode test --model-path ja_full_intent_classifier_ruri-v3-30m
```

Go:

```bash
export LD_LIBRARY_PATH=$(pwd)/../../../../candle-binding/target/release
CGO_ENABLED=1 go run ./verify_intent --model ja_full_intent_classifier_ruri-v3-30m
```

本リポジトリでは `global.model_catalog.system.intent_classifier` /
`global.model_catalog.modules.classifier.intent` スロットおよび
`routing.signals.intents`（`type: intent`）で使用する。
Go 互換の `category_mapping.json`（`category_to_idx` / `idx_to_category`）も出力するため、
`config.yaml` の `category_mapping_path` にそのまま指定できる
（統合手順は [README.md](README.md#configyaml-への反映) 参照）。

## 4. 学術分野分類 — `ja_domain_classifier_full.py`

14カテゴリ（`biology`, `business`, `chemistry`, `computer science`, `economics`,
`engineering`, `health`, `history`, `law`, `math`, `philosophy`, `physics`,
`psychology`, `other`）のマルチクラス分類。本リポジトリでは
`domain_classifier` / `routing.signals.domains` に接続する。

### 最低限必要なデータ

- **形式**: 「テキスト（質問文や文書）」＋「分野カテゴリラベル」。
- **データソース**: `nlp-waseda/JMMLU`（56 科目・約 7,500 問）。
  `datasets` のスクリプトが使えないため、`JMMLU.zip` を直接ダウンロードして
  `csv.DictReader` でパースする独自ローダーを実装している。
  `JMMLU_SUBJECT_TO_CATEGORY` で 56 科目 → 14 カテゴリへマッピングする。
- **最低件数の目安**: カテゴリあたり 100〜300 件、合計 1,500〜3,000 件程度。
- **他プロジェクトで自前データを使う場合**: `load_jmmlu_dataset()` を、
  自前データを `(text: str, category: str)` のリストとして返す関数に差し替えれば、
  以降の前処理・学習パイプラインはそのまま使える。

### 学習方法

```bash
python ja_domain_classifier_full.py \
  --mode train --model ruri-v3-30m \
  --epochs 3 --batch-size 8 --learning-rate 2e-5 \
  --max-samples 3000 --max-length 512 \
  --output-dir ja_full_domain_classifier_ruri-v3-30m
```

`ja_domain_classifier_full.py` のみ追加オプションあり:

- `--gpu-id <N>` … 使用 GPU を明示指定
- `--no-gradient-checkpointing` … 勾配チェックポイントを無効化（VRAM に余裕がある場合）

### 推論方法

Python:

```bash
python ja_domain_classifier_full.py --mode test --model-path ja_full_domain_classifier_ruri-v3-30m
```

Go:

```bash
export LD_LIBRARY_PATH=$(pwd)/../../../../candle-binding/target/release
CGO_ENABLED=1 go run ./verify_domain --model ja_full_domain_classifier_ruri-v3-30m
```

## 他プロジェクトへ移植する際のチェックリスト

1. `common_lora_utils.py` をコピーするか、`get_model_mapping()` /
   `get_max_length_for_model()` 相当の関数で `cl-nagoya/ruri-v3-30m`
   （最大系列長 8192）を解決できるようにする。
2. データ準備関数（`load_*_dataset` / `create_*_dataset`）だけを自プロジェクトの
   データソースに差し替える。学習ループ（`TrainingArguments` / `Trainer` 呼び出し）
   はタスク間でほぼ共通なので変更不要。
3. 学習後は `label_mapping.json`（Python 用）と、必要であれば Go 互換マッピング
   （`category_mapping.json` / `pii_type_mapping.json` / `jailbreak_type_mapping.json`）
   の両方が出力されていることを確認する。
4. Go/Rust から使う場合は `candle-binding` をビルドし
   （リポジトリルートで `make rust`、または `cd candle-binding && cargo build --release`）、
   `LD_LIBRARY_PATH` に `candle-binding/target/release` を通してから `verify_*` を実行する。
5. 本番投入前に `--max-samples` を実データ全量（または十分な件数）に増やし、
   複数エポックで評価指標（Accuracy/F1）を確認すること。本ガイド中の学習コマンドは
   スクリプトのデフォルト引数に基づく推奨パラメータだが、データ量や
   ドメインに応じて調整すること。

## トラブルシューティング

- **`ImportError: ... requires accelerate>=1.1.0`**: `uv pip install accelerate`
- **`ModuleNotFoundError: No module named 'transformers'`**:
  `uv pip install torch transformers accelerate datasets scikit-learn pandas requests`
- **`uv pip install` で PyTorch 以外のパッケージが見つからない**:
  `--index-url` ではなく `--extra-index-url` を使うこと。
- **`cannot find -lcandle_semantic_router`（Go ビルドエラー）**:
  `candle-binding` の Rust ライブラリを先にビルドし、`LD_LIBRARY_PATH` を
  設定すること（上記チェックリスト参照）。
