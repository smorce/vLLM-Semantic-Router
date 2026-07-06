# 変更概要（未コミット diff）

> 生成日: 2026-07-06  
> ブランチ: `main`（origin/main と同期）  
> ステータス: 未ステージ 8 ファイル + 未追跡 2 ファイル

---

## 全体像

ローカル開発（`vllm-sr serve`）を安定させるための **設定・CLI・ドキュメント整備** と、ルーター本体の **2 件のバグ修正／機能追加**、および Intent Classification 向け **フルファインチューニングスクリプトの追加** が含まれる。

| 領域 | 変更ファイル数 | 主な目的 |
| --- | --- | --- |
| ドキュメント | 2（+1 新規） | 起動手順・ストレージ依存の明文化 |
| 設定 | 1 | Docker コンテナ名への合わせ・backend 統一 |
| CLI | 1 | ratelimit 用 Redis の自動起動検出 |
| ルーター（Go） | 4 | fuzzy キーワード対応・埋め込み初期化順序修正 |
| 学習スクリプト | 2（+1 新規） | フル FT 追加と README 更新 |

---

## 1. ドキュメント（`README.md`）

### 追加内容

- **初回起動時の HF_TOKEN 必須化**: gated モデル（`mmbert` 等）のダウンロードに同シェルで `export HF_TOKEN=...` が必要である旨を追記。
- **`VLLM_SR_SIM_ENABLED=false`**: sim サイドカー不要時の任意設定例。
- **`google/embeddinggemma-300m`**: Hugging Face 利用申請が必要な場合の対処（ライセンス同意リンク付き）。
- **ローカルストレージ依存テーブル**: `response_api` / `ratelimit` / `router_replay` / `semantic_cache` / `memory` プラグインが要求する backend と、`serve` が起動するコンテナ名の対応表。
- **Milvus ホスト名注意**: `milvus` ではなく `vllm-sr-milvus`（コンテナ名）に合わせる必要がある旨。

---

## 2. 設定（`config/config.yaml`）

### ルーティング DSL 修正

- `routing.signals.keyword_rules` 内の `image_candidates` キーを削除し、候補を `candidates` リストに統合（YAML 構造の修正）。

### ストレージ backend の Docker 向け調整

| 設定項目 | 変更前 | 変更後 |
| --- | --- | --- |
| `response_api.store_backend` | `memory` | `redis` |
| `ratelimit.providers[].address` | `redis:6379` | `vllm-sr-redis:6379` |
| `startup_status.store_backend` | `file` | `redis`（+ redis 接続設定追加） |
| `router_replay.postgres.host` | `postgres` | `vllm-sr-postgres` |
| `semantic_cache.milvus.connection.host` | `milvus` | `vllm-sr-milvus` |
| `stores.memory.milvus.address` | `milvus:19530` | `vllm-sr-milvus:19530` |
| Flow プラグイン `state.store_backend` | `file` | `redis`（address/db/password 更新） |

**意図**: `vllm-sr serve` が起動する Docker ネットワーク上のコンテナ名（`vllm-sr-*`）と一致させ、reference config の各機能が正しく接続できるようにする。

---

## 3. CLI（`src/vllm-sr/cli/service_defaults.py`）

### 新規: `_ratelimit_requires_redis()`

- `global.services.ratelimit` が有効かつ provider の `type: redis` を含む場合、`detect_canonical_storage_backends()` の戻り値に `"redis"` を追加。
- **効果**: ratelimit 設定だけで Redis が必要な場合も、`serve` が `vllm-sr-redis` を自動起動する。

---

## 4. ルーター — キーワード分類（Go）

### `keyword_classifier.go`

- **`fuzzy` メソッドの正式サポート**: DSL で `method: fuzzy` を指定可能に。
- 新関数 `normalizeKeywordRuleMethod()`: `fuzzy` → 内部では `regex` + `FuzzyMatch: true` に正規化。
- エラーメッセージに `fuzzy` を valid method として追加。

### `keyword_classifier_structural_test.go`

- **`TestNewKeywordClassifierAcceptsFuzzyMethod`**: `method: "fuzzy"` で classifier 構築・分類（typo `"pasword"` → `"password (fuzzy)"` マッチ）を検証。

---

## 5. ルーター — 埋め込みランタイム初期化（Go）

### `router_runtime.go`

- **バグ修正**: マルチモーダル埋め込みタスク（`router.embedding.multimodal`）が、統一ファクトリ（`router.embedding.unified_factory`）より先に走ると `GLOBAL_MODEL_FACTORY` を先取りし、mmBERT が利用不可になる競合状態を修正。
- 統一モデル（qwen3/gemma/mmbert）が存在する場合、マルチモーダルタスクに `Dependencies: ["router.embedding.unified_factory"]` を付与。

### `router_runtime_test.go`

- **`TestMultiModalEmbeddingTaskDependsOnUnifiedFactory`**: 上記依存関係が正しく設定されることを検証。

---

## 6. Intent Classification 学習（Python）

### 新規: `classifier_model_fine_tuning_lora/ft_linear_full.py`

- LoRA 版 `ft_linear_lora.py` をベースに、**全パラメータフルファインチューニング**用スクリプトを追加。
- `peft` 非依存。`AutoModelForSequenceClassification` + 標準 `Trainer`。
- MMLU-Pro 14 カテゴリ分類。`gradient_checkpointing` デフォルト有効。
- 出力: `full_intent_classifier_{model}` ディレクトリ。

### 新規: `classifier_model_fine_tuning_lora/README.md`

- LoRA vs フル FT の比較表、使い方、推奨ハイパーパラメータ、トラブルシュート（日本語）。

### 更新: `src/training/model_classifier/README.md`

- Intent Classification セクションを MMLU-Pro 14 クラス・2 スクリプト構成に更新。
- ディレクトリツリーに `README.md` / `ft_linear_full.py` を反映。
- フル FT の実行例コマンドを追加。

---

## 変更ファイル一覧

### 変更済み（modified）

| ファイル | 種別 |
| --- | --- |
| `README.md` | ドキュメント |
| `config/config.yaml` | 設定 |
| `src/vllm-sr/cli/service_defaults.py` | CLI |
| `src/semantic-router/pkg/classification/keyword_classifier.go` | 機能追加 |
| `src/semantic-router/pkg/classification/keyword_classifier_structural_test.go` | テスト |
| `src/semantic-router/pkg/modelruntime/router_runtime.go` | バグ修正 |
| `src/semantic-router/pkg/modelruntime/router_runtime_test.go` | テスト |
| `src/training/model_classifier/README.md` | ドキュメント |

### 未追跡（untracked）

| ファイル | 種別 |
| --- | --- |
| `src/training/model_classifier/classifier_model_fine_tuning_lora/README.md` | ドキュメント（新規） |
| `src/training/model_classifier/classifier_model_fine_tuning_lora/ft_linear_full.py` | 学習スクリプト（新規） |

---

## 推奨検証

```bash
# Go ルーター（変更箇所のテスト）
make test-semantic-router

# エージェント lint（変更ファイル指定）
make agent-lint CHANGED_FILES="README.md config/config.yaml src/vllm-sr/cli/service_defaults.py src/semantic-router/pkg/classification/keyword_classifier.go src/semantic-router/pkg/classification/keyword_classifier_structural_test.go src/semantic-router/pkg/modelruntime/router_runtime.go src/semantic-router/pkg/modelruntime/router_runtime_test.go"

# ローカル serve 動作確認（HF_TOKEN 設定後）
export HF_TOKEN=hf_xxxxxxxx
UV_LINK_MODE=copy uv run --project src/vllm-sr vllm-sr serve --config config/config.yaml
```

---

## コミット分割の提案

関心ごとに分ける場合の例:

1. **`[Router]`** fuzzy キーワード method 対応 + テスト
2. **`[Router]`** マルチモーダル埋め込み初期化順序修正 + テスト
3. **`[CLI/Config]`** Docker 向け storage backend 調整 + ratelimit Redis 検出
4. **`[Docs]`** README 起動・ストレージ依存の追記
5. **`[Training]`** Intent Classification フル FT スクリプト + README
