# 変更概要（未コミット diff）

> 更新日: 2026-07-06 23:53（JST）  
> 対象: 現在の working tree / 未コミット差分  
> 前回サマリー: `CHANGE_SUMMARY_20260706_0239.md`

---

## 全体像（前回からの増分）

前回（02:39）の「ローカル開発の安定化（storage backend / docs / CLI）」と「Router の小規模修正（fuzzy・初期化順序）」に加え、今回の diff では次が大きく増えています。

- **`README.md`**: ローカル運用（ビルド・ルーティング・検証）を一通り実行できる形に拡充
- **`config/config.yaml`**: ルーティング精度の改善（閾値調整・キーワード補助・decision 優先順位の見直し）と、ローカル split topology（Envoy/Router）前提の設定整理
- **Router（Go）**: classification / config / extproc の広範な更新（intent 追加、preference/usage などのシグナル拡張を含む）
- **学習（Training）**: 既存の LoRA/Full FT に加え、日本語フルファインチューニング用のディレクトリ（`ja_full_finetuning/`）が新規追加
- **検証（新規）**: `test_model_routing.py`（MoM / cache / conversation をまとめて検証）
- **CLI（vllm-sr chat）**: チャットクライアント機能拡張とテスト追加

差分統計（作業ツリー）:

- **48 files changed**（約 **+789 / -120**）
- 未追跡（新規）に `test_model_routing.py` や `ja_full_finetuning/` などが追加

---

## 1. ドキュメント（`README.md`）

### 追加・更新ポイント

- **HF_TOKEN の export** を明示（初回の埋め込みモデル DL に必要）
- `vllm-sr serve` を **ローカルビルドイメージ**で起動するための `--image-pull-policy never` を追記
- **Go コード変更時の再ビルド手順**（`make vllm-sr-dev` / `docker build`）を追加
- **Docker ビルドでの permission denied**（Milvus data がビルドコンテキストに入る問題）への対処を追記
- **ルーティングの使い方**（MoM、自動/明示指定、Dashboard/Metrics、検証スクリプト）を追加
- **`test_model_routing.py` の実行方法**とオプションを追記
- Responses API のストア説明を **redis 前提**へ更新（必要に応じて `memory` に戻せる旨）

---

## 2. 設定（`config/config.yaml`）

今回の差分は「ローカルで MoM を回しながら検証できる reference config」方向の調整が中心です。

### モデル/バックエンド表現の整備

- `backend_refs[].type` を `chat` から **`vllm` / `llama`** へ明示（`api_key` も設定）
- `external_model_ids` に `vllm` / `chat` / `llama` を追加し、外部表現の揺れを吸収
- `provider_model_id` 表記を整理（例: `unsloth/...` → `Qwen3.6-27B-MTP-GGUF-UD-Q4_K_XL`）

### ルーティング精度・安定性の改善

- **キーワード補助 signal** 追加:
  - `business_keywords`（bm25）
  - `legal_fact_keywords`（bm25）
  - domain 分類器のブレを補い、フォールバックに落ちにくくする意図
- **complexity 閾値**を実測ベースで調整:
  - `needs_reasoning.threshold: 0.75 → 0.3`
  - hard/easy の candidates を現実の分布に寄せて margin を改善
- `static_business_route`:
  - `operator: AND → OR`（domain だけでなく keywords でも入れる）
  - tier の注意コメント（tier 選択が priority より強い問題への対処）
  - memory plugin を route 側で **明示的に無効化**（personalize 判定で cache がスキップされる挙動を避ける）
- `computer-science-remom-route`:
  - domain が外れても `code_keywords` + hard なら乗せる OR ルール追加
- `on_failure: block → skip`（vector store deref の失敗時挙動を緩和）
- `safe_only_svm_route`:
  - 真の最終フォールバック化のため priority を下げる（NOT jailbreak がほぼ常時 true で先に当たりやすい問題への対処）
- `global.router`:
  - `auto_model_name: exhaustive-reference → MoM`
  - `clear_route_cache: false → true`（複数 backend での誤ルート回避）

### Split topology（Envoy/Router 分離）向け調整

- `global.looper.endpoint` を `localhost:8899` ではなく **Envoy コンテナ名**経由に変更

### 分類器モデルの更新（日本語 Full FT）

- `global.models.system.*` が **日本語フルファインチューニングモデル**へ切替
  - jailbreak / domain / intent / pii（ruri-v3-30m ベース）
- intent classifier の設定ブロックを新設（`fallback_category: no_function_needed`）
- preference embedding の指定を `models/preference-encoder` から **`mmbert`** へ変更

---

## 3. Router（Go）: 追加・拡張された領域（概要）

前回の `keyword_classifier` / `router_runtime` 変更に加えて、今回の差分では以下が広く更新されています（ファイル数が多いため、詳細は個別 diff を参照）。

- **classification**:
  - classifier 構築・ライフサイクル・signal dispatch / results / usage の拡張
  - 新規 untracked:
    - `classifier_intent_init.go`
    - `intent_classifier_runtime.go`
    - `classifier_signal_preference_support_test.go`
- **config**:
  - canonical defaults/export/global/registry/signal config の追加・拡張
- **extproc / decision**:
  - decision engine 周りの調整
  - router build/components の拡張
- **cache（Milvus）**:
  - `milvus_cache.go` の更新

---

## 4. CLI（Python）: chat とローカル起動支援

- `src/vllm-sr/cli/service_defaults.py`:
  - ratelimit の redis provider を検出して backend 必須判定に追加（前回記載の継続）
- `src/vllm-sr/cli/chat_client.py`（更新）:
  - chat クライアント機能の追加/拡張
- `src/vllm-sr/cli/commands/chat.py`（更新） + `src/vllm-sr/tests/test_chat_command.py`（更新）:
  - chat コマンドの挙動をテストで担保

---

## 5. 検証（新規）: `test_model_routing.py`

Router（`:8801`）経由で次をまとめて検証するスクリプト。

- 到達性（Router / vLLM / llama-server）
- 明示モデル指定（`lfm2.5-1.2b-jp` / `qwen3.6-27b`）
- MoM ルーティング（複数の意図別テストケース）
- semantic cache（2回目 hit を確認）
- Responses API 会話継続（`previous_response_id`）

実行例:

```bash
UV_LINK_MODE=copy uv run --project src/vllm-sr python test_model_routing.py --skip-slow
```

---

## 6. Training: intent classifier（Full FT）と日本語 full finetuning 追加

前回追加の `classifier_model_fine_tuning_lora/ft_linear_full.py` などに加え、今回以下が増えています。

- 新規ディレクトリ: `src/training/model_classifier/ja_full_finetuning/`
  - `.gitignore`, `README.md`, `TRAINING_GUIDE.md`
  - 日本語向け full finetuning スクリプト群（domain/intent/jailbreak/pii）
  - Go の verify（`verify_*/main.go`）と `go.mod`
- `src/training/model_classifier/common_lora_utils.py` の更新

---

## 7. 未追跡（untracked）ファイル一覧（現時点）

※ `git ls-files --others --exclude-standard` の結果に基づく。

- `CHANGE_SUMMARY_20260706_0239.md`（前回サマリー）
- `config.yaml`（リポジトリルート直下。用途を要確認）
- `test_model_routing.py`
- `src/semantic-router/pkg/classification/classifier_intent_init.go`
- `src/semantic-router/pkg/classification/classifier_signal_preference_support_test.go`
- `src/semantic-router/pkg/classification/intent_classifier_runtime.go`
- `src/training/model_classifier/classifier_model_fine_tuning_lora/README.md`
- `src/training/model_classifier/classifier_model_fine_tuning_lora/ft_linear_full.py`
- `src/training/model_classifier/ja_full_finetuning/`（配下多数）

---

## 推奨検証（ローカル）

```bash
# Go
make test-semantic-router

# ルーティング E2E（serve 起動後）
UV_LINK_MODE=copy uv run --project src/vllm-sr python test_model_routing.py --skip-slow
```

---

## コミット分割の提案（レビューしやすさ重視）

1. **`[Router]`** intent/runtime・classification 拡張 + テスト
2. **`[Config]`** ルーティング精度調整（threshold / keyword 補助 / priority）+ split topology 修正
3. **`[CLI]`** chat client/command + tests
4. **`[Training]`** 日本語 full finetuning（`ja_full_finetuning/`）+ 既存 training 更新
5. **`[Test]`** `test_model_routing.py`
6. **`[Docs]`** README 拡充（build/serve/validate）

