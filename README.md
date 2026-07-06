https://github.com/smorce/vLLM-Semantic-Router

## 起動コマンド

初回起動時、ルーターはシグナル用の埋め込みモデル（`mmbert` など）を `models/` にダウンロードします。**同じシェルで `HF_TOKEN` を export してから** `serve` してください（コンテナへ自動で渡されます）。

```
export HF_TOKEN=hf_xxxxxxxx   # 必須（gated モデル・高速 DL 用）
export VLLM_SR_SIM_ENABLED=false   # 任意（sim サイドカー不要なら）

UV_LINK_MODE=copy uv run --project src/vllm-sr vllm-sr serve --config config/config.yaml --image-pull-policy never
```

### Go コード変更時の再ビルド

`vllm-sr serve` は **Docker コンテナ内の router バイナリ** を実行します。`src/semantic-router/` など Go 側を変更しただけでは、設定 YAML を直しただけでは **実行中イメージは更新されません**。変更を反映するには、serve の前に router イメージを必ず再ビルドしてください。

```bash
# 推奨: router イメージ再ビルド + CLI 再インストール
make vllm-sr-dev

# または router イメージだけ再ビルド
docker build -t ghcr.io/vllm-project/semantic-router/vllm-sr:latest -f src/vllm-sr/Dockerfile .
```

再ビルド後に serve します。

```bash
UV_LINK_MODE=copy uv run --project src/vllm-sr vllm-sr serve --config config/config.yaml --image-pull-policy never
```

**Docker ビルドが `.vllm-sr/milvus-data/etcd: permission denied` で失敗する場合**  
Milvus のデータディレクトリ（root 所有）がビルドコンテキストに含まれていたのが原因です。リポジトリルートの `.dockerignore` で `.vllm-sr/` を除外しているため、通常は再発しません。古いイメージのまま `serve` すると起動ログに以前のエラー（例: `method: fuzzy` 未対応）が残るので、ビルド成功を確認してから起動してください。

`google/embeddinggemma-300m`（`mom-embedding-flash`）は Hugging Face 上で利用申請・承認が必要な場合があります。ログに `Access denied. This repository requires approval` と出たら、[モデルページ](https://huggingface.co/google/embeddinggemma-300m) でライセンス同意を済ませてください。

予め vLLM を起動しておく必要あり。

`config/config.yaml` は v0.3 reference config をベースにしています。

主なモデルは次です。

- default (`lfm2.5-1.2b-jp`): `LiquidAI/LFM2.5-1.2B-JP` @ `172.17.0.1:3849` (vLLM)
- premium/analysis (`qwen3.6-27b`): `Qwen3.6-27B-MTP-GGUF-UD-Q4_K_XL` @ `172.17.0.1:1067` (llama-server)
- multimodal: `llava-omni`（reference 用・未接続）
- image: `sdxl-image`（reference 用・未接続）

`vllm-sr serve` は Docker コンテナ内で動くため、ホスト上の backend には `172.17.0.1` を使います。

## ルーティングの使い方

### 前提

1. **バックエンド LLM** をホスト側で起動する（本 config の例）
   - 軽量: vLLM @ `172.17.0.1:3849`（`lfm2.5-1.2b-jp`）
   - 重量: llama-server @ `172.17.0.1:1067`（`qwen3.6-27b`）
2. **Semantic Router** を起動する（上記「起動コマンド」）
3. クライアントは **vLLM / llama-server ではなく Router（`:8801`）** に向ける

### 基本: 自動ルーティング（MoM）

リクエストの `model` に **`MoM`**（Mixture-of-Models）を指定すると、Router がプロンプトのシグナル（ドメイン・複雑度・キーワード・埋め込み類似度など）と `config/config.yaml` の **`decisions`** ルールに従い、適切な論理モデル（`lfm2.5-1.2b-jp` や `qwen3.6-27b` など）へ振り分けます。

```bash
# ワンショット確認（CLI）。自動ルーティング（通常はこれ）
# serve と同じ config パスを --config で指定する（省略するとカレントの config.yaml を探す）
UV_LINK_MODE=copy uv run --project src/vllm-sr vllm-sr chat --config config/config.yaml --prompt "2次方程式 x^2+3x+2=0 を解いて"

# curl（Chat Completions）
curl -s http://localhost:8801/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "MoM",
    "messages": [{"role": "user", "content": "What is the derivative of x^2?"}]
  }'
```

**モデルを明示指定**したい場合は、論理モデル名をそのまま渡します（ルーティングをバイパスして固定 backend へ）。

| `model` の指定 | 挙動 |
| --- | --- |
| `MoM` | シグナル + decision ルールで自動選択（通常はこれ） |
| `lfm2.5-1.2b-jp` | 常に vLLM（軽量）へ |
| `qwen3.6-27b` | 常に llama-server（重量）へ |

### ルーティングが効いているか確認する

- **Dashboard**: http://localhost:8700 — リクエストごとの decision / 選択モデルを確認
- **Metrics**: http://localhost:9190/metrics
- 応答ヘッダや Dashboard のトレースで、意図した decision（例: `urgent_automix_route`, `computer-science-remom-route`）が選ばれているかを見る
- **自動テストスクリプト**: ルーティング・MoM・キャッシュ・会話継続をまとめて検証する

```bash
# フルスイート（ReMoM 含むと数分かかる場合あり）
UV_LINK_MODE=copy uv run --project src/vllm-sr python test_model_routing.py

# ReMoM（mom_cs_remom）を省略して速く回す
UV_LINK_MODE=copy uv run --project src/vllm-sr python test_model_routing.py --skip-slow

# ルーティングだけ（キャッシュ・会話継続を省略）
UV_LINK_MODE=copy uv run --project src/vllm-sr python test_model_routing.py \
  --skip-cache --skip-conversation
```

`test_model_routing.py` の主な検証項目:

| 区分 | 内容 |
| --- | --- |
| Preflight | Router / vLLM / llama-server の到達性とモデル ID |
| Explicit | `lfm2.5-1.2b-jp` / `qwen3.6-27b` の固定ルーティング |
| MoM | シグナル設計済みプロンプト（`multi_factor_route`, business/static 意図, ML キーワード意図, fact-check エスカレーション, CS/ReMoM 意図） |
| Cache | `static_business_route` 上の semantic-cache（2 回目で `x-vsr-cache-hit: true`）。business 分類に乗らない環境では SKIP |
| Conversation | Responses API の `previous_response_id`（「私の名前はステラです」→「覚えていますか？」） |

オプション:

- `--skip-slow` … fact-check / ReMoM 系の MoM ケースを省略
- `--require-cache` … business ルートに乗らずキャッシュ検証できない場合も FAIL にする（通常は SKIP）
- `--skip-cache` / `--skip-conversation` … 部分実行

前提:

- `vllm-sr serve` 実行中であること
- 複数バックエンド構成では `global.router.clear_route_cache: true` が必要（`false` だと Qwen が vLLM デフォルトへ落ちる）
- 会話継続テストは `response_api.enabled: true` かつ `vllm-sr-redis` が起動していること（`response_api.redis.config_path` は不要）

### 会話を続ける（Responses API）

複数ターンの履歴は **Response API** + Redis ストア（本 config は `response_api.store_backend: redis`）を使います。`previous_response_id` で前ターンを引き継ぎます（詳細は下記「クライアントからの接続」）。

## ローカルストレージ依存（`vllm-sr serve` が起動するコンテナ）

`response_api.store_backend: redis` など、reference config の機能が要求する backend に応じて `serve` が sidecar を自動起動します。

| 用途 | 設定 | 起動コンテナ |
| --- | --- | --- |
| Responses API 会話履歴 | `response_api.store_backend: redis` | `vllm-sr-redis` |
| 起動状態（Dashboard 表示） | `startup_status.store_backend: redis` | 同上 |
| Flow プラグイン状態 | `plugins.flow.state.store_backend: redis` | 同上 |
| レート制限 | `ratelimit.providers[].type: redis` | 同上 |
| ルーティングリプレイ | `router_replay.store_backend: postgres` | `vllm-sr-postgres` |
| セマンティックキャッシュ | `semantic_cache.backend_type: milvus` | `vllm-sr-milvus` |
| メモリプラグイン | `stores.memory.backend: milvus` | 同上（Milvus 共有） |

Milvus の `connection.host` は **`vllm-sr-milvus`**（コンテナ名）に合わせてあります。`milvus` のままだと `serve` が Milvus を起動しないため注意してください。

初回 `serve` 時のログ例: `Storage backends required by config: milvus, postgres, redis`

## クライアントからの接続（Chat Completions / Responses API）

アプリは **Semantic Router**（`http://127.0.0.1:8801`）に向けます。llama-server や vLLM に直接繋ぐと、シグナルによる振り分けは効きません。

```
あなたのアプリ（OpenAI SDK など）
        │
        │  POST /v1/chat/completions  または  POST /v1/responses
        ▼
Semantic Router (:8801)  … 依頼内容に応じて lfm2.5-1.2b-jp / qwen3.6-27b へルーティング
        │
        │  backend へは Chat Completions 形式で転送（Router 内部の話）
        ▼
vLLM (:3849)  /  llama-server (:1067)
```

| やりたいこと | クライアントが使う API | 補足 |
| --- | --- | --- |
| 1 回きりの質問 | `chat.completions` でも `responses.create` でも可 | Router が backend 用に変換するので、どちらでも同じようにルーティングされる |
| 会話を続ける（`previous_response_id`） | `responses.create` 推奨 | 前回の `response.id` を渡す。履歴は Router 側に保存される（下記） |
| 過去の応答を取得 | `GET /v1/responses/{id}` | Response API ストアが有効なときのみ |

### Response API ストア（会話履歴）

`previous_response_id` や `GET /v1/responses/{id}` は、Router が応答を覚えておく **ストア** が必要です。

- 本 config: `global.services.response_api.store_backend: redis`（再起動後も Redis に残る）
- 手軽な一時利用: `memory` に変更するとプロセス内のみ（再起動で消える）

1 ターンだけのリクエストなら、ストアの種類は気にしなくて大丈夫です。

### Python（`responses.create`）を Router 経由で使う例

llama-server 直ではなく Router を向け、**論理モデル名**を指定します。

```bash
export LLAMA_SERVER_BASE_URL=http://127.0.0.1:8801
export LLM_MODEL=qwen3.6-27b          # 重い処理を明示したいとき
# export LLM_MODEL=auto               # ルーターに任せるとき（設定による）
export LLAMA_SERVER_API_KEY=sk-local-no-key-required
```

```python
# call_llama_server.py の LlamaServerEnvConfig.from_env() のまま使える
cfg = LlamaServerEnvConfig.from_env()
text = await cfg.complete("こんにちは")
```

会話を続けるときは、前回レスポンスの `id` を次の `previous_response_id` に渡します（OpenAI SDK の Responses API と同じ使い方）。

## 停止コマンド

```
UV_LINK_MODE=copy uv run --project src/vllm-sr vllm-sr stop
```

停止して再開するまではキャッシュが効いているみたいで毎回同じ回答になる。
ただ、本当にキャッシュを使っているのかまでは分からない。vllm-sr serve の仕様を調査してみないと分からない。
GPT OSS に時間がかかるので、キャッシュは使えてない気がする。
→ semantic-cache が OFF だったので ON にしたら
TEST: T04_gpt_api_code
TEST: T05_gpt_compare
が爆速で返ってくるようになったのでキャッシュの効き方もOK。

---

## About

In the LLM era, the number of models is exploding. Different models vary across capability, scale, cost, and privacy boundaries. Choosing and connecting the right models to build semantic AI infrastructure is a system problem.

**vLLM Semantic Router** is a **signal-driven** intelligent router for that problem. It helps teams build model systems that are more **efficient**, **safer**, and more **adaptive** across cloud, data center, and edge environments.

![system](website/static/img/system.png)

It delivers three core values:

- **Token economics**: reduce wasted tokens, increase effective output, and maximize the value of every token.
- **LLM safety**: detect jailbreaks, sensitive leakage, and hallucinations so agents remain controllable, trustworthy, and auditable.
- **Fullmesh intelligence**: build personal AI at the edge and intelligent MaaS in the cloud by coordinating local, private, and frontier models across cost, privacy, and capability boundaries.

## Getting Started

### Install

```bash
curl -fsSL https://vllm-semantic-router.com/install.sh | bash
```

For platform notes, detailed setup options, and troubleshooting, see the **[Installation Guide](https://vllm-semantic-router.com/docs/installation/)**.

> [!IMPORTANT]
> Online [playground](https://play.vllm-semantic-router.com) default credentials:
>
> <!-- markdownlint-disable MD004 MD032 -->
> + username: `love@vllm-sr.ai`
> + password: `vllm-sr`
> <!-- markdownlint-enable MD004 MD032 -->

## Latest News

- [2026/03/24] Vision Paper Released: [The Workload-Router-Pool Architecture for LLM Inference Optimization](https://vllm-semantic-router.com/vision-paper)
- [2026/03/10] v0.2 Released: [vLLM Semantic Router v0.2 Athena Release](https://vllm.ai/blog/v0.2-vllm-sr-athena-release)
- [2026/02/27] White Paper Released: [Signal Driven Decision Routing for Mixture-of-Modality Models](https://vllm-semantic-router.com/white-paper/)
- [2026/01/05] Iris v0.1 Released: [vLLM Semantic Router v0.1 Iris: The First Major Release](https://blog.vllm.ai/2026/01/05/vllm-sr-iris.html)
- [2025/12/16] Collaboration: [AMD × vLLM Semantic Router: Building the System Intelligence Together](https://blog.vllm.ai/2025/12/16/vllm-sr-amd.html)
- [2025/11/19] New Blog: [Signal-Decision Driven Architecture: Reshaping Semantic Routing at Scale](https://blog.vllm.ai/2025/11/19/signal-decision.html)
- [2025/11/03] Paper Published: [Category-Aware Semantic Caching for Heterogeneous LLM Workloads](https://arxiv.org/abs/2510.26835)
- [2025/10/12] Paper Accepted: [When to Reason: Semantic Router for vLLM](https://arxiv.org/abs/2510.08731)

<details>
<summary>Earlier announcements</summary>

- [2025/12/15] New Blog: [Token-Level Truth: Real-Time Hallucination Detection for Production LLMs](https://blog.vllm.ai/2025/12/14/halugate.html)
- [2025/10/27] New Blog: [Scaling Semantic Routing with Extensible LoRA](https://blog.vllm.ai/2025/10/27/semantic-router-modular.html)
- [2025/10/08] Collaboration: vLLM Semantic Router with [vLLM Production Stack](https://github.com/vllm-project/production-stack) Team.
- [2025/09/01] Released the project: [vLLM Semantic Router: Next Phase in LLM inference](https://blog.vllm.ai/2025/09/11/semantic-router.html).

</details>

More announcements are available on the **[Blog](https://vllm-semantic-router.com/blog/)** and **[Publications](https://vllm-semantic-router.com/publications/)** pages.

## Community

For questions, feedback, or to contribute, please join the `#semantic-router` channel in vLLM Slack.

### Community Meetings

We host community meetings on the first and third Tuesday of each month to sync with contributors across different time zones:

- **First Tuesday of the month**: 9:00-10:00 AM EST (accommodates US EST, EU, and Asia Pacific contributors)
  - [Zoom Link](https://us05web.zoom.us/j/84122485631?pwd=BB88v03mMNLVHn60YzVk4PihuqBV9d.1)
  - [Google Calendar Invite](https://us05web.zoom.us/meeting/tZAsdeuspj4sGdVraOOR4UaXSstrH2jjPYFq/calendar/google/add?meetingMasterEventId=4jjzUKSLSLiBHtIKZpGc3g)
  - [ics file](https://drive.google.com/file/d/15wO8cg0ZjNxdr8OtGiZyAgkSS8_Wry0J/view?usp=sharing)
- **Third Tuesday of the month**: 1:00-2:00 PM EST (accommodates US EST and California contributors)
  - [Zoom Link](https://us06web.zoom.us/j/86871492845?pwd=LcTtXm9gtGu23JeWqXxbnLLCCvbumB.1)
  - [Google Calendar Invite](https://us05web.zoom.us/meeting/tZIlcOispzkiHtH2dlkWlLym68bEqvuf3MU5/calendar/google/add?meetingMasterEventId=PqWz2vk7TOCszPXqconGAA)
  - [ics file](https://drive.google.com/file/d/1T54mwYpXXoV9QfR76I56BFBPNbykSsTw/view?usp=sharing)
- Meeting recordings: [YouTube](https://www.youtube.com/@vLLMSemanticRouter/videos)

## Contributing

If you want to contribute, start with **[CONTRIBUTING.md](CONTRIBUTING.md)**.

For repository-native development workflow and validation commands, use **[AGENTS.md](AGENTS.md)** as the entrypoint and **[docs/agent/README.md](docs/agent/README.md)** as the canonical index.

## Citation

If you find Semantic Router helpful in your research or projects, please consider citing it:

```
@misc{semanticrouter2025,
  title={vLLM Semantic Router},
  author={vLLM Semantic Router Team},
  year={2025},
  howpublished={\url{https://github.com/vllm-project/semantic-router}},
}
```

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=vllm-project/semantic-router&type=Date)](https://www.star-history.com/#vllm-project/semantic-router&Date)

## Sponsors

We are grateful to our sponsors who support us:

---

[**AMD**](https://www.amd.com) provides us with GPU resources and [ROCm™](https://www.amd.com/en/products/software/rocm.html) software for training and researching frontier router models, enhancing E2E testing, and building the online models playground.

<div align="center">
<a href="https://www.amd.com">
  <img src="website/static/img/amd-logo.svg" alt="AMD" width="40%"/>
</a>
</div>

---
