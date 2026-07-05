以下を実装してテストまで実行してください。vLLM は起動済みで、LiquidAI/LFM2.5-1.2B-JP を使っています。
Ollama は Windows で動いていて vLLM は WSL で動いています。この環境は WSL です。


* Ollama は OpenAI互換（Chat Completions）を提供しますが、**別マシン/別コンテナ**から叩くには **到達できるアドレス**で待ち受ける必要があります。([Ollama][2])
* vLLM-SR は backend を `address`/`port` で定義してそこへ転送する作りです。([vLLM Semantic Router][3])
* vLLM-SR の入口は Envoy 経由（ポート `8801`）で OpenAI互換の `/v1/chat/completions` を受ける、という説明があります。([GitHub][4])

---

## 0) 前提（あなたの構成）

* **Windows 側**：Ollama（GPT-OSS 20B）
* **Linux/WSL/Docker 側**：vLLM（モデルA） + vLLM-SR

ここから「インストール → 設定 → 起動 → テスト」を Windows Ollama 前提で通します。

---

## 1) Windows 側：Ollama を外部から叩けるようにする
→done

## 2) vLLM-SR を動かす側：Windows Ollama へ届くことを確認

### 2-1. どのアドレスで Windows を指すか（2パターン）

* **(A) vLLM-SR が Docker コンテナ内で動く**：`host.docker.internal` が使えることが多いです（同種の OpenAI互換接続での定番）。([n8n Community][7])
* **(B) vLLM-SR が WSL2 / Linux で動く**：Windows の LAN IP（例：`192.168.x.y`）を使うのが確実です。

まずは vLLM-SR 側から、どちらかで疎通確認してください。

```bash
# パターンA（Docker想定）
curl -s http://host.docker.internal:11434/v1/models | head

# パターンB（WindowsのLAN IPを使う。ipconfigで調べた値に置換）
WIN_IP=192.168.1.50
curl -s http://$WIN_IP:11434/v1/models | head
```

どちらもダメなら、ほぼ確実に **(1) bind（OLLAMA_HOST）未設定**か **(2) ファイアウォール**です。([Ollama][1])

---

## 3) vLLM Semantic Router のインストール（vLLM-SR 側）

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install vllm-sr
vllm-sr --version
```

（設定は YAML 1つで signals/decisions/backend を書けます。）([vLLM Semantic Router][3])

---

## 4) `config.yaml`（Windows Ollama を指すように修正版）

### 4-1. まず値を決める

* `FAST_MODEL`：vLLM 側のモデルA（`http://127.0.0.1:8000/v1/models` の結果に合わせる）
* `GPT_MODEL`：Ollama 側の GPT-OSS 20B（`http://<windows>:11434/v1/models` の結果に合わせる）
* `OLLAMA_ADDR`：上の疎通で成功した方

  * Dockerなら `host.docker.internal`
  * それ以外は Windows の LAN IP

```bash
export FAST_MODEL="modelA"
export GPT_MODEL="gpt-oss:20b"
export OLLAMA_ADDR="host.docker.internal"   # 例：Dockerの場合
# export OLLAMA_ADDR="192.168.1.50"         # 例：LAN IPの場合
```

### 4-2. config.yaml を作成（「簡単→vLLM」「難しい/Tool→GPT」）

※シグナル6種（keyword / embedding / domain / fact_check / user_feedback / preference）を全部定義し、優先度で合成します。([vLLM Semantic Router][3])

```bash
cat > config.yaml <<'YAML'
bert_model:
  model_id: sentence-transformers/all-MiniLM-L12-v2
  threshold: 0.6
  use_cpu: true

vllm_endpoints:
  - name: "vllm_fast"
    address: "127.0.0.1"
    port: 8000
    models: ["__FAST_MODEL__"]
    weight: 1

  - name: "ollama_gpt"
    address: "__OLLAMA_ADDR__"
    port: 11434
    models: ["__GPT_MODEL__"]
    weight: 1

model_config:
  "__FAST_MODEL__":
    preferred_endpoints: ["vllm_fast"]
  "__GPT_MODEL__":
    preferred_endpoints: ["ollama_gpt"]

default_model: "__FAST_MODEL__"

classifier:
  category_model:
    model_id: "models/category_classifier_modernbert-base_model"
    use_modernbert: true
    threshold: 0.6
    use_cpu: true

signals:
  keywords:
    - name: "tool_or_hard_keywords"
      operator: "OR"
      case_sensitive: false
      keywords:
        - "検索"
        - "調べて"
        - "最新"
        - "URL"
        - "API"
        - "ツール"
        - "tool"
        - "function"
        - "呼び出し"
        - "実行して"
        - "設計"
        - "原因"
        - "分析"
        - "比較"
        - "提案"

  embeddings:
    - name: "simple_ja"
      threshold: 0.72
      candidates: ["こんにちは", "短く答えて", "要約して", "雑談"]
      aggregation_method: "max"

    - name: "hard_or_tool_ja"
      threshold: 0.70
      candidates:
        - "ウェブで調べて根拠を付けて"
        - "ツールやAPIを呼び出して作業して"
        - "複雑な条件で比較して提案して"
      aggregation_method: "max"

  domains:
    - name: "computer_science"
      description: "Programming / CS"
      mmlu_categories: ["computer_security", "machine_learning"]

  fact_check:
    - name: "needs_verification"
      description: "Likely needs factual verification."

  user_feedbacks:
    - name: "correction_needed"
      description: "User indicates previous answer was wrong / asks for correction."

  preferences:
    - name: "hard_or_tool"
      description: "難しい推論やツール呼び出しが必要そう"
      llm_endpoint: "http://__OLLAMA_ADDR__:11434"
    - name: "easy"
      description: "簡単な返答や雑談"
      llm_endpoint: "http://__OLLAMA_ADDR__:11434"

decisions:
  - name: "feedback_to_gpt"
    priority: 300
    rules:
      operator: "AND"
      conditions:
        - type: "user_feedback"
          name: "correction_needed"
    modelRefs:
      - model: "__GPT_MODEL__"
        use_reasoning: true

  - name: "hard_or_tool_to_gpt"
    priority: 200
    rules:
      operator: "OR"
      conditions:
        - type: "keyword"
          name: "tool_or_hard_keywords"
        - type: "embedding"
          name: "hard_or_tool_ja"
        - type: "domain"
          name: "computer_science"
        - type: "fact_check"
          name: "needs_verification"
        - type: "preference"
          name: "hard_or_tool"
    modelRefs:
      - model: "__GPT_MODEL__"
        use_reasoning: true

  - name: "default_to_fast"
    priority: 1
    rules:
      operator: "OR"
      conditions:
        - type: "embedding"
          name: "simple_ja"
        - type: "preference"
          name: "easy"
    modelRefs:
      - model: "__FAST_MODEL__"
        use_reasoning: false
YAML

python - <<'PY'
import os, pathlib
p = pathlib.Path("config.yaml")
s = p.read_text(encoding="utf-8")
s = s.replace("__FAST_MODEL__", os.environ["FAST_MODEL"])
s = s.replace("__GPT_MODEL__", os.environ["GPT_MODEL"])
s = s.replace("__OLLAMA_ADDR__", os.environ["OLLAMA_ADDR"])
p.write_text(s, encoding="utf-8")
print("config.yaml updated")
PY
```

---

## 5) 起動（vLLM-SR）

```bash
vllm-sr serve --config config.yaml
```

vLLM-SR は起動後、Envoy 経由の OpenAI互換エンドポイント（`http://localhost:8801/v1/chat/completions`）を案内する想定です。 ([GitHub][4])

---

## 6) テスト（ルーティング確認）

### 6-1. まず models 一覧

```bash
# Router API 側（8080）で見えることが多い
curl -s http://127.0.0.1:8080/v1/models | head
```

### 6-2. Chat Completions（ポート 8801）

vLLM-SR の Envoy は `8801` ポートで待ち受けています。([GitHub][4])

**簡単（vLLM=FAST 側に寄せたい）**

```bash
curl -s http://127.0.0.1:8801/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model":"MoM",
    "messages":[{"role":"user","content":"こんにちは！短く返事して。"}]
  }' | head
```

**難しい/ツール（Ollama=GPT 側に寄せたい）**

```bash
curl -s http://127.0.0.1:8801/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model":"MoM",
    "messages":[{"role":"user","content":"ウェブで調べて根拠付きで比較し、実行手順も書いて。"}]
  }' | head
```

---

## 【注意点・例外】

* **Ollama の公式 docs で「Responses API（/v1/responses）をサポート」と書かれている箇所があり、OpenAI互換の範囲は変わる可能性**があります。([Ollama][8])
  一方で、Ollama は Chat Completions 互換（`/v1/chat/completions`）を提供する説明もあります。([Ollama][2])
  → もし `/v1/chat/completions` が 404 になる場合は、Ollama のバージョン差分が疑わしいので更新確認が必要です（“互換は実験的”の注意もあります）。([ollama.readthedocs.io][9])
* Docker / WSL / ネットワーク構成で **Windows を指すアドレスが変わる**ので、`curl http://<addr>:11434/v1/models` が通るアドレスを「正」として config に入れてください。([Ollama][1])


ig/router-defaults.yaml（preference 用の外部LLM設定）

preference は「外部LLMを使う」ので、ドキュメント例では router-defaults.yaml の external_models に設定します。

ここでは「判定だけは軽くしたい」発想で modelA を preference 判定役にしています。精度が足りないなら openai/gpt-oss-20b 側に変更してください（その分レイテンシは増えます）。

# config/router-defaults.yaml

external_models:
  - llm_provider: "vllm"
    model_role: "preference"
    llm_endpoint:
      address: "127.0.0.1"
      port: 8000
    llm_model_name: "modelA"
    llm_timeout_seconds: 10
    parser_type: "json"
    access_key: ""



from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:8801/v1",
    api_key="dummy",  # 認証を使わない構成ならダミーで可
)

# 1) 簡単な返答 → modelA に行かせたい
resp1 = client.chat.completions.create(
    model="MoM",
    messages=[{"role": "user", "content": "こんにちは。今日の調子は？短く答えて。"}],
)
print(resp1.choices[0].message.content)

# 2) Tool/エージェント寄り → GPT-oss-20B に行かせたい
resp2 = client.chat.completions.create(
    model="MoM",
    messages=[{"role": "user", "content": "APIを叩いて結果を集計する手順とPythonコードを作って。エラー時のリトライも。"}],
)
print(resp2.choices[0].message.content)




─ Worked for 21m 32s ───────────────────────────────────────────────────────────────────────────────────────

• 完了しました。Instructions1.md の内容をこちらの環境に合わせて実行し、テストまで確認しました。

  要点だけ共有します。

  - Python環境は uv で構築（.venv / pyproject.toml / uv.lock）
  - 疎通確認結果
      - Ollama: http://172.25.240.1:11434/v1/models で到達OK
      - vLLM: http://127.0.0.1:3849/v1/models で到達OK
  - RouterはDockerで動くため、vLLM向けアドレスを 172.17.0.1:3849 に調整
  - config.yaml を作成・修正し、uv run vllm-sr validate は成功
  テスト結果です。

  - http://127.0.0.1:8080/v1/models はOK（MoM を確認）
  - 簡単プロンプトは成功（LiquidAI/LFM2.5-1.2B-JP にルーティング）
  - 難しいプロンプトはルーティング自体は gpt-oss:20b へ到達確認できましたが、Ollama側で memory layout cannot
    be allocated が返りました（リソース要因）

  停止する場合は次です。

  UV_LINK_MODE=copy uv run vllm-sr stop

  必要なら、Ollama側のエラー回避（より軽いモデルに一時切替など）まで続けて対応します。