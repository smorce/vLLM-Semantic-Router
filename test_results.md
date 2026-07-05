# vLLM Semantic Router テスト結果

テスト実施日時: 2026-01-27

## テスト環境

- **エンドポイント**: `http://127.0.0.1:8801/v1/chat/completions`
- **ルーター**: vLLM Semantic Router (Envoy経由)
- **モデル構成**:
  - FAST モデル: `LiquidAI/LFM2.5-1.2B-JP` (vLLM, ポート 3849)
  - GPT モデル: `gpt-oss:20b` (Ollama, ポート 11434)

---

## テスト1: 簡単な挨拶（FASTモデルへのルーティング）

### リクエスト

```bash
curl -s http://127.0.0.1:8801/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model":"MoM",
    "messages":[{"role":"user","content":"こんにちは！短く返事して。"}]
  }'
```

### レスポンス（生データ）

```json
{
  "id": "chatcmpl-406fd03c-303e-4f5a-9487-d32c2e31f3fd",
  "object": "chat.completion",
  "created": 1769474754,
  "model": "LiquidAI/LFM2.5-1.2B-JP",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "こんにちは！短いお返事ありがとうございます。他に必要なことはありますか？😊",
        "refusal": null,
        "annotations": null,
        "audio": null,
        "function_call": null,
        "tool_calls": [],
        "reasoning": null,
        "reasoning_content": null
      },
      "logprobs": null,
      "finish_reason": "stop",
      "stop_reason": null,
      "token_ids": null
    }
  ],
  "service_tier": null,
  "system_fingerprint": null,
  "usage": {
    "prompt_tokens": 20,
    "total_tokens": 43,
    "completion_tokens": 23,
    "prompt_tokens_details": null
  },
  "prompt_logprobs": null,
  "prompt_token_ids": null,
  "kv_transfer_params": null
}
```

### 結果分析

- **ルーティング先**: `LiquidAI/LFM2.5-1.2B-JP` (FASTモデル) ✅
- **実行時間**: 約7.9秒
- **トークン使用量**:
  - プロンプト: 20トークン
  - 完了: 23トークン
  - 合計: 43トークン
- **マッチしたシグナル**: `simple_ja` (embedding signal)
- **適用されたデシジョン**: `default_to_fast` (priority: 1)
- **レスポンス内容**: 簡潔な挨拶の返答

**評価**: 簡単な挨拶が正しくFASTモデルにルーティングされました。

---

## テスト2: 複雑な要求（GPTモデルへのルーティング）

### リクエスト

```bash
curl -s http://127.0.0.1:8801/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model":"MoM",
    "messages":[{"role":"user","content":"ウェブで調べて根拠付きで比較し、実行手順も書いて。"}]
  }'
```

### レスポンス（生データ）

```json
{
  "id": "chatcmpl-955",
  "object": "chat.completion",
  "created": 1769474944,
  "model": "gpt-oss:20b",
  "system_fingerprint": "fp_ollama",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "申し訳ありませんが、具体的な比較対象や調査したい内容が明確でないため、お手伝いできる範囲が限定されてしまいます。  \n以下の点について教えていただけますか？\n\n1. **比較したい項目・テーマ**（例：クラウドサービス、食材、ソフトウェアの機能など）  \n2. **比較に重視したいポイント**（コスト、性能、信頼性、導入手順など）  \n3. **実行手順に関して**、どのようなプロセスや作業環境を想定しているか（例：Windows環境、Linuxサーバー、個人用PCなど）  \n\n提供いただいた情報をもとに、既存の知識から根拠付きの比較と実装手順をまとめてご案内いたします。なお、私には現在ウェブを直接検索する機能はありませんが、2024年までの知識を活用して回答いたします。",
        "reasoning": "The user says: \"ウェブで調べて根拠付きで比較し、実行手順も書いて。\" They want to search the web and provide evidence-based comparison and write step-by-step execution procedures. But this is a policy issue: The user is requesting the assistant to browse the web. The policy says \"I cannot browse.\" Also \"Use up to date information if user asks a question about a specific topic requiring up-to-date knowledge.\" However the user said \"ウェブで調べて\" which implies they want the assistant to browse the web. The policy states we can't actually browse. We need to refuse, but we might provide a best effort answer using knowledge up to 2024. The policy: We can comply, but we cannot browse internet. We can say we can't browse but can give answer. The policy says we should refuse the request to browse. \"I cannot browse.\" We need to politely refuse to browse the web, but we can provide a thorough explanation based on what we know. The user is basically asking for a comparative analysis with evidence from the web. We can't browse, but we can rely on our knowledge up to training. It's permissible to provide information without citing sources exactly, but we can mention general knowledge. Provide a comparative analysis and execution steps for whatever topic? Wait: They didn't specify a specific comparison. They just said \"ウェブで調べて根拠付きで比較し、実行手順も書いて\" without context. That is ambiguous. They might have omitted the context inadvertently. We cannot ask clarifying. The user didn't specify a topic. Possibly a generic request that the assistant should answer about something relevant. But policy says: \"We do not have enough context.\" There's no subject. We might refuse because we can't interpret. But maybe we can respond: \"I can't browse the web, but I can provide a comparison of X and Y based on my training.\" But we don't know what \"X\" and \"Y\" are.\n\nThus we should refuse due to insufficient context. The policy says if request is ambiguous we refuse. Let's check policy: \"If the request is unclear or insufficiently specified, the assistant should ask for clarification.\" The user request is too vague: they haven't stated what to compare. So we should ask for clarification, not refuse. The user said \"ウェブで調べて根拠付きで比較し、実行手順も書いて.\" We can clarify what they want to compare. So we should ask what topic or items to compare, what kind of execution steps they want. Also, we cannot browse. We can say we cannot browse but can base on knowledge.\n\nWe refuse to browse but can comply otherwise. Let's propose: we cannot browse, but we can provide help if we have context. Ask clarifying question. We should not mention policy. Just respond with something like: \"Unfortunately, I can't browse the web, but if you let me know what you'd like to compare and the execution instructions you need, I can provide a detailed comparison based on my knowledge up to now.\" That's the correct approach.\n\nThus final: Ask for clarification."
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 89,
    "completion_tokens": 880,
    "total_tokens": 969
  }
}
```

### 結果分析

- **ルーティング先**: `gpt-oss:20b` (GPTモデル) ✅
- **実行時間**: 約100.8秒（1分40秒）
- **トークン使用量**:
  - プロンプト: 89トークン
  - 完了: 880トークン
  - 合計: 969トークン
- **マッチしたキーワード**: 「調べて」「比較」「実行手順」
- **適用されたデシジョン**: `hard_or_tool_to_gpt` (priority: 200)
- **レスポンス内容**: 
  - 質問の曖昧さを指摘
  - 具体的な情報を要求（比較項目、重視ポイント、実行環境）
  - ウェブ検索機能がないことを説明
  - 2024年までの知識で対応可能と提示
- **reasoning フィールド**: GPTモデルの内部思考プロセスが記録されている

**評価**: 複雑な要求が正しくGPTモデルにルーティングされ、より高度な推論能力を持つモデルが適切に応答しました。

---

## まとめ

### ルーティング精度

| テストケース | 期待されるモデル | 実際のモデル | 結果 |
|------------|----------------|------------|------|
| 簡単な挨拶 | FASTモデル | LiquidAI/LFM2.5-1.2B-JP | ✅ 成功 |
| 複雑な要求 | GPTモデル | gpt-oss:20b | ✅ 成功 |

### パフォーマンス比較

| モデル | 実行時間 | トークン/秒 | 用途 |
|--------|---------|-----------|------|
| LiquidAI/LFM2.5-1.2B-JP | 7.9秒 | 約2.9 | 簡単な会話、雑談 |
| gpt-oss:20b | 100.8秒 | 約8.7 | 複雑な推論、ツール呼び出し |

### シグナル検出状況

**テスト1で検出されたシグナル:**
- `simple_ja` (embedding): 「こんにちは」「短く答えて」にマッチ

**テスト2で検出されたシグナル:**
- `tool_or_hard_keywords` (keyword): 「調べて」「比較」「実行手順」にマッチ
- `hard_or_tool_ja` (embedding): 複雑な要求パターンにマッチ（推定）

### 結論

vLLM Semantic Routerは、設定通りに動作し、クエリの複雑さに応じて適切なモデルにルーティングしています。

- **簡単なクエリ**: 高速・軽量なモデルで効率的に処理
- **複雑なクエリ**: 高性能なモデルで詳細な推論を実行

この構成により、コストとパフォーマンスのバランスが最適化されています。
