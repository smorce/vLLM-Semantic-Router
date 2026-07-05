## 起動コマンド
```
UV_LINK_MODE=copy uv run vllm-sr serve --config config.yaml
```
予め vLLM を起動しておく必要あり。

使っている LLM は
- vLLM: `LiquidAI/LFM2.5-1.2B-JP`
- Oollama: `gpt-oss:20b`

## 停止コマンド
```
UV_LINK_MODE=copy uv run vllm-sr stop
```




停止して再開するまではキャッシュが効いているみたいで毎回同じ回答になる。
ただ、本当にキャッシュを使っているのかまでは分からない。vllm-sr serve の仕様を調査してみないと分からない。
GPT OSS に時間がかかるので、キャッシュは使えてない気がする。
→ semantic-cache が OFF だったので ON にしたら
TEST: T04_gpt_api_code
TEST: T05_gpt_compare
が爆速で返ってくるようになったのでキャッシュの効き方もOK。