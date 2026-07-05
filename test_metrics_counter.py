#!/usr/bin/env python3
"""
vLLM-SR Metrics Counter Test

このテストは以下を検証します：
1. メトリクスエンドポイントへの接続
2. APIリクエスト前後のカウンタ値の取得
3. カウンタが正常に増加することの確認
"""

import requests
import time
import re
from typing import Optional


class MetricsTestError(Exception):
    """メトリクステスト固有のエラー"""
    pass


def get_metric_value(metrics_url: str, metric_name: str, model_name: str) -> Optional[int]:
    """
    Prometheusメトリクスから指定されたメトリクスの値を取得する
    
    Args:
        metrics_url: メトリクスエンドポイントのURL
        metric_name: メトリクス名（例: llm_model_requests_total）
        model_name: モデル名（例: LiquidAI/LFM2.5-1.2B-JP）
        
    Returns:
        メトリクスの値（整数）、見つからない場合はNone
    """
    try:
        response = requests.get(metrics_url, timeout=10)
        response.raise_for_status()
        
        # メトリクスのパターンを構築（モデル名のスラッシュをエスケープ）
        escaped_model = model_name.replace('/', '\\/')
        pattern = rf'{metric_name}\{{model="{escaped_model}"\}}\s+(\d+)'
        
        match = re.search(pattern, response.text)
        if match:
            return int(match.group(1))
        return None
        
    except requests.exceptions.RequestException as e:
        raise MetricsTestError(f"Failed to fetch metrics: {e}")


def send_chat_completion(api_url: str, message: str = "Test message") -> dict:
    """
    チャット補完APIにリクエストを送信する
    
    Args:
        api_url: APIエンドポイントのURL
        message: 送信するメッセージ
        
    Returns:
        APIレスポンス（JSON）
    """
    try:
        response = requests.post(
            api_url,
            json={
                "model": "MoM",
                "messages": [
                    {"role": "user", "content": message}
                ]
            },
            headers={"Content-Type": "application/json"},
            timeout=30
        )
        response.raise_for_status()
        return response.json()
        
    except requests.exceptions.RequestException as e:
        raise MetricsTestError(f"Failed to send chat completion request: {e}")


def test_metrics_counter_increment():
    """
    メトリクスカウンタの増加をテストする
    
    テスト手順:
    1. 初期のカウンタ値を取得
    2. APIリクエストを送信
    3. 更新後のカウンタ値を取得
    4. カウンタが正確に1増加していることを検証
    """
    # 設定
    METRICS_URL = "http://127.0.0.1:9190/metrics"
    API_URL = "http://127.0.0.1:8801/v1/chat/completions"
    METRIC_NAME = "llm_model_requests_total"
    MODEL_NAME = "LiquidAI/LFM2.5-1.2B-JP"
    
    print(f"=== vLLM-SR Metrics Counter Test ===\n")
    print(f"Metrics URL: {METRICS_URL}")
    print(f"API URL: {API_URL}")
    print(f"Metric: {METRIC_NAME}")
    print(f"Model: {MODEL_NAME}\n")
    
    # ステップ1: 初期カウンタ値を取得
    print("Step 1: Getting initial counter value...")
    initial_value = get_metric_value(METRICS_URL, METRIC_NAME, MODEL_NAME)
    
    if initial_value is None:
        raise MetricsTestError(
            f"Could not find metric '{METRIC_NAME}' for model '{MODEL_NAME}'. "
            "Please check if the service is running and the model name is correct."
        )
    
    print(f"✓ Initial counter value: {initial_value}\n")
    
    # ステップ2: APIリクエストを送信
    print("Step 2: Sending chat completion request...")
    response = send_chat_completion(API_URL, "Hello, this is a test message for metrics validation")
    
    # レスポンスの詳細を表示
    print(f"✓ Request sent successfully")
    if "choices" in response and len(response["choices"]) > 0:
        content = response["choices"][0]["message"]["content"]
        print(f"  Response: {content[:100]}{'...' if len(content) > 100 else ''}")
    print()
    
    # メトリクスが更新されるまで少し待つ
    print("Step 3: Waiting for metrics to update...")
    time.sleep(2)
    print("✓ Wait complete\n")
    
    # ステップ3: 更新後のカウンタ値を取得
    print("Step 4: Getting updated counter value...")
    updated_value = get_metric_value(METRICS_URL, METRIC_NAME, MODEL_NAME)
    
    if updated_value is None:
        raise MetricsTestError(
            f"Could not find metric '{METRIC_NAME}' after request. "
            "The metric may have been removed or the service restarted."
        )
    
    print(f"✓ Updated counter value: {updated_value}\n")
    
    # ステップ4: カウンタの増加を検証
    print("Step 5: Validating counter increment...")
    increment = updated_value - initial_value
    
    print(f"Counter increment: {increment}")
    print(f"Expected increment: 1\n")
    
    # アサーション
    assert increment == 1, (
        f"Counter increment mismatch! "
        f"Expected increment of 1, but got {increment} "
        f"(initial: {initial_value}, updated: {updated_value})"
    )
    
    print("=" * 50)
    print("✓ TEST PASSED")
    print("=" * 50)
    print(f"Metrics counter successfully incremented from {initial_value} to {updated_value}")


def main():
    """メイン実行関数"""
    try:
        test_metrics_counter_increment()
        return 0
    except MetricsTestError as e:
        print("\n" + "=" * 50)
        print("✗ TEST FAILED")
        print("=" * 50)
        print(f"Error: {e}")
        return 1
    except AssertionError as e:
        print("\n" + "=" * 50)
        print("✗ TEST FAILED")
        print("=" * 50)
        print(f"Assertion Error: {e}")
        return 1
    except Exception as e:
        print("\n" + "=" * 50)
        print("✗ TEST FAILED (Unexpected Error)")
        print("=" * 50)
        print(f"Unexpected error: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
