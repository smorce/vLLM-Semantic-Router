#!/usr/bin/env python3
"""
vLLM-SR Semantic Cache Metrics Test

このテストは以下を検証します：
1. 同一クエリを複数回送信して、キャッシュヒットが発生することを確認
2. メトリクスのキャッシュヒットカウンタが正常に増加することを確認
"""

import requests
import time
import re
from typing import Optional, Dict


class CacheMetricsTestError(Exception):
    """キャッシュメトリクステスト固有のエラー"""
    pass


def get_cache_metric_value(
    metrics_url: str,
    metric_name: str,
    decision_name: str = "default_to_fast",
    plugin_type: str = "semantic-cache"
) -> Optional[int]:
    """
    Prometheusメトリクスからキャッシュメトリクスの値を取得する
    
    Args:
        metrics_url: メトリクスエンドポイントのURL
        metric_name: メトリクス名（llm_cache_plugin_hits_total など）
        decision_name: デシジョン名
        plugin_type: プラグインタイプ
        
    Returns:
        メトリクスの値（整数）、見つからない場合はNone
    """
    try:
        response = requests.get(metrics_url, timeout=10)
        response.raise_for_status()
        
        # メトリクスのパターンを構築
        pattern = (
            rf'{metric_name}\{{decision_name="{decision_name}",'
            rf'plugin_type="{plugin_type}"\}}\s+(\d+)'
        )
        
        match = re.search(pattern, response.text)
        if match:
            return int(match.group(1))
        return None
        
    except requests.exceptions.RequestException as e:
        raise CacheMetricsTestError(f"Failed to fetch metrics: {e}")


def get_cache_operation_metric(
    metrics_url: str,
    operation: str,
    status: str
) -> Optional[int]:
    """
    キャッシュ操作のメトリクスを取得する
    
    Args:
        metrics_url: メトリクスエンドポイントのURL
        operation: 操作名（find_similar, add_pending など）
        status: ステータス（hit, miss, success）
        
    Returns:
        メトリクスの値（整数）、見つからない場合はNone
    """
    try:
        response = requests.get(metrics_url, timeout=10)
        response.raise_for_status()
        
        pattern = (
            rf'llm_cache_operations_total\{{backend="memory",'
            rf'operation="{operation}",status="{status}"\}}\s+(\d+)'
        )
        
        match = re.search(pattern, response.text)
        if match:
            return int(match.group(1))
        return None
        
    except requests.exceptions.RequestException as e:
        raise CacheMetricsTestError(f"Failed to fetch metrics: {e}")


def send_chat_completion(api_url: str, message: str = "Test message") -> Dict:
    """
    チャット補完APIにリクエストを送信する
    
    Args:
        api_url: APIエンドポイントのURL
        message: 送信するメッセージ
        
    Returns:
        APIレスポンス（JSON）とヘッダー情報
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
        
        return {
            "json": response.json(),
            "headers": dict(response.headers),
            "status_code": response.status_code
        }
        
    except requests.exceptions.RequestException as e:
        raise CacheMetricsTestError(f"Failed to send chat completion request: {e}")


def test_cache_hit_metrics():
    """
    セマンティックキャッシュのヒットメトリクスをテストする
    
    テスト手順:
    1. 初期のキャッシュヒットカウンタを取得
    2. 新しいクエリを送信（キャッシュミスになるはず）
    3. 同じクエリを再度送信（キャッシュヒットになるはず）
    4. キャッシュヒットカウンタが増加していることを検証
    """
    # 設定
    METRICS_URL = "http://127.0.0.1:9190/metrics"
    API_URL = "http://127.0.0.1:8801/v1/chat/completions"
    DECISION_NAME = "default_to_fast"
    PLUGIN_TYPE = "semantic-cache"
    
    # タイムスタンプを含む一意のメッセージを生成
    test_message = f"こんにちは、元気ですか？今日の日付は {time.time()}"
    
    print(f"=== vLLM-SR Semantic Cache Metrics Test ===\n")
    print(f"Metrics URL: {METRICS_URL}")
    print(f"API URL: {API_URL}")
    print(f"Decision: {DECISION_NAME}")
    print(f"Plugin: {PLUGIN_TYPE}")
    print(f"Test Message: {test_message}\n")
    
    # ステップ1: 初期メトリクスを取得
    print("Step 1: Getting initial metrics...")
    
    initial_hits = get_cache_metric_value(
        METRICS_URL, "llm_cache_plugin_hits_total", DECISION_NAME, PLUGIN_TYPE
    )
    initial_misses = get_cache_metric_value(
        METRICS_URL, "llm_cache_plugin_misses_total", DECISION_NAME, PLUGIN_TYPE
    )
    initial_find_hits = get_cache_operation_metric(METRICS_URL, "find_similar", "hit")
    initial_find_misses = get_cache_operation_metric(METRICS_URL, "find_similar", "miss")
    
    print(f"✓ Initial plugin hits: {initial_hits}")
    print(f"✓ Initial plugin misses: {initial_misses}")
    print(f"✓ Initial find_similar hits: {initial_find_hits}")
    print(f"✓ Initial find_similar misses: {initial_find_misses}\n")
    
    # ステップ2: 最初のリクエスト（キャッシュミスになるはず）
    print("Step 2: Sending first request (expect cache miss)...")
    response1 = send_chat_completion(API_URL, test_message)
    
    print(f"✓ First request sent")
    print(f"  Status: {response1['status_code']}")
    print(f"  Decision: {response1['headers'].get('x-vsr-selected-decision', 'N/A')}")
    print(f"  Model: {response1['headers'].get('x-vsr-selected-model', 'N/A')}")
    
    if "choices" in response1["json"] and len(response1["json"]["choices"]) > 0:
        content = response1["json"]["choices"][0]["message"]["content"]
        print(f"  Response: {content[:80]}{'...' if len(content) > 80 else ''}")
    print()
    
    # メトリクスが更新されるまで待つ
    time.sleep(2)
    
    # ステップ3: 同じリクエストを再送信（キャッシュヒットになるはず）
    print("Step 3: Sending second request with same message (expect cache hit)...")
    response2 = send_chat_completion(API_URL, test_message)
    
    print(f"✓ Second request sent")
    print(f"  Status: {response2['status_code']}")
    print(f"  Decision: {response2['headers'].get('x-vsr-selected-decision', 'N/A')}")
    
    if "choices" in response2["json"] and len(response2["json"]["choices"]) > 0:
        content = response2["json"]["choices"][0]["message"]["content"]
        print(f"  Response: {content[:80]}{'...' if len(content) > 80 else ''}")
    print()
    
    # メトリクスが更新されるまで待つ
    time.sleep(2)
    
    # ステップ4: 更新後のメトリクスを取得
    print("Step 4: Getting updated metrics...")
    
    updated_hits = get_cache_metric_value(
        METRICS_URL, "llm_cache_plugin_hits_total", DECISION_NAME, PLUGIN_TYPE
    )
    updated_misses = get_cache_metric_value(
        METRICS_URL, "llm_cache_plugin_misses_total", DECISION_NAME, PLUGIN_TYPE
    )
    updated_find_hits = get_cache_operation_metric(METRICS_URL, "find_similar", "hit")
    updated_find_misses = get_cache_operation_metric(METRICS_URL, "find_similar", "miss")
    
    print(f"✓ Updated plugin hits: {updated_hits}")
    print(f"✓ Updated plugin misses: {updated_misses}")
    print(f"✓ Updated find_similar hits: {updated_find_hits}")
    print(f"✓ Updated find_similar misses: {updated_find_misses}\n")
    
    # ステップ5: メトリクスの変化を検証
    print("Step 5: Validating metrics changes...")
    
    hits_increment = updated_hits - initial_hits if (updated_hits and initial_hits) else 0
    misses_increment = updated_misses - initial_misses if (updated_misses and initial_misses) else 0
    find_hits_increment = updated_find_hits - initial_find_hits if (updated_find_hits and initial_find_hits) else 0
    find_misses_increment = updated_find_misses - initial_find_misses if (updated_find_misses and initial_find_misses) else 0
    
    print(f"Plugin hits increment: {hits_increment}")
    print(f"Plugin misses increment: {misses_increment}")
    print(f"Find_similar hits increment: {find_hits_increment}")
    print(f"Find_similar misses increment: {find_misses_increment}\n")
    
    # アサーション: 2回目のリクエストでキャッシュヒットが発生しているはず
    print("Validating assertions...")
    
    # 最低でも1回のキャッシュヒットが発生しているはずです
    # （1回目がミス、2回目がヒット）
    assert hits_increment >= 1, (
        f"Expected at least 1 cache hit, but got {hits_increment}. "
        f"Cache may not be working properly."
    )
    print(f"✓ Assertion passed: Cache hits increased by {hits_increment}")
    
    # find_similarのヒットも増加しているはず
    assert find_hits_increment >= 1, (
        f"Expected at least 1 find_similar hit, but got {find_hits_increment}. "
        f"Cache lookup may not be working properly."
    )
    print(f"✓ Assertion passed: Find_similar hits increased by {find_hits_increment}")
    
    print("\n" + "=" * 60)
    print("✓ TEST PASSED")
    print("=" * 60)
    print(f"Cache successfully detected similar queries and returned cached responses")
    print(f"Cache hit rate improvement: +{hits_increment} hits")


def main():
    """メイン実行関数"""
    try:
        test_cache_hit_metrics()
        return 0
    except CacheMetricsTestError as e:
        print("\n" + "=" * 60)
        print("✗ TEST FAILED")
        print("=" * 60)
        print(f"Error: {e}")
        return 1
    except AssertionError as e:
        print("\n" + "=" * 60)
        print("✗ TEST FAILED")
        print("=" * 60)
        print(f"Assertion Error: {e}")
        return 1
    except Exception as e:
        print("\n" + "=" * 60)
        print("✗ TEST FAILED (Unexpected Error)")
        print("=" * 60)
        print(f"Unexpected error: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
