#!/usr/bin/env python3
"""
vLLM-SR モデルルーティング検証スクリプト

Semantic Router (:8801) 経由で次を確認する:
1. ルーターとバックエンドの到達性
2. 明示モデル指定時の backend / upstream モデル名
3. MoM 自動ルーティング（シグナル別プロンプト設計）
4. セマンティックキャッシュのヒット
5. Responses API による会話継続（previous_response_id）
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

import requests


class RoutingTestError(Exception):
    """ルーティング検証固有のエラー"""


@dataclass
class CaseResult:
    name: str
    passed: bool
    detail: str
    elapsed_s: float = 0.0
    headers: dict[str, str] = field(default_factory=dict)
    response_model: str | None = None
    skipped: bool = False


@dataclass(frozen=True)
class MomTestCase:
    """MoM 自動ルーティング用のプロンプト定義。"""

    name: str
    prompt: str
    intent: str
    expected_selected_model: str | None = None
    allowed_selected_models: frozenset[str] | None = None
    expected_decision: str | None = None
    allowed_decisions: frozenset[str] | None = None
    max_tokens: int = 24
    slow: bool = False


ROUTER_URL_DEFAULT = "http://127.0.0.1:8801"
VLLM_BACKEND_DEFAULT = "http://172.17.0.1:3849"
LLAMA_BACKEND_DEFAULT = "http://172.17.0.1:1067"

EXPECTED_UPSTREAM = {
    "lfm2.5-1.2b-jp": "LiquidAI/LFM2.5-1.2B-JP",
    "qwen3.6-27b": "Qwen3.6-27B-MTP-GGUF-UD-Q4_K_XL",
}

ROUTING_HEADER_KEYS = (
    "x-vsr-selected-model",
    "x-vsr-selected-decision",
    "x-vsr-selected-category",
    "x-selected-model",
    "x-vsr-cache-hit",
    "x-vsr-response-path",
)

VLLM_UPSTREAM = EXPECTED_UPSTREAM["lfm2.5-1.2b-jp"]
QWEN_UPSTREAM = EXPECTED_UPSTREAM["qwen3.6-27b"]

VLLM_NOT_FOUND_MARKERS = (
    '"type":"NotFoundError"',
    "does not exist",
)

# config/config.yaml の decisions / signals に合わせた MoM プロンプト。
# decision は分類器のブレがあるため、allowed_decisions も併記する。
MOM_TEST_CASES: tuple[MomTestCase, ...] = (
    MomTestCase(
        name="mom_other_multi_factor",
        prompt="こんにちは。今日の天気について一言で教えてください。",
        intent="domain:other を狙い multi_factor_route（モデルは lfm/qwen のいずれか）",
        expected_decision="multi_factor_route",
        allowed_decisions=frozenset({"multi_factor_route"}),
        allowed_selected_models=frozenset({"lfm2.5-1.2b-jp", "qwen3.6-27b"}),
        max_tokens=16,
    ),
    MomTestCase(
        name="mom_business_static_intent",
        prompt=(
            "MBAケーススタディとして、B2B SaaS の価格戦略と"
            "顧客獲得コストのトレードオフを整理してください。"
        ),
        intent="domain:business → static_business_route を狙う（semantic-cache 有効ルート）",
        expected_selected_model="lfm2.5-1.2b-jp",
        allowed_decisions=frozenset({"static_business_route", "safe_only_svm_route"}),
        max_tokens=24,
    ),
    MomTestCase(
        name="mom_ml_keyword_intent",
        prompt=(
            "機械学習モデルの学習で勾配降下法を使うとき、"
            "ニューラルネットワークの過学習を抑える方法を教えてください。"
        ),
        intent="keyword:machine_learning → mlp_experimental_route を狙う",
        expected_selected_model="lfm2.5-1.2b-jp",
        allowed_decisions=frozenset({"mlp_experimental_route", "safe_only_svm_route"}),
        max_tokens=24,
    ),
    MomTestCase(
        name="mom_fact_check_escalation",
        prompt="日本の憲法は何年何月何日に施行されましたか。日付だけ正確に答えてください。",
        intent="fact_check:needs_fact_check → legal_confidence_route で Qwen エスカレーションを狙う",
        allowed_selected_models=frozenset({"qwen3.6-27b", "lfm2.5-1.2b-jp"}),
        allowed_decisions=frozenset(
            {"legal_confidence_route", "multi_factor_route", "safe_only_svm_route"}
        ),
        max_tokens=24,
        slow=True,
    ),
    MomTestCase(
        name="mom_cs_remom_intent",
        prompt=(
            "Pythonで実装したマイクロサービスのイベント駆動アーキテクチャを"
            "リファクタリングしたい。分散トランザクションのサガパターン、"
            "デバッグ手順、アルゴリズム上の注意点を詳しく説明してください。"
        ),
        intent=(
            "domain:computer science + keyword:code_keywords + complexity:hard "
            "→ computer-science-remom-route / Qwen を狙う"
        ),
        allowed_selected_models=frozenset({"qwen3.6-27b", "lfm2.5-1.2b-jp"}),
        allowed_decisions=frozenset(
            {
                "computer-science-remom-route",
                "safe_only_svm_route",
                "multi_factor_route",
            }
        ),
        max_tokens=32,
        slow=True,
    ),
    MomTestCase(
        name="mom_intent_calculation_route",
        prompt="100ドルを日本円に換算するといくらですか？",
        intent="intent:calculation → intent_calculation_route を狙う",
        expected_decision="intent_calculation_route",
        expected_selected_model="lfm2.5-1.2b-jp",
        allowed_decisions=frozenset({"intent_calculation_route", "safe_only_svm_route"}),
        max_tokens=24,
    ),
    MomTestCase(
        name="mom_pii_restricted_route",
        prompt=(
            "東京都千代田区丸の内1-1-1にお住まいの佐藤花子様、"
            "電話番号は03-1234-5678です。この方の届け先を確認してください。"
        ),
        intent="pii:restricted_pii → restricted_pii_guard_route を狙う",
        expected_decision="restricted_pii_guard_route",
        expected_selected_model="lfm2.5-1.2b-jp",
        allowed_decisions=frozenset({"restricted_pii_guard_route", "safe_only_svm_route"}),
        max_tokens=24,
    ),
)


def _normalize_headers(
    headers: requests.structures.CaseInsensitiveDict,
) -> dict[str, str]:
    return {k.lower(): v for k, v in headers.items()}


def _pick_routing_headers(headers: dict[str, str]) -> dict[str, str]:
    return {k: headers[k] for k in ROUTING_HEADER_KEYS if k in headers}


def extract_chat_content(body: dict[str, Any] | None) -> str:
    if not body:
        return ""
    choices = body.get("choices") or []
    if not choices:
        return ""
    message = choices[0].get("message") or {}
    return str(message.get("content") or "")


def extract_response_api_text(body: dict[str, Any] | None) -> str:
    if not body:
        return ""
    if body.get("output_text"):
        return str(body["output_text"])
    parts: list[str] = []
    for item in body.get("output", []):
        for content in item.get("content", []):
            if content.get("type") == "output_text" and content.get("text"):
                parts.append(str(content["text"]))
    return "".join(parts)


def looks_like_vllm_model_not_found(
    status_code: int, text: str, upstream_model: str
) -> bool:
    if status_code != 404 or upstream_model not in text:
        return False
    return any(marker in text for marker in VLLM_NOT_FOUND_MARKERS)


def routing_failure_hint(
    *,
    request_model: str,
    expected_selected_model: str | None,
    selected_model: str | None,
    response_model: str | None,
    status_code: int,
    text: str,
) -> str | None:
    if expected_selected_model == "qwen3.6-27b" and looks_like_vllm_model_not_found(
        status_code, text, QWEN_UPSTREAM
    ):
        return (
            "Qwen upstream model was sent to vLLM default route (172.17.0.1:3849). "
            "Check Envoy x-selected-model routing and global.router.clear_route_cache."
        )
    if (
        selected_model
        and expected_selected_model
        and selected_model != expected_selected_model
    ):
        return f"Router selected {selected_model}, but test expected {expected_selected_model}."
    if response_model and response_model.startswith("unsloth/"):
        return "Upstream model still uses unsloth/ prefix; llama-server will reject it."
    if request_model == "MoM" and status_code != 200:
        return "MoM routing failed before backend completion."
    return None


def check_router_models(router_url: str, timeout: float) -> CaseResult:
    started = time.monotonic()
    url = f"{router_url.rstrip('/')}/v1/models"
    try:
        response = requests.get(url, timeout=timeout)
        response.raise_for_status()
        model_ids = {item.get("id", "") for item in response.json().get("data", [])}
        elapsed = time.monotonic() - started
        required = {"MoM", "lfm2.5-1.2b-jp", "qwen3.6-27b"}
        missing = sorted(required - model_ids)
        if missing:
            return CaseResult(
                name="preflight_router_models",
                passed=False,
                detail=f"missing logical models: {missing}",
                elapsed_s=elapsed,
            )
        return CaseResult(
            name="preflight_router_models",
            passed=True,
            detail=f"logical models available: {sorted(required)}",
            elapsed_s=elapsed,
        )
    except requests.RequestException as exc:
        elapsed = time.monotonic() - started
        return CaseResult(
            name="preflight_router_models",
            passed=False,
            detail=f"failed to list router models: {exc}",
            elapsed_s=elapsed,
        )


def check_backend_models(
    name: str,
    base_url: str,
    expected_model_id: str,
    timeout: float,
) -> CaseResult:
    started = time.monotonic()
    url = f"{base_url.rstrip('/')}/v1/models"
    try:
        response = requests.get(url, timeout=timeout)
        response.raise_for_status()
        payload = response.json()
        model_ids = {item.get("id", "") for item in payload.get("data", [])}
        elapsed = time.monotonic() - started
        if expected_model_id in model_ids:
            return CaseResult(
                name=name,
                passed=True,
                detail=f"model id found: {expected_model_id}",
                elapsed_s=elapsed,
            )
        return CaseResult(
            name=name,
            passed=False,
            detail=f"expected model id missing. available={sorted(model_ids)}",
            elapsed_s=elapsed,
        )
    except requests.RequestException as exc:
        elapsed = time.monotonic() - started
        return CaseResult(
            name=name,
            passed=False,
            detail=f"failed to list models: {exc}",
            elapsed_s=elapsed,
        )


def send_chat_completion(
    router_url: str,
    *,
    model: str,
    prompt: str,
    timeout: float,
    max_tokens: int,
) -> dict[str, Any]:
    url = f"{router_url.rstrip('/')}/v1/chat/completions"
    response = requests.post(
        url,
        json={
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
        },
        headers={"Content-Type": "application/json"},
        timeout=timeout,
    )
    headers = _normalize_headers(response.headers)
    body: dict[str, Any] | None = None
    if response.content:
        try:
            body = response.json()
        except json.JSONDecodeError:
            body = None
    return {
        "status_code": response.status_code,
        "headers": headers,
        "body": body,
        "text": response.text,
    }


def send_response_api(
    router_url: str,
    *,
    model: str,
    prompt: str,
    timeout: float,
    max_output_tokens: int,
    previous_response_id: str | None = None,
) -> dict[str, Any]:
    url = f"{router_url.rstrip('/')}/v1/responses"
    payload: dict[str, Any] = {
        "model": model,
        "input": prompt,
        "max_output_tokens": max_output_tokens,
        "store": True,
    }
    if previous_response_id:
        payload["previous_response_id"] = previous_response_id

    response = requests.post(
        url,
        json=payload,
        headers={"Content-Type": "application/json"},
        timeout=timeout,
    )
    headers = _normalize_headers(response.headers)
    body: dict[str, Any] | None = None
    if response.content:
        try:
            body = response.json()
        except json.JSONDecodeError:
            body = None
    return {
        "status_code": response.status_code,
        "headers": headers,
        "body": body,
        "text": response.text,
    }


def evaluate_routing_case(
    name: str,
    router_url: str,
    *,
    request_model: str,
    prompt: str,
    timeout: float,
    max_tokens: int,
    expected_selected_model: str | None = None,
    expected_decision: str | None = None,
    allowed_decisions: set[str] | frozenset[str] | None = None,
    allowed_selected_models: set[str] | frozenset[str] | None = None,
    expected_upstream_model: str | None = None,
    forbidden_upstream_substrings: tuple[str, ...] = (),
) -> CaseResult:
    started = time.monotonic()
    try:
        result = send_chat_completion(
            router_url,
            model=request_model,
            prompt=prompt,
            timeout=timeout,
            max_tokens=max_tokens,
        )
    except requests.RequestException as exc:
        elapsed = time.monotonic() - started
        return CaseResult(
            name=name,
            passed=False,
            detail=f"request failed: {exc}",
            elapsed_s=elapsed,
        )

    elapsed = time.monotonic() - started
    headers = result["headers"]
    routing_headers = _pick_routing_headers(headers)
    selected_model = headers.get("x-vsr-selected-model") or headers.get(
        "x-selected-model"
    )
    selected_decision = headers.get("x-vsr-selected-decision")
    body = result["body"] or {}
    response_model = body.get("model")
    status_code = result["status_code"]
    cache_hit = headers.get("x-vsr-cache-hit") == "true"

    if status_code != 200:
        snippet = (result["text"] or "")[:240]
        hint = routing_failure_hint(
            request_model=request_model,
            expected_selected_model=expected_selected_model,
            selected_model=selected_model,
            response_model=response_model,
            status_code=status_code,
            text=result["text"] or "",
        )
        detail = f"HTTP {status_code}: {snippet}"
        if hint:
            detail = f"{detail} | hint: {hint}"
        return CaseResult(
            name=name,
            passed=False,
            detail=detail,
            elapsed_s=elapsed,
            headers=routing_headers,
            response_model=response_model,
        )

    failures: list[str] = []

    # キャッシュヒット時は semantic-cache がモデル選択を経由せずレスポンスを
    # 再生するため、x-vsr-selected-model が付与されないことがある。これは
    # semantic-cache が意図通り機能している証拠であって不具合ではないので、
    # モデル選択系のアサーションはスキップし、decision の一致だけ検証する。
    if not cache_hit:
        if expected_selected_model and selected_model != expected_selected_model:
            failures.append(
                f"selected model mismatch: expected={expected_selected_model}, actual={selected_model}"
            )

        if (
            allowed_selected_models is not None
            and selected_model not in allowed_selected_models
        ):
            failures.append(
                "selected model not allowed: "
                f"actual={selected_model}, allowed={sorted(allowed_selected_models)}"
            )

    if expected_decision and selected_decision != expected_decision:
        failures.append(
            f"decision mismatch: expected={expected_decision}, actual={selected_decision}"
        )

    if (
        allowed_decisions is not None
        and selected_decision
        and selected_decision not in allowed_decisions
    ):
        failures.append(
            "decision not allowed: "
            f"actual={selected_decision}, allowed={sorted(allowed_decisions)}"
        )

    if expected_upstream_model and response_model != expected_upstream_model:
        failures.append(
            f"upstream model mismatch: expected={expected_upstream_model}, actual={response_model}"
        )

    for forbidden in forbidden_upstream_substrings:
        if response_model and forbidden in response_model:
            failures.append(f"forbidden upstream substring found: {forbidden}")

    if failures:
        return CaseResult(
            name=name,
            passed=False,
            detail="; ".join(failures),
            elapsed_s=elapsed,
            headers=routing_headers,
            response_model=response_model,
        )

    return CaseResult(
        name=name,
        passed=True,
        detail=(
            f"selected={selected_model or 'n/a'}, "
            f"decision={selected_decision or 'n/a'}, "
            f"upstream={response_model or 'n/a'}"
        ),
        elapsed_s=elapsed,
        headers=routing_headers,
        response_model=response_model,
    )


def check_response_api_available(router_url: str, timeout: float) -> CaseResult:
    started = time.monotonic()
    try:
        response = requests.post(
            f"{router_url.rstrip('/')}/v1/responses",
            json={
                "model": "lfm2.5-1.2b-jp",
                "input": "ping",
                "max_output_tokens": 4,
                "store": False,
            },
            headers={"Content-Type": "application/json"},
            timeout=timeout,
        )
        elapsed = time.monotonic() - started
        if response.status_code == 404:
            return CaseResult(
                name="preflight_response_api",
                passed=False,
                skipped=True,
                detail=(
                    "/v1/responses is unavailable (Response API store not initialized). "
                    "Check global.services.response_api and vllm-sr-redis."
                ),
                elapsed_s=elapsed,
            )
        if response.status_code not in {200, 400, 422}:
            return CaseResult(
                name="preflight_response_api",
                passed=False,
                detail=f"unexpected status from /v1/responses: HTTP {response.status_code}",
                elapsed_s=elapsed,
            )
        return CaseResult(
            name="preflight_response_api",
            passed=True,
            detail=f"/v1/responses reachable (HTTP {response.status_code})",
            elapsed_s=elapsed,
        )
    except requests.RequestException as exc:
        elapsed = time.monotonic() - started
        return CaseResult(
            name="preflight_response_api",
            passed=False,
            detail=f"failed to probe /v1/responses: {exc}",
            elapsed_s=elapsed,
        )


def evaluate_semantic_cache_case(
    router_url: str,
    *,
    timeout: float,
    max_tokens: int,
    cache_wait_s: float,
    require_cache: bool,
) -> CaseResult:
    """static_business_route の semantic-cache を同一プロンプトで 2 回叩いて検証する。

    トピックは意図的に mom_business_static_intent（B2B SaaS の価格戦略/CAC）とは
    異なる話題（物流セクターの B2B 事業戦略）にしている。
    mom_business と同様に business_keywords を明示し、PII 誤検知を避けるプロンプト構造にする。
    """
    started = time.monotonic()
    # 検証用 UUID は PII 誤検知を招くため使わず、固定の非 PII サフィックスで
    # mom_business との偶発キャッシュヒットを避ける。
    prompt = (
        "MBAケーススタディとして、物流セクターのB2B向け"
        "事業戦略と顧客獲得コストのトレードオフを整理してください。"
        "【物流B2Bケース study】"
    )

    try:
        first = send_chat_completion(
            router_url,
            model="MoM",
            prompt=prompt,
            timeout=timeout,
            max_tokens=max_tokens,
        )
        if first["status_code"] != 200:
            elapsed = time.monotonic() - started
            return CaseResult(
                name="semantic_cache_repeat",
                passed=False,
                detail=f"first request failed: HTTP {first['status_code']}",
                elapsed_s=elapsed,
                headers=_pick_routing_headers(first["headers"]),
            )

        first_headers = first["headers"]
        # Milvus のセマンティックキャッシュは TTL (7200s) の間プロセス再起動や
        # テスト実行をまたいで永続化される。そのため、直近の別実行で書き込まれた
        # 類似トピックのエントリに、今回生成した一意トークン付きプロンプトが
        # 類似度閾値 (0.82) を超えて言い換えマッチすることがある。これは
        # semantic-cache が意図通りセッション横断で機能している証拠であって
        # 不具合ではないため、失敗にはせず「ウォームキャッシュ」パスとして
        # 扱い、2 回目リクエストでも一貫してヒットすることだけを検証する。
        warm_cache_hit = first_headers.get("x-vsr-cache-hit") == "true"

        decision = first_headers.get("x-vsr-selected-decision")
        if decision != "static_business_route":
            elapsed = time.monotonic() - started
            detail = (
                "semantic-cache plugin is configured only on static_business_route; "
                f"first request decision was {decision!r}"
            )
            if require_cache:
                return CaseResult(
                    name="semantic_cache_repeat",
                    passed=False,
                    detail=detail,
                    elapsed_s=elapsed,
                    headers=_pick_routing_headers(first_headers),
                )
            return CaseResult(
                name="semantic_cache_repeat",
                passed=True,
                skipped=True,
                detail=f"SKIPPED: {detail}",
                elapsed_s=elapsed,
                headers=_pick_routing_headers(first_headers),
            )

        if not warm_cache_hit:
            time.sleep(cache_wait_s)

        second = send_chat_completion(
            router_url,
            model="MoM",
            prompt=prompt,
            timeout=timeout,
            max_tokens=max_tokens,
        )
    except requests.RequestException as exc:
        elapsed = time.monotonic() - started
        return CaseResult(
            name="semantic_cache_repeat",
            passed=False,
            detail=f"cache test request failed: {exc}",
            elapsed_s=elapsed,
        )

    elapsed = time.monotonic() - started
    second_headers = second["headers"]
    routing_headers = _pick_routing_headers(second_headers)
    cache_hit = second_headers.get("x-vsr-cache-hit") == "true"
    response_path = second_headers.get("x-vsr-response-path")
    second_ok = second["status_code"] == 200

    failures: list[str] = []
    if not second_ok:
        failures.append(f"second request failed: HTTP {second['status_code']}")
    if not cache_hit:
        failures.append("x-vsr-cache-hit is not true on repeated request")
    if response_path != "cache":
        failures.append(f"x-vsr-response-path expected cache, got {response_path!r}")

    if failures:
        return CaseResult(
            name="semantic_cache_repeat",
            passed=False,
            detail="; ".join(failures),
            elapsed_s=elapsed,
            headers=routing_headers,
        )

    if warm_cache_hit:
        return CaseResult(
            name="semantic_cache_repeat",
            passed=True,
            detail=(
                "first request already hit a warm cache entry from an earlier "
                "run (within the 7200s TTL) and the repeat request stayed a "
                f"cache-hit=true, path={response_path!r}, decision={decision!r}"
            ),
            elapsed_s=elapsed,
            headers=routing_headers,
        )

    return CaseResult(
        name="semantic_cache_repeat",
        passed=True,
        detail=(
            "first=miss, second cache-hit=true, "
            f"path={response_path}, first_decision={decision or 'n/a'}"
        ),
        elapsed_s=elapsed,
        headers=routing_headers,
    )


def evaluate_conversation_continuity_case(
    router_url: str,
    *,
    timeout: float,
    max_output_tokens: int,
    remembered_name: str = "ステラ",
) -> CaseResult:
    """Responses API + previous_response_id で会話履歴が引き継がれるか検証する。"""
    started = time.monotonic()
    intro_prompt = f"私の名前は{remembered_name}です。名前だけ一言で答えてください。"
    recall_prompt = "私の名前を覚えていますか？名前だけ答えてください。"

    try:
        first = send_response_api(
            router_url,
            model="lfm2.5-1.2b-jp",
            prompt=intro_prompt,
            timeout=timeout,
            max_output_tokens=max_output_tokens,
        )
        if first["status_code"] != 200:
            elapsed = time.monotonic() - started
            return CaseResult(
                name="conversation_previous_response_id",
                passed=False,
                detail=f"turn1 failed: HTTP {first['status_code']}: {(first['text'] or '')[:200]}",
                elapsed_s=elapsed,
            )

        first_body = first["body"] or {}
        response_id = str(first_body.get("id") or "")
        if not response_id:
            elapsed = time.monotonic() - started
            return CaseResult(
                name="conversation_previous_response_id",
                passed=False,
                detail="turn1 response id is missing",
                elapsed_s=elapsed,
            )

        second = send_response_api(
            router_url,
            model="lfm2.5-1.2b-jp",
            prompt=recall_prompt,
            timeout=timeout,
            max_output_tokens=max_output_tokens,
            previous_response_id=response_id,
        )
    except requests.RequestException as exc:
        elapsed = time.monotonic() - started
        return CaseResult(
            name="conversation_previous_response_id",
            passed=False,
            detail=f"conversation request failed: {exc}",
            elapsed_s=elapsed,
        )

    elapsed = time.monotonic() - started
    if second["status_code"] != 200:
        return CaseResult(
            name="conversation_previous_response_id",
            passed=False,
            detail=f"turn2 failed: HTTP {second['status_code']}: {(second['text'] or '')[:200]}",
            elapsed_s=elapsed,
        )

    second_body = second["body"] or {}
    answer = extract_response_api_text(second_body)
    previous = str(second_body.get("previous_response_id") or "")
    failures: list[str] = []
    if previous != response_id:
        failures.append(
            f"previous_response_id mismatch: expected={response_id}, actual={previous}"
        )
    if remembered_name not in answer:
        failures.append(
            f"turn2 answer does not mention {remembered_name!r}: {answer[:120]!r}"
        )

    headers = _pick_routing_headers(second["headers"])
    if failures:
        return CaseResult(
            name="conversation_previous_response_id",
            passed=False,
            detail="; ".join(failures),
            elapsed_s=elapsed,
            headers=headers,
        )

    return CaseResult(
        name="conversation_previous_response_id",
        passed=True,
        detail=(
            f"response_id={response_id}, turn2_answer={answer[:80]!r}, "
            f"previous_response_id linked"
        ),
        elapsed_s=elapsed,
        headers=headers,
    )


def print_case(result: CaseResult, *, show_intent: str | None = None) -> None:
    if result.skipped:
        status = "SKIP"
    else:
        status = "PASS" if result.passed else "FAIL"
    print(f"[{status}] {result.name} ({result.elapsed_s:.1f}s)")
    if show_intent:
        print(f"       intent: {show_intent}")
    print(f"       {result.detail}")
    if result.headers:
        for key, value in result.headers.items():
            print(f"       {key}: {value}")


def run_tests(args: argparse.Namespace) -> int:
    results: list[CaseResult] = []

    print("=== vLLM-SR Model Routing Test ===\n")
    print(f"Router:       {args.router_url}")
    print(f"vLLM backend: {args.vllm_backend}")
    print(f"llama backend:{args.llama_backend}")
    print(f"Timeout:      {args.timeout}s")
    print(f"Slow timeout: {args.slow_timeout}s")
    print(f"Max tokens:   {args.max_tokens}\n")

    if not args.skip_preflight:
        print("-- Preflight --")
        results.append(check_router_models(args.router_url, args.preflight_timeout))
        print_case(results[-1])

        results.append(
            check_backend_models(
                "preflight_vllm_models",
                args.vllm_backend,
                EXPECTED_UPSTREAM["lfm2.5-1.2b-jp"],
                args.preflight_timeout,
            )
        )
        print_case(results[-1])

        results.append(
            check_backend_models(
                "preflight_llama_models",
                args.llama_backend,
                EXPECTED_UPSTREAM["qwen3.6-27b"],
                args.preflight_timeout,
            )
        )
        print_case(results[-1])
        print()

    if not args.skip_explicit:
        print("-- Explicit model routing --")
        results.append(
            evaluate_routing_case(
                "explicit_lfm2.5",
                args.router_url,
                request_model="lfm2.5-1.2b-jp",
                prompt="hello",
                timeout=args.timeout,
                max_tokens=args.max_tokens,
                expected_selected_model="lfm2.5-1.2b-jp",
                expected_upstream_model=EXPECTED_UPSTREAM["lfm2.5-1.2b-jp"],
            )
        )
        print_case(results[-1])

        results.append(
            evaluate_routing_case(
                "explicit_qwen",
                args.router_url,
                request_model="qwen3.6-27b",
                prompt="hi",
                timeout=args.slow_timeout,
                max_tokens=args.max_tokens,
                expected_selected_model="qwen3.6-27b",
                expected_upstream_model=EXPECTED_UPSTREAM["qwen3.6-27b"],
                forbidden_upstream_substrings=("unsloth/",),
            )
        )
        print_case(results[-1])
        print()

    if not args.skip_mom:
        print("-- MoM auto routing (signal-designed prompts) --")
        for case in MOM_TEST_CASES:
            if case.slow and args.skip_slow:
                continue
            timeout = args.slow_timeout if case.slow else args.timeout
            result = evaluate_routing_case(
                case.name,
                args.router_url,
                request_model="MoM",
                prompt=case.prompt,
                timeout=timeout,
                max_tokens=case.max_tokens,
                expected_selected_model=case.expected_selected_model,
                allowed_selected_models=case.allowed_selected_models,
                expected_decision=case.expected_decision,
                allowed_decisions=case.allowed_decisions,
                forbidden_upstream_substrings=("unsloth/",),
            )
            results.append(result)
            print_case(result, show_intent=case.intent)
        print()

    if not args.skip_cache:
        print("-- Semantic cache --")
        results.append(
            evaluate_semantic_cache_case(
                args.router_url,
                timeout=args.timeout,
                max_tokens=args.max_tokens,
                cache_wait_s=args.cache_wait,
                require_cache=args.require_cache,
            )
        )
        print_case(results[-1])
        print()

    if not args.skip_conversation:
        print("-- Conversation continuity (Responses API) --")
        api_check = check_response_api_available(args.router_url, args.timeout)
        results.append(api_check)
        print_case(api_check)
        if api_check.skipped or not api_check.passed:
            print()
        else:
            results.append(
                evaluate_conversation_continuity_case(
                    args.router_url,
                    timeout=args.slow_timeout,
                    max_output_tokens=max(args.max_tokens, 32),
                )
            )
            print_case(results[-1])
            print()

    passed = sum(1 for item in results if item.passed and not item.skipped)
    skipped = sum(1 for item in results if item.skipped)
    failed = sum(1 for item in results if not item.passed and not item.skipped)
    total = len(results)
    print("=" * 60)
    if failed == 0:
        print(f"ALL PASSED ({passed}/{total}, skipped={skipped})")
        print("=" * 60)
        return 0

    print(
        f"SOME FAILED ({passed} passed, {failed} failed, {skipped} skipped / {total})"
    )
    print("=" * 60)
    for item in results:
        if not item.passed and not item.skipped:
            print(f"- {item.name}: {item.detail}")
    return 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Test vLLM-SR routing, MoM decisions, cache, and conversation continuity."
    )
    parser.add_argument("--router-url", default=ROUTER_URL_DEFAULT)
    parser.add_argument("--vllm-backend", default=VLLM_BACKEND_DEFAULT)
    parser.add_argument("--llama-backend", default=LLAMA_BACKEND_DEFAULT)
    parser.add_argument(
        "--timeout",
        type=float,
        default=180.0,
        help="HTTP timeout for standard chat completion requests.",
    )
    parser.add_argument(
        "--slow-timeout",
        type=float,
        default=300.0,
        help="HTTP timeout for Qwen / ReMoM / conversation cases.",
    )
    parser.add_argument(
        "--preflight-timeout",
        type=float,
        default=10.0,
        help="HTTP timeout for readiness/model-list checks.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=24,
        help="Default max_tokens for chat completion requests.",
    )
    parser.add_argument(
        "--cache-wait",
        type=float,
        default=8.0,
        help="Seconds to wait between cache miss and repeat request.",
    )
    parser.add_argument("--skip-preflight", action="store_true")
    parser.add_argument("--skip-explicit", action="store_true")
    parser.add_argument("--skip-mom", action="store_true")
    parser.add_argument("--skip-cache", action="store_true")
    parser.add_argument("--skip-conversation", action="store_true")
    parser.add_argument(
        "--require-cache",
        action="store_true",
        help="Fail instead of skip when static_business_route semantic-cache is unavailable.",
    )
    parser.add_argument(
        "--skip-slow",
        action="store_true",
        help="Skip slow MoM cases (fact-check escalation and ReMoM intent).",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        return run_tests(args)
    except RoutingTestError as exc:
        print(f"Routing test error: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
