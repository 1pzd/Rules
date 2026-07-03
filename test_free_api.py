"""Test OpenAI-compatible FREE API chat models.

Usage examples:
    python scripts/test_free_api.py
    python scripts/test_free_api.py --models qwen/qwen3.6-27b gemma-4-31b-it
    python scripts/test_free_api.py --raw
    python scripts/test_free_api.py --dry-run

Environment variables (optional overrides):
    FREE_API_KEY        - API key (default: built-in)
    FREE_API_BASE_URL   - Base URL (default: built-in)
    FREE_API_MODELS     - Comma-separated model list
    FREE_API_PROMPT     - Custom prompt
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from typing import Iterable
from urllib import error, request


DEFAULT_MODELS = ["qwen/qwen3.6-27b", "gemma-4-31b-it"]
DEFAULT_PROMPT = "Explain what a large language model is in 3 sentences."
DEFAULT_API_KEY = "sk-ytjoldSoalyUQAWqUkQ6Zle7mgtsDVcWKImdZyhooJbZw8GR"
DEFAULT_BASE_URL = "https://ieuwbn-123ghiuueiud1-great.onrender.com/v1"

_THINK_PAIRED = re.compile(r"<think>.*?</think>", re.DOTALL)
_THINK_OPEN_ONLY = re.compile(r"<think>.*", re.DOTALL)
_THINK_CLOSE_ONLY = re.compile(r".*</think>", re.DOTALL)


def strip_thinking_tags(text: str) -> str:
    """Remove <think>...</think> chain-of-thought blocks leaked by FREE API.

    Handles three cases:
    1. Properly paired <think>...</think> → remove the block
    <think> without </think> → remove everything from <think> onward
    <think> without <think> → remove everything before </think>    """
    if not text:
        return text
    has_open = "<think>" in text
    has_close = "</think>" in text
    if has_open and has_close:
        # Case 1: paired tags
        return _THINK_PAIRED.sub("", text).strip()
    if has_open:
        # Case 2: open tag only - remove from <think> to end
        return _THINK_OPEN_ONLY.sub("", text).strip()
    if has_close:
        # Case 3: close tag only - remove from start to </think>
        return _THINK_CLOSE_ONLY.sub("", text).strip()
    return text.strip()


@dataclass(frozen=True)
class TestConfig:
    api_key: str
    base_url: str
    models: list[str]
    prompt: str
    temperature: float
    max_tokens: int
    timeout: float
    dry_run: bool
    raw: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Test FREE API models through an OpenAI-compatible /v1/chat/completions endpoint."
    )
    parser.add_argument(
        "--api-key",
        default=os.getenv("FREE_API_KEY", DEFAULT_API_KEY),
        help="FREE API key. Defaults to built-in key or FREE_API_KEY.",
    )
    parser.add_argument(
        "--base-url",
        default=os.getenv("FREE_API_BASE_URL", DEFAULT_BASE_URL),
        help="OpenAI-compatible base URL, usually ending with /v1. Defaults to built-in URL or FREE_API_BASE_URL.",
    )
    parser.add_argument(
        "--models",
        nargs="*",
        default=None,
        help="Model names to test. Defaults to FREE_API_MODELS or built-in examples.",
    )
    parser.add_argument(
        "--prompt",
        default=os.getenv("FREE_API_PROMPT", DEFAULT_PROMPT),
        help="Prompt used for every model test. Defaults to FREE_API_PROMPT.",
    )
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument(
    "--dry-run",
    action="store_true",
    help="Print resolved config and exit without sending requests.",
)
    parser.add_argument(
    "--raw",
    action="store_true",
    help="Print the raw response before stripping thinking tags (for debugging).",
)
    return parser.parse_args()


def split_models(values: Iterable[str] | None) -> list[str]:
    if values:
        raw = ",".join(values)
    else:
        raw = os.getenv("FREE_API_MODELS", "")

    models = [item.strip() for item in raw.split(",") if item.strip()]
    return models or DEFAULT_MODELS


def build_config(args: argparse.Namespace) -> TestConfig:
    return TestConfig(
        api_key=args.api_key.strip(),
        base_url=args.base_url.strip().rstrip("/"),
        models=split_models(args.models),
        prompt=args.prompt,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        timeout=args.timeout,
        dry_run=args.dry_run,
        raw=args.raw,
    )


def chat_completions_url(base_url: str) -> str:
    if base_url.endswith("/chat/completions"):
        return base_url
    return f"{base_url}/chat/completions"


def mask_key(api_key: str) -> str:
    if not api_key:
        return "<missing>"
    if len(api_key) <= 8:
        return "***"
    return f"{api_key[:4]}...{api_key[-4:]}"


def validate_config(config: TestConfig) -> None:
    if config.dry_run:
        return
    missing = []
    if not config.api_key:
        missing.append("FREE_API_KEY or --api-key")
    if not config.base_url:
        missing.append("FREE_API_BASE_URL or --base-url")
    if missing:
        raise ValueError("Missing required configuration: " + ", ".join(missing))


def make_payload(config: TestConfig, model: str) -> bytes:
    payload = {
        "model": model,
        "messages": [
            {"role": "user", "content": config.prompt},
        ],
    }
    return json.dumps(payload, ensure_ascii=False).encode("utf-8")


def post_chat_completion(config: TestConfig, model: str) -> tuple[float, str, str]:
    req = request.Request(
        chat_completions_url(config.base_url),
        data=make_payload(config, model),
        headers={
            "Authorization": f"Bearer {config.api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    started = time.perf_counter()
    with request.urlopen(req, timeout=config.timeout) as resp:
        body = resp.read().decode("utf-8")
    elapsed = time.perf_counter() - started

    data = json.loads(body)
    raw_content = data["choices"][0]["message"]["content"]
    return elapsed, raw_content, strip_thinking_tags(raw_content)


def print_config(config: TestConfig) -> None:
    print("FREE API test configuration")
    print(f"  base_url: {config.base_url or '<missing>'}")
    print(f"  api_key:  {mask_key(config.api_key)}")
    print(f"  models:   {', '.join(config.models)}")
    print(f"  prompt:   {config.prompt}")
    print(f"  timeout:  {config.timeout}s")


def run(config: TestConfig) -> int:
    print_config(config)
    if config.dry_run:
        print("\nDry run only: no request was sent.")
        return 0

    validate_config(config)
    failures = 0
    for model in config.models:
        print(f"\n[{model}] testing...")
        try:
            elapsed, raw_content, content = post_chat_completion(config, model)
        except error.HTTPError as exc:
            failures += 1
            detail = exc.read().decode("utf-8", errors="replace")
            print(f"  FAIL HTTP {exc.code}: {detail[:500]}")
        except (error.URLError, TimeoutError, json.JSONDecodeError, KeyError) as exc:
            failures += 1
            print(f"  FAIL {type(exc).__name__}: {exc}")
        else:
            if config.raw and raw_content != content:
                print(f"  [raw response]: {raw_content[:1000]}")
                print(f"  [after strip]:  {content[:500]}")
            if not content:
                failures += 1
                print(f"  FAIL latency={elapsed:.2f}s: empty response")
            else:
                snippet = content.replace("\n", " ")[:300]
                print(f"  OK latency={elapsed:.2f}s")
                print(f"  response: {snippet}")

    if failures:
        print(f"\nCompleted with {failures} failed model(s).")
        return 1
    print("\nAll model tests passed.")
    return 0


def main() -> int:
    config = build_config(parse_args())
    try:
        return run(config)
    except ValueError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
