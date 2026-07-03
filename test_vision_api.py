"""Test FREE API vision capability with a local image.

Usage:
    1. Set IMAGE_PATH below to your image file
    2. Run: python test_vision_api.py
"""

from __future__ import annotations

import base64
import json
import mimetypes
import socket
import sys
import time
from pathlib import Path
from urllib import error, request


# ========== CONFIGURATION ==========
IMAGE_PATH = "mvtec_anomaly_detection/bottle/train/good/000.png"
PROMPT = "Describe this image in detail. What do you see?"
MODELS = ["gemma-4-31b-it"]
API_KEY = "sk-ytjoldSoalyUQAWqUkQ6Zle7mgtsDVcWKImdZyhooJbZw8GR"
BASE_URL = "https://ieuwbn-123ghiuueiud1-great.onrender.com/v1"
TIMEOUT = 60.0
STREAM = False                  # gemma doesn't need streaming
# ====================================


def encode_image(image_path: str) -> tuple[str, str]:
    """Read image file and return (base64_data, mime_type)."""
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    mime_type = mimetypes.guess_type(str(path))[0] or "image/jpeg"
    with open(path, "rb") as f:
        data = base64.b64encode(f.read()).decode("utf-8")
    return data, mime_type


def make_payload(model: str, image_b64: str, mime_type: str, prompt: str, stream: bool = False) -> bytes:
    """Build OpenAI vision API payload."""
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{mime_type};base64,{image_b64}"
                        },
                    },
                ],
            }
        ],
        "max_tokens": 1024,
        "stream": stream,
    }
    return json.dumps(payload, ensure_ascii=False).encode("utf-8")


def test_vision(
    base_url: str,
    api_key: str,
    model: str,
    image_b64: str,
    mime_type: str,
    prompt: str,
    timeout: float = 60.0,
    stream: bool = False,
) -> tuple[float, str]:
    """Send vision request and return (latency, response_text)."""
    url = f"{base_url}/chat/completions"
    payload = make_payload(model, image_b64, mime_type, prompt, stream)

    req = request.Request(
        url,
        data=payload,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    started = time.perf_counter()
    try:
        with request.urlopen(req, timeout=timeout) as resp:
            if stream:
                # Parse SSE stream with chunk reading
                content_parts = []
                buffer = ""
                chunk_size = 4096
                print("  [streaming]", end="", flush=True)
                
                while True:
                    try:
                        chunk = resp.read(chunk_size)
                        if not chunk:
                            break
                        buffer += chunk.decode("utf-8", errors="replace")
                        
                        # Process complete lines
                        while "\n" in buffer:
                            line, buffer = buffer.split("\n", 1)
                            line = line.strip()
                            if not line or line == "data: [DONE]":
                                continue
                            if line.startswith("data: "):
                                try:
                                    data = json.loads(line[6:])
                                    delta = data.get("choices", [{}])[0].get("delta", {})
                                    if "content" in delta:
                                        content_parts.append(delta["content"])
                                        print(".", end="", flush=True)
                                except json.JSONDecodeError:
                                    continue
                    except socket.timeout:
                        print(" [timeout]", flush=True)
                        break
                
                print(f" [done]", flush=True)
                content = "".join(content_parts)
            else:
                body = resp.read().decode("utf-8")
                data = json.loads(body)
                content = data["choices"][0]["message"]["content"]
    except socket.timeout:
        raise TimeoutError(f"Request timed out after {timeout}s")
    elapsed = time.perf_counter() - started

    return elapsed, content


def main() -> int:
    print(f"Image: {IMAGE_PATH}")
    print(f"Prompt: {PROMPT}")
    print(f"Models: {', '.join(MODELS)}")
    print()

    # Encode image once
    try:
        image_b64, mime_type = encode_image(IMAGE_PATH)
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    print(f"Image loaded: {mime_type}, {len(image_b64)} chars base64\n")

    failures = 0
    for model in MODELS:
        print(f"[{model}] testing vision...")
        try:
            elapsed, content = test_vision(
                BASE_URL, API_KEY, model,
                image_b64, mime_type, PROMPT, TIMEOUT, STREAM
            )
            print(f"  OK latency={elapsed:.2f}s")
            print(f"  response: {content[:500]}")
        except error.HTTPError as exc:
            failures += 1
            detail = exc.read().decode("utf-8", errors="replace")
            print(f"  FAIL HTTP {exc.code}: {detail[:300]}")
        except Exception as exc:
            failures += 1
            print(f"  FAIL {type(exc).__name__}: {exc}")
        print()

    if failures:
        print(f"Completed with {failures} failure(s).")
        return 1
    print("All vision tests passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
