import argparse
import base64
import json
import mimetypes
import time
import urllib.error
import urllib.request
from pathlib import Path


API_KEY = "sk-ytjoldSoalyUQAWqUkQ6Zle7mgtsDVcWKImdZyhooJbZw8GR"
BASE_URL = "https://ieuwbn-123ghiuueiud1-great.onrender.com/v1"
MODEL = "gemma-4-31b-it"
TIMEOUT = 45.0
VERBOSE = True


CHECK_PROMPT = """You are an industrial quality inspection expert. Analyze this image and decide whether it conforms to the single provided inspection rule.

Inspection rule:
{rule}

Your task: Determine whether the visible image content satisfies the inspection rule.

Focus on these aspects when they are relevant to the rule:
1. APPEARANCE: Shape, color, texture, surface finish, transparency, patterns
2. LOGIC: Spatial relationships between components, alignment, symmetry, expected positions
3. QUANTITY: Number of objects, components, holes, edges, or features present

Requirements:
- Use only the provided inspection rule as the decision standard
- Judge only the object content visible in the image
- If the image satisfies the rule, answer y
- If the image does not satisfy the rule, answer n
- Do not explain your reasoning
- Do not return punctuation, markdown, or any other text

Output format: exactly one lowercase letter: y or n."""


# Detailed prompt variant for manual reference only.
# Use this version if you want the model to return y/n plus a detailed reason.
# It is intentionally commented out so the runtime behavior still returns only y or n.
#
# CHECK_PROMPT_WITH_REASON = """You are an industrial quality inspection expert. Analyze this image and decide whether it conforms to the single provided inspection rule.
#
# Inspection rule:
# {rule}
#
# Your task: Determine whether the visible image content satisfies the inspection rule, then explain the reason in detail.
#
# Focus on these aspects when they are relevant to the rule:
# 1. APPEARANCE: Shape, color, texture, surface finish, transparency, patterns
# 2. LOGIC: Spatial relationships between components, alignment, symmetry, expected positions
# 3. QUANTITY: Number of objects, components, holes, edges, or features present
#
# Requirements:
# - Use only the provided inspection rule as the decision standard
# - Judge only the object content visible in the image
# - First answer y if the image satisfies the rule, or n if it does not
# - Then explain the visual evidence that supports the decision
# - Mention which part of the rule is satisfied or violated
# - Do not discuss unrelated background details unless they affect the rule
#
# Output format:
# Answer: y or n
# Reason: detailed explanation of the visual evidence."""


def encode_image(image_path):
    path = Path(image_path)
    mime_type, _ = mimetypes.guess_type(path)
    if mime_type is None:
        mime_type = "image/png"

    with path.open("rb") as image_file:
        encoded = base64.b64encode(image_file.read()).decode("utf-8")
    return encoded, mime_type


def call_chat_completion(messages, max_tokens=3):
    payload = {
        "model": MODEL,
        "messages": messages,
        "max_tokens": max_tokens,
        "stream": False,
    }

    request = urllib.request.Request(
        f"{BASE_URL}/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT) as response:
            data = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code}: {body}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"API request failed: {exc}") from exc

    try:
        return data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError(f"Unexpected API response: {data}") from exc


def verify_api_connection():
    messages = [{"role": "user", "content": "Return exactly: ok"}]
    content = call_chat_completion(messages, max_tokens=5).strip()
    if VERBOSE:
        print(f"API verification response: {content}")


def read_rule(rules_file):
    path = Path(rules_file)
    if not path.exists():
        raise FileNotFoundError(f"Rule file not found: {path}")

    rule = path.read_text(encoding="utf-8").strip()
    if not rule:
        raise ValueError(f"Rule file is empty: {path}")
    return rule


def build_prompt(rule):
    return CHECK_PROMPT.format(rule=rule)


def normalize_y_n(content):
    text = content.strip().lower()
    if text == "y" or text.startswith("y") or text.startswith("yes"):
        return "y"
    if text == "n" or text.startswith("n") or text.startswith("no"):
        return "n"
    raise ValueError(f"Model did not return y/n: {content!r}")


def check_image(rule, image_path, max_retries=3, retry_delay=2.0):
    """Check an image against a rule with automatic retry on transient failures.

    Args:
        rule: The inspection rule text.
        image_path: Path to the image file.
        max_retries: Maximum number of retry attempts (default 3).
        retry_delay: Base delay in seconds between retries; doubles each attempt.

    Returns:
        "y" or "n", or None if all retries fail.
    """
    encoded_image, mime_type = encode_image(image_path)
    prompt = build_prompt(rule)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:{mime_type};base64,{encoded_image}"},
                },
            ],
        }
    ]

    last_error = None
    for attempt in range(max_retries):
        try:
            content = call_chat_completion(messages, max_tokens=5)
            return normalize_y_n(content)
        except (RuntimeError, ValueError) as exc:
            last_error = exc
            if attempt < max_retries - 1:
                delay = retry_delay * (2 ** attempt)
                if VERBOSE:
                    print(f"  Retry {attempt + 1}/{max_retries} after {delay:.1f}s: {exc}")
                time.sleep(delay)

    if VERBOSE:
        print(f"  All {max_retries} retries failed: {last_error}")
    return None


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Check one image against the single rule in rules_detected.txt and print y or n."
    )
    parser.add_argument("image", nargs="?", help="Image path to check.")
    parser.add_argument("--image", dest="image_option", help="Image path to check.")
    parser.add_argument(
        "--rules-file",
        default="rules_detected.txt",
        help="Single-rule text file to read. Defaults to rules_detected.txt.",
    )
    parser.add_argument("--verify", action="store_true", help="Verify the API with a text-only request before checking the image.")
    parser.add_argument("--skip-verify", action="store_true", help="Skip API verification even when --verify is set.")
    return parser


def main():
    parser = build_arg_parser()
    args = parser.parse_args()

    image_path = args.image_option or args.image
    if not image_path:
        parser.error("an image path is required, either as a positional argument or with --image")

    image = Path(image_path)
    if not image.exists():
        raise FileNotFoundError(f"Image file not found: {image}")

    if args.verify and not args.skip_verify:
        verify_api_connection()

    rule = read_rule(args.rules_file)
    answer = check_image(rule, image)
    if answer is None:
        print("error: all retries failed")
        raise SystemExit(1)
    print(answer)


if __name__ == "__main__":
    main()
