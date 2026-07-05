import argparse
import base64
import json
import mimetypes
import random
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path


API_KEY = "sk-ytjoldSoalyUQAWqUkQ6Zle7mgtsDVcWKImdZyhooJbZw8GR"
BASE_URL = "https://ieuwbn-123ghiuueiud1-great.onrender.com/v1"
MODEL = "Qwen/Qwen3.5-9B"
TIMEOUT = 120.0
VERBOSE = True

ROOT_DIR = Path(__file__).resolve().parent
DATASET_ROOTS = {
    "mvtec_ad": ROOT_DIR / "mvtec_anomaly_detection",
    "mvtec_loco": ROOT_DIR / "mvtec_loco_anomaly_detection",
    "visa": ROOT_DIR / "VisA_20220922" / "VisA_20220922",
}
DATASET_CATEGORIES = {
    "mvtec_ad": [
        "bottle",
        "cable",
        "capsule",
        "carpet",
        "grid",
        "hazelnut",
        "leather",
        "metal_nut",
        "pill",
        "screw",
        "tile",
        "toothbrush",
        "transistor",
        "wood",
        "zipper",
    ],
    "mvtec_loco": [
        "breakfast_box",
        "juice_bottle",
        "pushpins",
        "screw_bag",
        "splicing_connectors",
    ],
    "visa": [
        "candle",
        "capsules",
        "cashew",
        "chewinggum",
        "fryum",
        "macaroni1",
        "macaroni2",
        "pcb1",
        "pcb2",
        "pcb3",
        "pcb4",
        "pipe_fryum",
    ],
}
CATEGORY_ALIASES = {
    "mvtec_ad": {
        "瓶子": "bottle",
        "线缆": "cable",
        "胶囊": "capsule",
        "地毯": "carpet",
        "网格": "grid",
        "榛子": "hazelnut",
        "金属螺母": "metal_nut",
        "药片": "pill",
        "螺丝": "screw",
        "瓷砖": "tile",
        "牙刷": "toothbrush",
        "晶体管": "transistor",
        "木材": "wood",
        "拉链": "zipper",
        "皮革": "leather",
    },
    "mvtec_loco": {
        "早餐盒": "breakfast_box",
        "果汁瓶": "juice_bottle",
        "图钉": "pushpins",
        "螺丝袋": "screw_bag",
        "拼接连接器": "splicing_connectors",
    },
    "visa": {
        "蜡烛": "candle",
        "胶囊": "capsules",
        "腰果": "cashew",
        "口香糖": "chewinggum",
        "环形脆片（fryum）": "fryum",
        "环形脆片(fryum)": "fryum",
        "通心粉 1": "macaroni1",
        "通心粉 2": "macaroni2",
        "印刷电路板 1（pcb1）": "pcb1",
        "印刷电路板 1(pcb1)": "pcb1",
        "印刷电路板 2（pcb2）": "pcb2",
        "印刷电路板 2(pcb2)": "pcb2",
        "印刷电路板 3（pcb3）": "pcb3",
        "印刷电路板 3(pcb3)": "pcb3",
        "印刷电路板 4（pcb4）": "pcb4",
        "印刷电路板 4(pcb4)": "pcb4",
        "棒状脆块（pipe_fryum）": "pipe_fryum",
        "棒状脆块(pipe_fryum)": "pipe_fryum",
    },
}
ABNORMAL_HEADING_KEYS = {
    "MVTec": "mvtec_ad",
    "MVTec-LOCO": "mvtec_loco",
    "VisA": "visa",
}
LOCO_DEFECT_LABELS = {
    "logical_anomalies": "逻辑异常",
    "structural_anomalies": "结构异常",
}
DEFECT_RULE_ALIASES = {
    "mvtec_ad/metal_nut/flip": "mvtec_ad/metal_nut/bent",
    "mvtec_ad/pill/pill_type": "mvtec_ad/pill/combined",
}
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

DATASET_HEADING_KEYS = {
    "MVTec AD": "mvtec_ad",
    "MVTec LOCO AD": "mvtec_loco",
    "VisA": "visa",
}
RULE_LINE_RE = re.compile(r"^\s*\d+\.\s+([^:：]+)[：:]\s*(.+)\s*$")
NUMBERED_CATEGORY_RE = re.compile(r"^\s*\d+\.\s+([^:：]+)[：:]\s*(.*)\s*$")
DEFECT_LINE_RE = re.compile(r"^\s*([^:：]+)[：:]\s*(.+)\s*$")


@dataclass(frozen=True)
class EvaluationCase:
    dataset: str
    category: str
    source: str
    expected: str
    rule: str
    image_path: Path


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


CLASSIFICATION_PROMPT = """You are an industrial quality inspection expert. Your task is to determine whether the condition described by the inspection rule is present and visible in this image.

Inspection rule:
{rule}

Step 1 - Analyze: Carefully examine the image. Describe what you see, focusing on the specific features mentioned in the rule. Note any abnormalities, defects, or conditions that match or contradict the rule.

Step 2 - Decide: Based on your analysis, determine whether the condition described by the rule is present in the image.
- If the condition is clearly present, answer y
- If the condition is clearly absent, answer n
- If unsure, lean towards answering y (detecting the condition)

Output format:
Analysis: [your observations]
Answer: y or n"""


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


def call_chat_completion(messages, max_tokens=4096):
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
    content = call_chat_completion(messages, max_tokens=4096).strip()
    if VERBOSE:
        print(f"API verification response: {content}")


def normalize_rule_key(rule_key):
    if "/" not in rule_key:
        raise ValueError("--rule-key must use dataset/category format, for example mvtec_ad/bottle")
    dataset, category = rule_key.split("/", 1)
    dataset = dataset.strip().lower()
    category = category.strip()
    if not dataset or not category:
        raise ValueError("--rule-key must use dataset/category format, for example mvtec_ad/bottle")
    return f"{dataset}/{category}"


def normalize_category(dataset, category):
    category = category.strip()
    return CATEGORY_ALIASES.get(dataset, {}).get(category, category)


def normalize_normal_rule_key(dataset, category):
    return f"{dataset}/{normalize_category(dataset, category)}"


def load_normal_rules(rules_file):
    path = Path(rules_file)
    if not path.exists():
        raise FileNotFoundError(f"Normal rule file not found: {path}")
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"Normal rule file is empty: {path}")
    found_heading, rules = parse_rules_txt(text)
    if not found_heading:
        raise ValueError(f"Normal rule file must contain dataset headings: {path}")
    return rules


def parse_abnormal_rules_txt(text):
    rules = {}
    current_dataset = None
    current_category = None
    found_heading = False
    last_key = None

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line in ABNORMAL_HEADING_KEYS:
            current_dataset = ABNORMAL_HEADING_KEYS[line]
            current_category = None
            found_heading = True
            last_key = None
            continue
        if current_dataset is None:
            continue

        category_match = NUMBERED_CATEGORY_RE.match(line)
        if category_match:
            current_category = normalize_category(current_dataset, category_match.group(1))
            last_key = None
            continue
        if current_category is None:
            continue

        defect_match = DEFECT_LINE_RE.match(line)
        if defect_match:
            defect = defect_match.group(1).strip()
            rule = defect_match.group(2).strip()
            last_key = f"{current_dataset}/{current_category}/{defect}"
            rules[last_key] = rule
        elif last_key:
            rules[last_key] = f"{rules[last_key]} {line}"

    return found_heading, rules


def load_abnormal_rules(rules_file):
    path = Path(rules_file)
    if not path.exists():
        raise FileNotFoundError(f"Abnormal rule file not found: {path}")
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"Abnormal rule file is empty: {path}")
    found_heading, rules = parse_abnormal_rules_txt(text)
    if not found_heading:
        raise ValueError(f"Abnormal rule file must contain dataset headings: {path}")
    return rules


def parse_rules_txt(text):
    rules = {}
    current_dataset = None
    found_heading = False
    last_key = None

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line in DATASET_HEADING_KEYS:
            current_dataset = DATASET_HEADING_KEYS[line]
            found_heading = True
            last_key = None
            continue
        if current_dataset is None:
            continue

        match = RULE_LINE_RE.match(line)
        if match:
            category = match.group(1).strip()
            rule = match.group(2).strip()
            last_key = normalize_normal_rule_key(current_dataset, category)
            rules[last_key] = rule
        elif last_key:
            rules[last_key] = f"{rules[last_key]} {line}"

    return found_heading, rules


def read_rule(rules_file, rule_key=None):
    path = Path(rules_file)
    if not path.exists():
        raise FileNotFoundError(f"Rule file not found: {path}")

    text = path.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"Rule file is empty: {path}")

    found_heading, rules = parse_rules_txt(text)
    if not found_heading:
        return text

    if not rule_key:
        raise ValueError(
            "Rule file contains multiple dataset/category rules; pass --rule-key dataset/category "
            "(for example mvtec_ad/bottle)."
        )

    key = normalize_rule_key(rule_key)
    if key not in rules:
        available = ", ".join(sorted(rules))
        raise ValueError(f"Rule key not found: {key}. Available keys: {available}")
    return rules[key]


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
            content = call_chat_completion(messages, max_tokens=4096)
            return normalize_y_n(content)
        except (RuntimeError, ValueError, TimeoutError) as exc:
            last_error = exc
            if attempt < max_retries - 1:
                delay = retry_delay * (2 ** attempt)
                if VERBOSE:
                    print(f"  Retry {attempt + 1}/{max_retries} after {delay:.1f}s: {exc}")
                time.sleep(delay)

    if VERBOSE:
        print(f"  All {max_retries} retries failed: {last_error}")
    return None


def build_classification_prompt(rule):
    return CLASSIFICATION_PROMPT.format(rule=rule)


def normalize_classification(content):
    text = content.strip().lower()
    for line in reversed(text.splitlines()):
        line = line.strip()
        if "answer:" in line:
            after = line.split("answer:", 1)[1].strip()
            if after.startswith("y"):
                return "y"
            if after.startswith("n"):
                return "n"
    return normalize_y_n(content)


def classify_image(rule, image_path, max_retries=3, retry_delay=2.0):
    """Return y/n for whether an image satisfies a rule, without exposing the expected label."""
    encoded_image, mime_type = encode_image(image_path)
    prompt = build_classification_prompt(rule)
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
            content = call_chat_completion(messages, max_tokens=4096)
            return normalize_classification(content)
        except (RuntimeError, ValueError, TimeoutError) as exc:
            last_error = exc
            if attempt < max_retries - 1:
                delay = retry_delay * (2 ** attempt)
                if VERBOSE:
                    print(f"  Retry {attempt + 1}/{max_retries} after {delay:.1f}s: {exc}")
                time.sleep(delay)

    if VERBOSE:
        print(f"  All {max_retries} retries failed: {last_error}")
    return None


def discover_images(directory):
    if not directory.exists():
        raise FileNotFoundError(f"Image directory not found: {directory}")
    return sorted(
        path for path in directory.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def sample_images(directory, samples_per_source, rng):
    images = discover_images(directory)
    if not images:
        raise ValueError(f"No images found in {directory}")
    if len(images) <= samples_per_source:
        return images
    return rng.sample(images, samples_per_source)


def require_rule(rules, key):
    if key not in rules:
        alias_key = DEFECT_RULE_ALIASES.get(key)
        if alias_key and alias_key in rules:
            return rules[alias_key]
        raise KeyError(f"Missing rule definition: {key}")
    return rules[key]


def selected_datasets(dataset):
    if dataset == "all":
        return ["mvtec_ad", "mvtec_loco", "visa"]
    return [dataset]


def add_sampled_cases(cases, dataset, category, source, expected, rule, directory, samples_per_source, rng):
    for image_path in sample_images(directory, samples_per_source, rng):
        cases.append(EvaluationCase(dataset, category, source, expected, rule, image_path))


def build_evaluation_cases(
    normal_rules_file="Rules.txt",
    abnormal_rules_file="Rules_yichang",
    samples_per_source=1,
    seed=None,
    dataset="all",
):
    if samples_per_source < 1:
        raise ValueError("--samples-per-source must be at least 1")

    normal_rules = load_normal_rules(normal_rules_file)
    abnormal_rules = load_abnormal_rules(abnormal_rules_file)
    rng = random.Random(seed)
    cases = []

    for dataset_name in selected_datasets(dataset):
        root = DATASET_ROOTS[dataset_name]
        if not root.exists():
            raise FileNotFoundError(f"Dataset root not found: {root}")
        for category in DATASET_CATEGORIES[dataset_name]:
            normal_key = f"{dataset_name}/{category}"
            normal_rule = require_rule(normal_rules, normal_key)

            if dataset_name == "visa":
                normal_dir = root / category / "Data" / "Images" / "Normal"
                abnormal_dir = root / category / "Data" / "Images" / "Anomaly"
                abnormal_key = f"{dataset_name}/{category}/bad"
                abnormal_rule = require_rule(abnormal_rules, abnormal_key)
                add_sampled_cases(
                    cases,
                    dataset_name,
                    category,
                    "Normal",
                    "normal",
                    normal_rule,
                    normal_dir,
                    samples_per_source,
                    rng,
                )
                add_sampled_cases(
                    cases,
                    dataset_name,
                    category,
                    "Anomaly",
                    "abnormal",
                    abnormal_rule,
                    abnormal_dir,
                    samples_per_source,
                    rng,
                )
            elif dataset_name == "mvtec_loco":
                test_dir = root / category / "test"
                add_sampled_cases(
                    cases,
                    dataset_name,
                    category,
                    "good",
                    "normal",
                    normal_rule,
                    test_dir / "good",
                    samples_per_source,
                    rng,
                )
                for source, defect_label in LOCO_DEFECT_LABELS.items():
                    abnormal_key = f"{dataset_name}/{category}/{defect_label}"
                    abnormal_rule = require_rule(abnormal_rules, abnormal_key)
                    add_sampled_cases(
                        cases,
                        dataset_name,
                        category,
                        source,
                        "abnormal",
                        abnormal_rule,
                        test_dir / source,
                        samples_per_source,
                        rng,
                    )
            else:
                test_dir = root / category / "test"
                add_sampled_cases(
                    cases,
                    dataset_name,
                    category,
                    "good",
                    "normal",
                    normal_rule,
                    test_dir / "good",
                    samples_per_source,
                    rng,
                )
                for defect_dir in sorted(path for path in test_dir.iterdir() if path.is_dir() and path.name != "good"):
                    abnormal_key = f"{dataset_name}/{category}/{defect_dir.name}"
                    abnormal_rule = require_rule(abnormal_rules, abnormal_key)
                    add_sampled_cases(
                        cases,
                        dataset_name,
                        category,
                        defect_dir.name,
                        "abnormal",
                        abnormal_rule,
                        defect_dir,
                        samples_per_source,
                        rng,
                    )

    return cases


def collect_evaluation_results(cases):
    totals = {
        "correct": 0,
        "total": 0,
        "normal_correct": 0,
        "normal_total": 0,
        "abnormal_correct": 0,
        "abnormal_total": 0,
    }
    results = []
    total_cases = len(cases)
    for index, case in enumerate(cases, start=1):
        print(
            f"[{index}/{total_cases}] checking {case.dataset}/{case.category}/{case.source}: {case.image_path}",
            flush=True,
        )
        match = classify_image(case.rule, case.image_path)
        prediction = None if match is None else case.expected if match == "y" else opposite_label(case.expected)
        is_correct = prediction == case.expected
        totals["total"] += 1
        totals[f"{case.expected}_total"] += 1
        if is_correct:
            totals["correct"] += 1
            totals[f"{case.expected}_correct"] += 1
        results.append(
            {
                "dataset": case.dataset,
                "category": case.category,
                "source": case.source,
                "image_path": str(case.image_path),
                "expected": case.expected,
                "model_answer": match,
                "prediction": prediction,
                "correct": is_correct,
                "rule": case.rule,
            }
        )
        print(
            f"[{index}/{total_cases}] answer={match} prediction={prediction} expected={case.expected} correct={is_correct}",
            flush=True,
        )
    return totals, results


def run_evaluation(cases):
    totals, _ = collect_evaluation_results(cases)
    return totals


def opposite_label(label):
    if label == "normal":
        return "abnormal"
    if label == "abnormal":
        return "normal"
    raise ValueError(f"Unknown label: {label}")


def print_evaluation_summary(totals):
    print(f"total correct: {totals['correct']}/{totals['total']}")
    print(f"normal correct: {totals['normal_correct']}/{totals['normal_total']}")
    print(f"abnormal correct: {totals['abnormal_correct']}/{totals['abnormal_total']}")


def write_evaluation_results(output_file, totals, results):
    output_path = Path(output_file)
    payload = {"summary": totals, "results": results}
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return output_path


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Check one image against an inspection rule and print y or n."
    )
    parser.add_argument("image", nargs="?", help="Image path to check.")
    parser.add_argument("--image", dest="image_option", help="Image path to check.")
    parser.add_argument(
        "--rules-file",
        default="Rules.txt",
        help="Rule file to read. Defaults to Rules.txt.",
    )
    parser.add_argument(
        "--rule-key",
        help="Dataset/category key to extract from a multi-rule Rules.txt file, for example mvtec_ad/bottle.",
    )
    parser.add_argument(
        "--evaluate-random",
        action="store_true",
        help="Randomly evaluate normal and abnormal samples from all configured datasets.",
    )
    parser.add_argument(
        "--normal-rules-file",
        default="Rules.txt",
        help="Normal rules file for --evaluate-random. Defaults to Rules.txt.",
    )
    parser.add_argument(
        "--abnormal-rules-file",
        default="Rules_yichang",
        help="Abnormal rules file for --evaluate-random. Defaults to Rules_yichang.",
    )
    parser.add_argument(
        "--samples-per-source",
        type=int,
        default=1,
        help="Images to sample from each evaluation source. Defaults to 1.",
    )
    parser.add_argument("--seed", type=int, help="Random seed for --evaluate-random.")
    parser.add_argument(
        "--output-file",
        default="evaluation_results.json",
        help="JSON file to save evaluation details. Defaults to evaluation_results.json.",
    )
    parser.add_argument(
        "--dataset",
        choices=["all", "mvtec_ad", "mvtec_loco", "visa"],
        default="all",
        help="Dataset to evaluate. Defaults to all.",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify the API with a text-only request before checking the image.",
    )
    parser.add_argument(
        "--skip-verify",
        action="store_true",
        help="Skip API verification even when --verify is set.",
    )
    return parser


def main():
    parser = build_arg_parser()
    args = parser.parse_args()

    if args.evaluate_random:
        if args.verify and not args.skip_verify:
            verify_api_connection()
        cases = build_evaluation_cases(
            normal_rules_file=args.normal_rules_file,
            abnormal_rules_file=args.abnormal_rules_file,
            samples_per_source=args.samples_per_source,
            seed=args.seed,
            dataset=args.dataset,
        )
        totals, results = collect_evaluation_results(cases)
        print_evaluation_summary(totals)
        output_path = write_evaluation_results(args.output_file, totals, results)
        print(f"results saved to: {output_path}")
        return

    image_path = args.image_option or args.image
    if not image_path:
        parser.error("an image path is required, either as a positional argument or with --image")

    image = Path(image_path)
    if not image.exists():
        raise FileNotFoundError(f"Image file not found: {image}")

    if args.verify and not args.skip_verify:
        verify_api_connection()

    rule = read_rule(args.rules_file, args.rule_key)
    answer = check_image(rule, image)
    if answer is None:
        print("error: all retries failed")
        raise SystemExit(1)
    print(answer)


if __name__ == "__main__":
    main()
