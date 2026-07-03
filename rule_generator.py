"""
Unified Rule Generator for Industrial Anomaly Detection Datasets.

Uses gemma-4-31b-it to generate normal appearance rules from train/good images.
Supports: MVTec AD, MVTec LOCO AD, VisA (~30 categories total).

Usage:
    1. Set DATASET_ROOTS below to your dataset paths
    2. Run: python rule_generator.py
    3. Rules saved to: generated_rules.json + Rules.txt
"""

from __future__ import annotations

import base64
import json
import mimetypes
import random
import sys
import time
from pathlib import Path
from urllib import request, error

# ========== CONFIGURATION ==========
API_KEY = "sk-ytjoldSoalyUQAWqUkQ6Zle7mgtsDVcWKImdZyhooJbZw8GR"
BASE_URL = "https://ieuwbn-123ghiuueiud1-great.onrender.com/v1"
MODEL = "gemma-4-31b-it"
TIMEOUT = 90.0
SAMPLES_PER_CATEGORY = 3
OUTPUT_FILE = "generated_rules.json"
RULES_TXT_FILE = "Rules.txt"
VERBOSE = True  # Set to True to see full API responses

# Dataset roots - UPDATE THESE PATHS WHEN YOU HAVE THE DATASETS
DATASET_ROOTS = {
    "mvtec_ad": r"E:\MvTeC\mvtec_anomaly_detection",
    "mvtec_loco": r"E:\MvTeC\mvtec_loco_anomaly_detection",
    "visa": r"E:\MvTeC\VisA_20220922\VisA_20220922",
}

# MVTec AD categories (15 industrial objects)
MVTEC_AD_CATEGORIES = [
    "bottle", "cable", "capsule", "carpet", "grid",
    "hazelnut", "leather", "metal_nut", "pill", "screw",
    "tile", "toothbrush", "transistor", "wood", "zipper",
]

# MVTec LOCO AD categories (logic anomalies focus)
MVTEC_LOCO_CATEGORIES = [
    "breakfast_box", "juice_bottle", "pushpins", "screw_bag", "splicing_connectors",
]

# VisA categories (12 industrial objects)
VISA_CATEGORIES = [
    "candle", "capsules", "cashew", "chewinggum",
    "fryum", "macaroni1", "macaroni2",
    "pcb1", "pcb2", "pcb3", "pcb4", "pipe_fryum",
]
# ====================================


UNIFIED_PROMPT = """You are an industrial quality inspection expert. Analyze these 3 images of NORMAL (good) samples from the same product category.

Your task: Generate a concise rule that describes what a NORMAL sample should look like.

Focus on these aspects:
1. APPEARANCE: Shape, color, texture, surface finish, transparency, patterns
2. LOGIC: Spatial relationships between components, alignment, symmetry, expected positions
3. QUANTITY: Number of objects, components, holes, edges, or features present

Requirements:
- IGNORE background variations - focus only on the object itself
- Describe ONLY what normal looks like - do not describe defects
- Be specific and measurable where possible (e.g., "cylindrical" not "round")
- Merge essential and optional features into one coherent description
- When multiple discrete objects appear and the count is consistent across all 3 images, state the exact number as it is part of the normal specification (e.g., "four candles", "six screws"). Use precise numbers rather than vague terms like "many" or "several".

Output format: A single paragraph describing the normal appearance rule."""


def encode_image(image_path: str) -> tuple[str, str]:
    """Read image file and return (base64_data, mime_type)."""
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    mime_type = mimetypes.guess_type(str(path))[0] or "image/jpeg"
    with open(path, "rb") as f:
        data = base64.b64encode(f.read()).decode("utf-8")
    return data, mime_type


def make_multi_image_payload(
    model: str,
    images: list[tuple[str, str]],  # list of (base64, mime_type)
    prompt: str,
) -> bytes:
    """Build OpenAI vision API payload with multiple images."""
    content = [{"type": "text", "text": prompt}]
    for b64, mime in images:
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:{mime};base64,{b64}"},
        })

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "max_tokens": 4096,
        "stream": False,
    }
    return json.dumps(payload, ensure_ascii=False).encode("utf-8")


def call_vision_api(
    base_url: str,
    api_key: str,
    model: str,
    images: list[tuple[str, str]],
    prompt: str,
    timeout: float = 90.0,
) -> tuple[float, str]:
    """Send multi-image vision request and return (latency, response_text)."""
    url = f"{base_url}/chat/completions"
    payload = make_multi_image_payload(model, images, prompt)

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
            body = resp.read().decode("utf-8")
            data = json.loads(body)
            content = data["choices"][0]["message"]["content"]
    except error.HTTPError as e:
        detail = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {e.code}: {detail[:300]}") from e
    elapsed = time.perf_counter() - started
    return elapsed, content


def sample_good_images(dataset_root: str, category: str, n: int = 3, dataset_name: str = "mvtec_ad") -> list[str]:
    """Sample n random images from the normal/good directory.

    Directory structure differs by dataset:
    - MVTec AD / LOCO: category/train/good/
    - VisA: category/Data/Images/Normal/
    """
    if dataset_name == "visa":
        good_dir = Path(dataset_root) / category / "Data" / "Images" / "Normal"
    else:
        good_dir = Path(dataset_root) / category / "train" / "good"

    if not good_dir.exists():
        raise FileNotFoundError(f"Good directory not found: {good_dir}")

    images = sorted(good_dir.glob("*.png")) + sorted(good_dir.glob("*.jpg")) + sorted(good_dir.glob("*.jpeg"))
    if len(images) < n:
        raise ValueError(f"Not enough images in {good_dir}: found {len(images)}, need {n}")

    selected = random.sample(images, n)
    return [str(img) for img in selected]


def generate_rule_for_category(
    dataset_name: str,
    category: str,
    dataset_root: str,
) -> dict:
    """Generate a rule for a single category."""
    print(f"  [{dataset_name}/{category}] Sampling {SAMPLES_PER_CATEGORY} images...")

    try:
        image_paths = sample_good_images(dataset_root, category, SAMPLES_PER_CATEGORY, dataset_name)
    except (FileNotFoundError, ValueError) as e:
        return {
            "dataset": dataset_name,
            "category": category,
            "status": "error",
            "error": str(e),
            "rule": None,
        }

    # Encode all images
    images = []
    for path in image_paths:
        try:
            b64, mime = encode_image(path)
            images.append((b64, mime))
        except FileNotFoundError as e:
            return {
                "dataset": dataset_name,
                "category": category,
                "status": "error",
                "error": str(e),
                "rule": None,
            }

    # Call API with retry for empty rules
    print(f"  [{dataset_name}/{category}] Calling {MODEL}...")
    print(f"  [{dataset_name}/{category}] Images: {[Path(p).name for p in image_paths]}")
    
    max_retries = 3
    for attempt in range(max_retries):
        try:
            elapsed, rule = call_vision_api(
                BASE_URL, API_KEY, MODEL, images, UNIFIED_PROMPT, TIMEOUT
            )
            
            # If rule is empty and we have retries left, try again
            if not rule.strip() and attempt < max_retries - 1:
                print(f"  [{dataset_name}/{category}] Empty response, retrying ({attempt + 2}/{max_retries})...")
                time.sleep(2.0)
                continue
            
            print(f"  [{dataset_name}/{category}] OK ({elapsed:.1f}s)")
            if VERBOSE:
                print(f"  [{dataset_name}/{category}] Rule preview: {rule[:200]}...")
            return {
                "dataset": dataset_name,
                "category": category,
                "status": "success",
                "latency": round(elapsed, 2),
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "images": [str(p) for p in image_paths],
                "rule": rule.strip(),
                "attempts": attempt + 1,
            }
        except Exception as e:
            print(f"  [{dataset_name}/{category}] FAIL: {e}")
            if attempt < max_retries - 1:
                print(f"  [{dataset_name}/{category}] Retrying ({attempt + 2}/{max_retries})...")
                time.sleep(2.0)
                continue
            return {
                "dataset": dataset_name,
                "category": category,
                "status": "error",
                "error": str(e),
                "rule": None,
                "attempts": attempt + 1,
            }


def get_all_categories() -> list[tuple[str, str, str]]:
    """Return list of (dataset_name, category, dataset_root) tuples."""
    categories = []
    for cat in MVTEC_AD_CATEGORIES:
        categories.append(("mvtec_ad", cat, DATASET_ROOTS["mvtec_ad"]))
    for cat in MVTEC_LOCO_CATEGORIES:
        categories.append(("mvtec_loco", cat, DATASET_ROOTS["mvtec_loco"]))
    for cat in VISA_CATEGORIES:
        categories.append(("visa", cat, DATASET_ROOTS["visa"]))
    return categories


# Display names for datasets
DATASET_DISPLAY_NAMES = {
    "mvtec_ad": "MVTec AD",
    "mvtec_loco": "MVTec LOCO AD",
    "visa": "VisA",
}


def write_rules_txt(results: list[dict], output_path: str) -> None:
    """Write rules to a text file grouped by dataset.

    Format:
        VisA
        1. candle: A single white candle...
        2. capsules: Blister pack with...

        MVTec AD
        1. bottle: Cylindrical transparent...
    """
    # Group results by dataset, only include successful ones
    grouped: dict[str, list[dict]] = {}
    for r in results:
        if r["status"] != "success":
            continue
        ds = r["dataset"]
        if ds not in grouped:
            grouped[ds] = []
        grouped[ds].append(r)

    lines = []
    # Maintain dataset order: visa, mvtec_ad, mvtec_loco
    dataset_order = ["visa", "mvtec_ad", "mvtec_loco"]
    for ds in dataset_order:
        if ds not in grouped:
            continue
        display_name = DATASET_DISPLAY_NAMES.get(ds, ds)
        lines.append(display_name)
        for idx, r in enumerate(grouped[ds], 1):
            category = r["category"]
            rule = r["rule"]
            # Clean up rule: replace newlines with spaces, collapse whitespace
            rule = " ".join(rule.split())
            lines.append(f"{idx}. {category}: {rule}")
        lines.append("")  # blank line between datasets

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main() -> int:
    print("=" * 60)
    print("Unified Rule Generator for Anomaly Detection Datasets")
    print("=" * 60)
    print(f"Model: {MODEL}")
    print(f"Samples per category: {SAMPLES_PER_CATEGORY}")
    print(f"Output: {OUTPUT_FILE}")
    print(f"Verbose: {VERBOSE}")
    print()

    # Check for single category mode
    single_category = None
    if len(sys.argv) > 1 and sys.argv[1] == "--single":
        if len(sys.argv) > 2:
            single_category = sys.argv[2]  # e.g., "mvtec_ad/leather"
        else:
            print("Usage: python rule_generator.py --single dataset/category")
            print("Example: python rule_generator.py --single mvtec_ad/leather")
            return 1

    # Verify API is working with a quick test
    print("Verifying API connection...")
    try:
        test_payload = {
            "model": MODEL,
            "messages": [{"role": "user", "content": "Say 'API OK' in 2 words."}],
            "max_tokens": 10,
        }
        req = request.Request(
            f"{BASE_URL}/chat/completions",
            data=json.dumps(test_payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {API_KEY}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        with request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            print(f"API Response: {data['choices'][0]['message']['content']}")
            print("API is working!")
    except Exception as e:
        print(f"API Error: {e}")
        print("Check your API_KEY and BASE_URL.")
        return 1
    print()

    # Validate dataset roots
    for name, root in DATASET_ROOTS.items():
        if not Path(root).exists():
            print(f"WARNING: Dataset root not found: {name} = {root}")
            print(f"  Update DATASET_ROOTS['{name}'] in the script.")
            print()

    # Single category mode
    if single_category:
        parts = single_category.split("/")
        if len(parts) != 2:
            print(f"Invalid format: {single_category}")
            print("Expected format: dataset/category (e.g., mvtec_ad/leather)")
            return 1
        
        dataset_name, category = parts
        if dataset_name not in DATASET_ROOTS:
            print(f"Unknown dataset: {dataset_name}")
            print(f"Available datasets: {list(DATASET_ROOTS.keys())}")
            return 1
        
        dataset_root = DATASET_ROOTS[dataset_name]
        print(f"Single category mode: {dataset_name}/{category}")
        print()
        
        result = generate_rule_for_category(dataset_name, category, dataset_root)
        
        if result["status"] == "success" and result.get("rule"):
            print(f"\nRule generated successfully!")
            print(f"Rule: {result['rule']}")
        else:
            print(f"\nFailed: {result.get('error', 'Unknown error')}")
            return 1
        
        return 0

    categories = get_all_categories()
    print(f"Total categories: {len(categories)}")
    print()

    # Process each category
    results = []
    successes = 0
    failures = 0

    for dataset_name, category, dataset_root in categories:
        print(f"Processing: {dataset_name}/{category}")
        result = generate_rule_for_category(dataset_name, category, dataset_root)
        results.append(result)

        if result["status"] == "success" and result.get("rule"):
            successes += 1
        else:
            failures += 1

        # Rate limiting: wait between requests
        time.sleep(1.0)
        print()

    # Save results
    output_path = Path(OUTPUT_FILE)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # Save rules to text file
    rules_txt_path = Path(RULES_TXT_FILE)
    write_rules_txt(results, str(rules_txt_path))

    print("=" * 60)
    print(f"JSON saved to: {output_path.absolute()}")
    print(f"Rules saved to: {rules_txt_path.absolute()}")
    print(f"Success: {successes}, Failed: {failures}")
    print("=" * 60)

    return 0 if failures == 0 else 1


if __name__ == "__main__":
    # Optional: Run with --verify to test a single category twice
    if len(sys.argv) > 1 and sys.argv[1] == "--verify":
        print("VERIFICATION MODE: Testing same category twice with different random images")
        print("If rules are similar but not identical, the model is generating fresh rules.")
        print()
        # Use first available dataset
        test_ds = "mvtec_ad"
        test_cat = "bottle"
        test_root = DATASET_ROOTS[test_ds]
        print(f"Testing: {test_ds}/{test_cat}")
        print()
        for i in range(2):
            print(f"--- Run {i+1} ---")
            result = generate_rule_for_category(test_ds, test_cat, test_root)
            if result["status"] == "success":
                print(f"Images used: {result['images']}")
                print(f"Rule: {result['rule']}")
            else:
                print(f"Error: {result['error']}")
            print()
        print("If the images differ between runs, the rules prove model is generating fresh.")
    else:
        raise SystemExit(main())


