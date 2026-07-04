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

# CoEvoSkills: import checker configuration and verification function
from rule_checker import API_KEY, BASE_URL, MODEL, TIMEOUT, check_image as checker_check_image

# ========== CONFIGURATION ==========
SAMPLES_PER_CATEGORY = 3
RULES_PER_CATEGORY = 3  # Trace2Skill: generate N rules with different images, then consolidate
OUTPUT_FILE = "generated_rules.json"
RULES_TXT_FILE = "Rules.txt"
VERBOSE = True  # Set to True to see full API responses

# CoEvoSkills: Verification loop config
VERIFICATION_ENABLED = True        # Enable/disable verification loop
VERIFICATION_MIN_ACCURACY = 0.8   # Minimum accuracy threshold (80%)
VERIFICATION_MAX_ITERATIONS = 2   # Max generate-verify-refine iterations
VERIFICATION_TEST_PER_CLASS = 2   # Number of good + bad test images per verification

SCRIPT_DIR = Path(__file__).resolve().parent

# Dataset roots
DATASET_ROOTS = {
    "mvtec_ad": str(SCRIPT_DIR / "mvtec_anomaly_detection"),
    "mvtec_loco": str(SCRIPT_DIR / "mvtec_loco_anomaly_detection"),
    "visa": str(SCRIPT_DIR / "VisA_20220922" / "VisA_20220922"),
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


# ========== SkillRL-Style Hierarchical Prompts ==========
# Layer 1: Dataset-level system prompt (shared across ALL categories)
BASE_SYSTEM_PROMPT = """You are an industrial quality inspection expert specializing in visual anomaly detection.
Your expertise covers appearance inspection (shape, color, texture, surface finish), logic inspection (spatial relationships, alignment, component arrangement), and quantity inspection (count of objects, holes, edges).

Core principles:
- Focus ONLY on the object itself, never on background
- Describe what NORMAL looks like, never describe defects
- Be specific and measurable (e.g., "cylindrical" not "round", "four screws" not "several")"""

# Layer 2: Category-specific task prompt (focused, shorter)
CATEGORY_TASK_PROMPT = """Analyze these 3 images of NORMAL samples. Generate ONE concise rule describing their normal appearance.

Cover these aspects in a single paragraph:
- APPEARANCE: shape, color, texture, surface finish, patterns
- LOGIC: spatial arrangement, alignment, symmetry of components
- QUANTITY: exact count of discrete objects/components if consistent across images

Output: One paragraph, no bullet points, no headers."""

# Trace2Skill: Consolidation prompt for merging multiple rule candidates
CONSOLIDATE_PROMPT = """You are given {n} different rule descriptions for the same product category, each generated from a different set of normal sample images.

Your task: Merge them into ONE best rule that:
1. Keeps ALL specific details that appear in any candidate (union of information)
2. Removes redundancies and contradictions
3. Prioritizes measurable/quantitative details over vague descriptions
4. Results in a single, coherent paragraph

Rules to merge:
{rules}

Output: One merged paragraph describing the normal appearance. No bullet points, no headers."""

# CoEvoSkills: Refinement prompt - tells model why previous rule failed and how to improve
REFINEMENT_PROMPT = """Your previous rule for this category was tested on sample images and did not pass verification.

Previous rule:
{previous_rule}

Problems detected:
{problems}

Your task: Generate an IMPROVED rule that fixes these specific issues while keeping the correct parts.
- If good images were rejected: the rule is too strict, broaden the criteria
- If bad images were accepted: the rule is too loose, add stricter constraints
- Focus on the specific aspects mentioned in the problems

Output: One improved paragraph. No bullet points, no headers."""


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


def consolidate_rules(
    category: str,
    rules: list[str],
) -> str | None:
    """Trace2Skill: Merge multiple rule candidates into one best rule."""
    if len(rules) == 1:
        return rules[0]

    # Format rules for the consolidation prompt
    rules_text = "\n\n".join(f"--- Rule {i+1} ---\n{r}" for i, r in enumerate(rules))
    prompt = CONSOLIDATE_PROMPT.format(n=len(rules), rules=rules_text)

    last_err = None
    for attempt in range(3):
        try:
            _, merged = call_vision_api(BASE_URL, API_KEY, MODEL, [], prompt, TIMEOUT)
            return merged.strip() if merged.strip() else None
        except Exception as e:
            last_err = e
            if attempt < 2:
                delay = 3.0 * (2 ** attempt)
                print(f"  [{category}] Consolidation retry {attempt + 1}/3 after {delay:.0f}s: {e}")
                time.sleep(delay)
    print(f"  [{category}] Consolidation failed after 3 attempts: {last_err}, falling back to longest rule")
    return max(rules, key=len) if rules else None


# ========== CoEvoSkills: Verification Functions ==========

def sample_test_images(
    dataset_root: str,
    category: str,
    n: int = 3,
    dataset_name: str = "mvtec_ad",
    image_class: str = "good",
) -> list[str]:
    """Sample n random images from test directory.

    Directory structure differs by dataset and class:
    - MVTec AD good: category/test/good/
    - MVTec AD bad: category/test/{defect_type}/ (any non-good subdir)
    - LOCO AD good: category/test/good/
    - LOCO AD bad: category/test/{logical_anomalies|structural_anomalies}/
    - VisA good: category/Data/Images/Normal/
    - VisA bad: category/Data/Images/Anomaly/
    """
    root = Path(dataset_root) / category

    if dataset_name == "visa":
        if image_class == "good":
            test_dir = root / "Data" / "Images" / "Normal"
        else:
            test_dir = root / "Data" / "Images" / "Anomaly"
    else:
        if image_class == "good":
            test_dir = root / "test" / "good"
        else:
            # Pick a random non-good subdirectory
            test_root = root / "test"
            if not test_root.exists():
                raise FileNotFoundError(f"Test directory not found: {test_root}")
            bad_dirs = [d for d in test_root.iterdir() if d.is_dir() and d.name != "good"]
            if not bad_dirs:
                raise FileNotFoundError(f"No defect directories found in {test_root}")
            test_dir = random.choice(bad_dirs)

    if not test_dir.exists():
        raise FileNotFoundError(f"Test directory not found: {test_dir}")

    images = sorted(test_dir.glob("*.png")) + sorted(test_dir.glob("*.jpg")) + sorted(test_dir.glob("*.jpeg"))
    if len(images) < n:
        raise ValueError(f"Not enough images in {test_dir}: found {len(images)}, need {n}")

    selected = random.sample(images, n)
    return [str(img) for img in selected]


def verify_rule(
    rule: str,
    dataset_root: str,
    category: str,
    dataset_name: str = "mvtec_ad",
    n_per_class: int = 3,
) -> tuple[float, list[str]]:
    """CoEvoSkills: Verify a rule against known good and bad test images.

    Returns (accuracy, problems_list).
    - accuracy: fraction of correct predictions (y for good, n for bad)
    - problems: list of human-readable problem descriptions
    """
    correct = 0
    total = 0
    problems = []

    # Test on good images (should return "y")
    try:
        good_images = sample_test_images(dataset_root, category, n_per_class, dataset_name, "good")
    except (FileNotFoundError, ValueError) as e:
        print(f"    [verify] Could not sample good test images: {e}")
        return 0.0, [f"Cannot access good test images: {e}"]

    for img_path in good_images:
        result = checker_check_image(rule, img_path)
        total += 1
        if result is None:
            problems.append(f"Good image check failed (API error): {Path(img_path).name}")
        elif result == "y":
            correct += 1
        else:
            problems.append(f"Good image rejected (false negative): {Path(img_path).name}")

    # Test on bad images (should return "n")
    try:
        bad_images = sample_test_images(dataset_root, category, n_per_class, dataset_name, "bad")
    except (FileNotFoundError, ValueError) as e:
        print(f"    [verify] Could not sample bad test images: {e}")
        # If no bad images available, score based on good images only
        accuracy = correct / total if total > 0 else 0.0
        return accuracy, problems + [f"Cannot access bad test images: {e}"]

    for img_path in bad_images:
        result = checker_check_image(rule, img_path)
        total += 1
        if result is None:
            problems.append(f"Bad image check failed (API error): {Path(img_path).name}")
        elif result == "n":
            correct += 1
        else:
            problems.append(f"Bad image accepted (false positive): {Path(img_path).name}")

    accuracy = correct / total if total > 0 else 0.0
    return accuracy, problems


def refine_rule(
    category: str,
    previous_rule: str,
    problems: list[str],
) -> str | None:
    """CoEvoSkills: Refine a rule based on verification failures."""
    problems_text = "\n".join(f"- {p}" for p in problems)
    prompt = REFINEMENT_PROMPT.format(
        previous_rule=previous_rule,
        problems=problems_text,
    )

    try:
        _, refined = call_vision_api(BASE_URL, API_KEY, MODEL, [], prompt, TIMEOUT)
        return refined.strip() if refined.strip() else None
    except Exception as e:
        print(f"  [{category}] Refinement failed: {e}")
        return None


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
    """Generate a rule for a single category using Trace2Skill parallel generation.

    SkillRL: Uses BASE_SYSTEM_PROMPT + CATEGORY_TASK_PROMPT (hierarchical).
    Trace2Skill: Generates RULES_PER_CATEGORY rules with different image samples,
                 then consolidates into one best rule.
    """
    candidates = []

    for rule_idx in range(RULES_PER_CATEGORY):
        label = f"{dataset_name}/{category}#{rule_idx+1}"
        print(f"  [{label}] Sampling {SAMPLES_PER_CATEGORY} images...", flush=True)

        try:
            image_paths = sample_good_images(dataset_root, category, SAMPLES_PER_CATEGORY, dataset_name)
        except (FileNotFoundError, ValueError) as e:
            print(f"  [{label}] Sampling failed: {e}", flush=True)
            continue

        # Encode all images
        images = []
        skip = False
        for path in image_paths:
            try:
                b64, mime = encode_image(path)
                images.append((b64, mime))
            except FileNotFoundError as e:
                print(f"  [{label}] Image error: {e}", flush=True)
                skip = True
                break
        if skip:
            continue

        # Call API with retry - using hierarchical prompt (SkillRL)
        print(f"  [{label}] Calling {MODEL}...", flush=True)
        prompt = f"{BASE_SYSTEM_PROMPT}\n\n{CATEGORY_TASK_PROMPT}"

        max_retries = 3
        for attempt in range(max_retries):
            try:
                elapsed, rule = call_vision_api(
                    BASE_URL, API_KEY, MODEL, images, prompt, TIMEOUT
                )

                if not rule.strip() and attempt < max_retries - 1:
                    print(f"  [{label}] Empty response, retrying ({attempt + 2}/{max_retries})...", flush=True)
                    time.sleep(2.0)
                    continue

                print(f"  [{label}] OK ({elapsed:.1f}s)", flush=True)
                if VERBOSE:
                    print(f"  [{label}] Rule preview: {rule[:150]}...", flush=True)
                candidates.append(rule.strip())
                break
            except Exception as e:
                print(f"  [{label}] FAIL: {e}", flush=True)
                if attempt < max_retries - 1:
                    print(f"  [{label}] Retrying ({attempt + 2}/{max_retries})...", flush=True)
                    time.sleep(2.0)
                    continue

    # No candidates at all
    if not candidates:
        return {
            "dataset": dataset_name,
            "category": category,
            "status": "error",
            "error": "All rule generation attempts failed",
            "rule": None,
        }

    # Trace2Skill: Consolidate multiple candidates into one best rule
    if len(candidates) > 1:
        print(f"  [{dataset_name}/{category}] Consolidating {len(candidates)} candidates...", flush=True)
        final_rule = consolidate_rules(f"{dataset_name}/{category}", candidates)
    else:
        final_rule = candidates[0]

    # CoEvoSkills: Verify and refine loop
    current_rule = final_rule
    best_rule = final_rule
    best_accuracy = -1.0
    verification_history = []
    if VERIFICATION_ENABLED and final_rule:

        for verify_iter in range(VERIFICATION_MAX_ITERATIONS):
            print(f"  [{dataset_name}/{category}] Verification round {verify_iter + 1}/{VERIFICATION_MAX_ITERATIONS}...", flush=True)
            accuracy, problems = verify_rule(
                current_rule, dataset_root, category, dataset_name, VERIFICATION_TEST_PER_CLASS
            )
            verification_history.append({"iteration": verify_iter + 1, "accuracy": accuracy, "problems": problems})

            print(f"  [{dataset_name}/{category}] Accuracy: {accuracy:.1%} ({'PASS' if accuracy >= VERIFICATION_MIN_ACCURACY else 'FAIL'})", flush=True)
            if problems:
                for p in problems:
                    print(f"    - {p}", flush=True)

            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_rule = current_rule

            if accuracy >= VERIFICATION_MIN_ACCURACY:
                print(f"  [{dataset_name}/{category}] Verification PASSED at round {verify_iter + 1}", flush=True)
                break

            if verify_iter < VERIFICATION_MAX_ITERATIONS - 1:
                print(f"  [{dataset_name}/{category}] Refining rule based on failures...", flush=True)
                refined = refine_rule(f"{dataset_name}/{category}", current_rule, problems)
                if refined:
                    current_rule = refined
                    if VERBOSE:
                        print(f"  [{dataset_name}/{category}] Refined rule preview: {refined[:150]}...", flush=True)
                else:
                    print(f"  [{dataset_name}/{category}] Refinement failed, keeping previous rule", flush=True)
                    break
            else:
                print(f"  [{dataset_name}/{category}] Max verification rounds reached, using best rule (accuracy: {best_accuracy:.1%})", flush=True)

        final_rule = best_rule

    return {
        "dataset": dataset_name,
        "category": category,
        "status": "success",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "rule": final_rule,
        "candidates_count": len(candidates),
        "verification": verification_history if VERIFICATION_ENABLED else None,
        "final_accuracy": best_accuracy if VERIFICATION_ENABLED else None,
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
    import sys
    print("=" * 60, flush=True)
    print("Unified Rule Generator for Anomaly Detection Datasets", flush=True)
    print("=" * 60, flush=True)
    print(f"Model: {MODEL}", flush=True)
    print(f"Samples per category: {SAMPLES_PER_CATEGORY}", flush=True)
    print(f"Rules per category (Trace2Skill): {RULES_PER_CATEGORY}", flush=True)
    print(f"Verification (CoEvoSkills): {'ON' if VERIFICATION_ENABLED else 'OFF'}", flush=True)
    if VERIFICATION_ENABLED:
        print(f"  Min accuracy: {VERIFICATION_MIN_ACCURACY:.0%}, Max iterations: {VERIFICATION_MAX_ITERATIONS}, Test images: {VERIFICATION_TEST_PER_CLASS}", flush=True)
    print(f"Output: {OUTPUT_FILE}", flush=True)
    print(f"Verbose: {VERBOSE}", flush=True)
    print(flush=True)

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
    print("Verifying API connection...", flush=True)
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
            print(f"API Response: {data['choices'][0]['message']['content']}", flush=True)
            print("API is working!", flush=True)
    except Exception as e:
        print(f"API Error: {e}", flush=True)
        print("Check your API_KEY and BASE_URL.", flush=True)
        return 1
    print(flush=True)

    # Validate dataset roots
    for name, root in DATASET_ROOTS.items():
        if not Path(root).exists():
            print(f"WARNING: Dataset root not found: {name} = {root}", flush=True)
            print(f"  Update DATASET_ROOTS['{name}'] in the script.", flush=True)
            print(flush=True)

    # Single category mode
    if single_category:
        parts = single_category.split("/")
        if len(parts) != 2:
            print(f"Invalid format: {single_category}", flush=True)
            print("Expected format: dataset/category (e.g., mvtec_ad/leather)", flush=True)
            return 1
        
        dataset_name, category = parts
        if dataset_name not in DATASET_ROOTS:
            print(f"Unknown dataset: {dataset_name}", flush=True)
            print(f"Available datasets: {list(DATASET_ROOTS.keys())}", flush=True)
            return 1
        
        dataset_root = DATASET_ROOTS[dataset_name]
        print(f"Single category mode: {dataset_name}/{category}", flush=True)
        print(flush=True)
        
        result = generate_rule_for_category(dataset_name, category, dataset_root)
        
        if result["status"] == "success" and result.get("rule"):
            print("\nRule generated successfully!", flush=True)
            print(f"Rule: {result['rule']}", flush=True)
            if result.get("verification"):
                print(f"Verification rounds: {len(result['verification'])}", flush=True)
                print(f"Final accuracy: {result['final_accuracy']:.1%}", flush=True)
        else:
            print(f"\nFailed: {result.get('error', 'Unknown error')}", flush=True)
            return 1
        
        return 0

    categories = get_all_categories()
    print(f"Total categories: {len(categories)}", flush=True)
    print(flush=True)

    # Process each category
    results = []
    successes = 0
    failures = 0

    for dataset_name, category, dataset_root in categories:
        print(f"Processing: {dataset_name}/{category}", flush=True)
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

    print("=" * 60, flush=True)
    print(f"JSON saved to: {output_path.absolute()}", flush=True)
    print(f"Rules saved to: {rules_txt_path.absolute()}", flush=True)
    print(f"Success: {successes}, Failed: {failures}", flush=True)
    print("=" * 60, flush=True)

    return 0 if failures == 0 else 1


if __name__ == "__main__":
    # Optional: Run with --verify to test a single category twice
    if len(sys.argv) > 1 and sys.argv[1] == "--verify":
        print("VERIFICATION MODE: Testing same category twice with different random images", flush=True)
        print("If rules are similar but not identical, the model is generating fresh rules.", flush=True)
        print(flush=True)
        # Use first available dataset
        test_ds = "mvtec_ad"
        test_cat = "bottle"
        test_root = DATASET_ROOTS[test_ds]
        print(f"Testing: {test_ds}/{test_cat}", flush=True)
        print(flush=True)
        for i in range(2):
            print(f"--- Run {i+1} ---", flush=True)
            result = generate_rule_for_category(test_ds, test_cat, test_root)
            if result["status"] == "success":
                print(f"Rule: {result['rule']}", flush=True)
                if result.get("verification"):
                    print(f"Verification: {len(result['verification'])} rounds, accuracy: {result['final_accuracy']:.1%}", flush=True)
            else:
                print(f"Error: {result['error']}", flush=True)
            print(flush=True)
        print("If the images differ between runs, the rules prove model is generating fresh.", flush=True)
    else:
        raise SystemExit(main())


