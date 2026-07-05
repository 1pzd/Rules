import argparse
import csv
import json
import os
import pickle
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy.ndimage import gaussian_filter
from scipy.spatial.distance import mahalanobis
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T
from torchvision.models import (
    ResNet18_Weights,
    Wide_ResNet50_2_Weights,
    resnet18,
    wide_resnet50_2,
)
from tqdm import tqdm


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EVALUATION_JSON = ROOT / "evaluation_results.json"
DEFAULT_OUTPUT_DIR = ROOT / "baseline_results"
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".JPG"}


@dataclass(frozen=True)
class EvalRecord:
    index: int
    dataset: str
    category: str
    source: str
    image_path: Path
    expected: str


class PathImageDataset(Dataset):
    def __init__(self, image_paths, resize=256, cropsize=224):
        self.image_paths = [Path(path) for path in image_paths]
        self.resize = resize
        self.cropsize = cropsize
        self.transform = T.Compose([
            T.Resize(resize, Image.Resampling.LANCZOS),
            T.CenterCrop(cropsize),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image = Image.open(self.image_paths[index]).convert("RGB")
        return self.transform(image)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate PaDiM and SPADE on the exact 127 evaluation_results.json samples."
    )
    parser.add_argument("--evaluation-json", type=Path, default=DEFAULT_EVALUATION_JSON)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--method", choices=["both", "padim", "spade"], default="both")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--weights", choices=["imagenet", "random"], default="imagenet")
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args()


def load_records(evaluation_json, limit=None):
    with open(evaluation_json, "r", encoding="utf-8") as f:
        payload = json.load(f)

    records = []
    for index, item in enumerate(payload["results"]):
        image_path = Path(item["image_path"])
        expected = item["expected"]
        if expected not in {"normal", "abnormal"}:
            raise ValueError(f"Unsupported expected label at row {index}: {expected}")
        records.append(EvalRecord(
            index=index,
            dataset=item["dataset"],
            category=item["category"],
            source=item["source"],
            image_path=image_path,
            expected=expected,
        ))

    if limit is not None:
        records = records[:limit]

    missing = [str(record.image_path) for record in records if not record.image_path.exists()]
    if missing:
        preview = "\n".join(missing[:10])
        raise FileNotFoundError(f"Missing {len(missing)} target image files:\n{preview}")
    return records


def list_images(directory):
    directory = Path(directory)
    if not directory.exists():
        return []
    return sorted(
        path for path in directory.iterdir()
        if path.is_file() and path.suffix in IMAGE_EXTENSIONS
    )


def train_good_images(dataset, category):
    if dataset == "mvtec_ad":
        paths = list_images(ROOT / "mvtec_anomaly_detection" / category / "train" / "good")
    elif dataset == "mvtec_loco":
        paths = list_images(ROOT / "mvtec_loco_anomaly_detection" / category / "train" / "good")
    elif dataset == "visa":
        paths = visa_train_normal_images(category)
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")

    if not paths:
        raise FileNotFoundError(f"No train-good images found for {dataset}/{category}")
    return paths


def visa_train_normal_images(category):
    visa_root = ROOT / "VisA_20220922" / "VisA_20220922"
    split_csv = visa_root / "split_csv" / "1cls.csv"
    paths = []
    if split_csv.exists():
        with open(split_csv, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row["object"] == category and row["split"] == "train" and row["label"] == "normal":
                    paths.append(visa_root / row["image"])
    if not paths:
        paths = list_images(visa_root / category / "Data" / "Images" / "Normal")
    missing = [path for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"VisA split lists missing files for {category}: {missing[:3]}")
    return sorted(paths)


def grouped_records(records):
    groups = defaultdict(list)
    for record in records:
        groups[(record.dataset, record.category)].append(record)
    return dict(groups)


def build_loader(paths, batch_size, device):
    dataset = PathImageDataset(paths)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, pin_memory=device.type == "cuda")


def build_resnet18(weights_mode, device):
    try:
        weights = ResNet18_Weights.DEFAULT if weights_mode == "imagenet" else None
        model = resnet18(weights=weights).to(device)
    except Exception as exc:
        raise RuntimeError(
            "Could not load ResNet18 weights. Re-run with --weights random, or make sure "
            "torchvision ImageNet weights are available locally."
        ) from exc
    model.eval()
    return model


def build_wide_resnet(weights_mode, device):
    try:
        weights = Wide_ResNet50_2_Weights.DEFAULT if weights_mode == "imagenet" else None
        model = wide_resnet50_2(weights=weights).to(device)
    except Exception as exc:
        raise RuntimeError(
            "Could not load Wide ResNet50-2 weights. Re-run with --weights random, or make sure "
            "torchvision ImageNet weights are available locally."
        ) from exc
    model.eval()
    return model


def embedding_concat(x, y):
    batch, channels_1, height_1, width_1 = x.size()
    _, channels_2, height_2, width_2 = y.size()
    stride = int(height_1 / height_2)
    x = F.unfold(x, kernel_size=stride, dilation=1, stride=stride)
    x = x.view(batch, channels_1, -1, height_2, width_2)
    z = torch.zeros(batch, channels_1 + channels_2, x.size(2), height_2, width_2)
    for i in range(x.size(2)):
        z[:, :, i, :, :] = torch.cat((x[:, :, i, :, :], y), 1)
    z = z.view(batch, -1, height_2 * width_2)
    return F.fold(z, kernel_size=stride, output_size=(height_1, width_1), stride=stride)


def padim_index():
    random.seed(1024)
    torch.manual_seed(1024)
    return torch.tensor(random.sample(range(0, 448), 100))


def extract_padim_embeddings(model, paths, batch_size, device, idx, desc):
    outputs = {"layer1": [], "layer2": [], "layer3": []}

    def make_hook(name):
        def hook(_module, _input, output):
            outputs[name].append(output.detach().cpu())
        return hook

    handles = [
        model.layer1[-1].register_forward_hook(make_hook("layer1")),
        model.layer2[-1].register_forward_hook(make_hook("layer2")),
        model.layer3[-1].register_forward_hook(make_hook("layer3")),
    ]
    try:
        for batch in tqdm(build_loader(paths, batch_size, device), desc=desc):
            with torch.no_grad():
                model(batch.to(device))
    finally:
        for handle in handles:
            handle.remove()

    merged = torch.cat(outputs["layer1"], dim=0)
    for layer_name in ["layer2", "layer3"]:
        merged = embedding_concat(merged, torch.cat(outputs[layer_name], dim=0))
    return torch.index_select(merged, 1, idx)


def padim_distribution(train_embeddings):
    batch_size, channels, height, width = train_embeddings.size()
    vectors = train_embeddings.view(batch_size, channels, height * width)
    mean = torch.mean(vectors, dim=0).numpy()
    cov = torch.zeros(channels, channels, height * width).numpy()
    identity = np.identity(channels)
    for i in range(height * width):
        cov[:, :, i] = np.cov(vectors[:, :, i].numpy(), rowvar=False) + 0.01 * identity
    return mean, cov


def padim_scores_from_distribution(target_embeddings, distribution):
    batch_size, channels, height, width = target_embeddings.size()
    vectors = target_embeddings.view(batch_size, channels, height * width).numpy()
    mean, cov = distribution
    score_maps = []
    for i in range(height * width):
        try:
            cov_inv = np.linalg.inv(cov[:, :, i])
        except np.linalg.LinAlgError:
            cov_inv = np.linalg.pinv(cov[:, :, i])
        distances = [mahalanobis(sample_vector[:, i], mean[:, i], cov_inv) for sample_vector in vectors]
        score_maps.append(distances)

    score_maps = np.asarray(score_maps).T.reshape(batch_size, height, width)
    score_maps = torch.tensor(score_maps)
    score_maps = F.interpolate(score_maps.unsqueeze(1), size=256, mode="bilinear", align_corners=False)
    score_maps = score_maps.squeeze(1).numpy()
    for i in range(score_maps.shape[0]):
        score_maps[i] = gaussian_filter(score_maps[i], sigma=4)

    min_score = float(score_maps.min())
    max_score = float(score_maps.max())
    if max_score == min_score:
        normalized = np.zeros_like(score_maps)
    else:
        normalized = (score_maps - min_score) / (max_score - min_score)
    return normalized.reshape(normalized.shape[0], -1).max(axis=1)


def run_padim(records, args, device):
    model = build_resnet18(args.weights, device)
    idx = padim_index()
    rows = []
    cache_dir = args.output_dir / "cache" / "padim"
    cache_dir.mkdir(parents=True, exist_ok=True)

    for (dataset, category), group in grouped_records(records).items():
        train_paths = train_good_images(dataset, category)
        cache_path = cache_dir / f"{dataset}_{category}_resnet18.pkl"
        if cache_path.exists():
            with open(cache_path, "rb") as f:
                distribution = pickle.load(f)
        else:
            train_embeddings = extract_padim_embeddings(
                model, train_paths, args.batch_size, device, idx, f"PaDiM train {dataset}/{category}"
            )
            distribution = padim_distribution(train_embeddings)
            with open(cache_path, "wb") as f:
                pickle.dump(distribution, f)

        target_paths = [record.image_path for record in group]
        target_embeddings = extract_padim_embeddings(
            model, target_paths, args.batch_size, device, idx, f"PaDiM target {dataset}/{category}"
        )
        scores = padim_scores_from_distribution(target_embeddings, distribution)
        rows.extend(scored_rows("PaDiM", group, scores))
    return rows


def extract_spade_features(model, paths, batch_size, device, desc):
    outputs = []

    def hook(_module, _input, output):
        outputs.append(output.detach().cpu())

    handle = model.avgpool.register_forward_hook(hook)
    try:
        for batch in tqdm(build_loader(paths, batch_size, device), desc=desc):
            with torch.no_grad():
                model(batch.to(device))
    finally:
        handle.remove()
    return torch.cat(outputs, dim=0)


def run_spade(records, args, device):
    model = build_wide_resnet(args.weights, device)
    rows = []
    cache_dir = args.output_dir / "cache" / "spade"
    cache_dir.mkdir(parents=True, exist_ok=True)

    for (dataset, category), group in grouped_records(records).items():
        train_paths = train_good_images(dataset, category)
        cache_path = cache_dir / f"{dataset}_{category}_wide_resnet50_2_avgpool.pkl"
        if cache_path.exists():
            with open(cache_path, "rb") as f:
                train_features = pickle.load(f)
        else:
            train_features = extract_spade_features(
                model, train_paths, args.batch_size, device, f"SPADE train {dataset}/{category}"
            )
            with open(cache_path, "wb") as f:
                pickle.dump(train_features, f)

        target_paths = [record.image_path for record in group]
        target_features = extract_spade_features(
            model, target_paths, args.batch_size, device, f"SPADE target {dataset}/{category}"
        )
        distances = torch.cdist(torch.flatten(target_features, 1), torch.flatten(train_features, 1))
        k = min(args.top_k, distances.size(1))
        topk_values, _ = torch.topk(distances, k=k, dim=1, largest=False)
        scores = torch.mean(topk_values, dim=1).cpu().detach().numpy()
        rows.extend(scored_rows("SPADE", group, scores))
    return rows


def scored_rows(method, records, scores):
    labels = np.asarray([1 if record.expected == "abnormal" else 0 for record in records], dtype=int)
    threshold, threshold_source = choose_threshold(labels, scores)
    rows = []
    for record, score in zip(records, scores):
        score_value = float(score)
        prediction = "abnormal" if score_value > threshold else "normal"
        rows.append({
            "method": method,
            "index": record.index,
            "dataset": record.dataset,
            "category": record.category,
            "source": record.source,
            "image_path": str(record.image_path),
            "expected": record.expected,
            "anomaly_score": f"{score_value:.8f}",
            "threshold": f"{threshold:.8f}",
            "threshold_source": threshold_source,
            "prediction": prediction,
            "correct": str(prediction == record.expected).lower(),
        })
    return rows


def choose_threshold(labels, scores):
    labels = np.asarray(labels).astype(int)
    scores = np.asarray(scores, dtype=np.float64)
    if scores.size == 0:
        raise ValueError("Cannot infer threshold from empty scores.")

    if np.unique(labels).size < 2:
        normal_scores = scores[labels == 0]
        if normal_scores.size:
            return float(normal_scores.max()), "target_normal_max"
        return float(np.median(scores)), "target_median_fallback"

    best_threshold = float(np.unique(scores)[0])
    best_f1 = -1.0
    for threshold in np.unique(scores):
        predictions = scores > threshold
        true_positive = np.logical_and(predictions, labels == 1).sum()
        false_positive = np.logical_and(predictions, labels == 0).sum()
        false_negative = np.logical_and(~predictions, labels == 1).sum()
        precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
        recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
        f1_score = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        if f1_score > best_f1:
            best_f1 = f1_score
            best_threshold = float(threshold)
    return best_threshold, "target_f1"


def write_method_csv(output_csv, rows):
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "method", "index", "dataset", "category", "source", "image_path", "expected",
        "anomaly_score", "threshold", "threshold_source", "prediction", "correct",
    ]
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize(records, method_outputs):
    summary = {
        "source_evaluation_file": str(DEFAULT_EVALUATION_JSON),
        "input_total": len(records),
        "threshold_policy": "per_method_dataset_category_target_f1",
        "methods": {},
    }
    for method, output in method_outputs.items():
        rows = output["rows"]
        correct = sum(1 for row in rows if row["correct"] == "true")
        summary["methods"][method] = {
            "total": len(rows),
            "correct": correct,
            "accuracy": correct / len(rows) if rows else 0.0,
            "output_csv": str(output["csv"]),
        }
    return summary


def validate_outputs(records, method_outputs):
    expected_paths = sorted(str(record.image_path.resolve()) for record in records)
    for method, output in method_outputs.items():
        csv_path = output["csv"]
        if not csv_path.exists():
            raise FileNotFoundError(f"Missing {method} output CSV: {csv_path}")
        with open(csv_path, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        if len(rows) != len(records):
            raise AssertionError(f"{method} row count {len(rows)} != expected {len(records)}")
        actual_paths = sorted(str(Path(row["image_path"]).resolve()) for row in rows)
        if actual_paths != expected_paths:
            raise AssertionError(f"{method} image_path coverage does not match evaluation_results.json")
        for row in rows:
            if row["expected"] not in {"normal", "abnormal"}:
                raise AssertionError(f"{method} invalid expected label: {row}")
            if row["prediction"] not in {"normal", "abnormal"}:
                raise AssertionError(f"{method} invalid prediction label: {row}")
            correct = str(row["prediction"] == row["expected"]).lower()
            if row["correct"] != correct:
                raise AssertionError(f"{method} incorrect correctness flag: {row}")


def output_csv_for(method, output_dir):
    name = "padim_127_image_scores.csv" if method == "PaDiM" else "spade_127_image_scores.csv"
    return output_dir / name


def selected_methods(method_arg):
    if method_arg == "both":
        return ["PaDiM", "SPADE"]
    if method_arg == "padim":
        return ["PaDiM"]
    return ["SPADE"]


def read_existing_method_outputs(methods, output_dir):
    return {method: {"csv": output_csv_for(method, output_dir), "rows": []} for method in methods}


def main():
    args = parse_args()
    records = load_records(args.evaluation_json, args.limit)
    methods = selected_methods(args.method)
    method_outputs = read_existing_method_outputs(methods, args.output_dir)

    if args.validate_only:
        validate_outputs(records, method_outputs)
        print(f"Validated {len(records)} rows for {', '.join(methods)}")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if "PaDiM" in methods:
        rows = run_padim(records, args, device)
        csv_path = output_csv_for("PaDiM", args.output_dir)
        write_method_csv(csv_path, rows)
        method_outputs["PaDiM"] = {"csv": csv_path, "rows": rows}
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if "SPADE" in methods:
        rows = run_spade(records, args, device)
        csv_path = output_csv_for("SPADE", args.output_dir)
        write_method_csv(csv_path, rows)
        method_outputs["SPADE"] = {"csv": csv_path, "rows": rows}
        if device.type == "cuda":
            torch.cuda.empty_cache()

    summary = summarize(records, method_outputs)
    summary_path = args.output_dir / "baseline_127_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    validate_outputs(records, method_outputs)
    print(f"Wrote summary to {summary_path}")
    for method, output in method_outputs.items():
        stats = summary["methods"][method]
        print(f"{method}: {stats['correct']}/{stats['total']} accuracy={stats['accuracy']:.4f} csv={output['csv']}")


if __name__ == "__main__":
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    main()
