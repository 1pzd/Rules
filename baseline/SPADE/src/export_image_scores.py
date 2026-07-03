import argparse
import csv
import os
import pickle
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.models import wide_resnet50_2
from tqdm import tqdm

import datasets.mvtec as mvtec
from main import calc_dist_matrix


def default_mvtec_path():
    return str(Path(__file__).resolve().parents[3] / 'mvtec_anomaly_detection')


def parse_args():
    parser = argparse.ArgumentParser(
        'Export SPADE image-level anomaly scores to CSV'
    )
    parser.add_argument('--data_path', default=default_mvtec_path(),
                        help='Path to mvtec_anomaly_detection or its parent directory.')
    parser.add_argument('--class_name', default='bottle', choices=mvtec.CLASS_NAMES)
    parser.add_argument('--save_path', default='./result')
    parser.add_argument('--output_csv', default=None)
    parser.add_argument('--top_k', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--threshold', type=float, default=None,
                        help='Manual anomaly threshold. If omitted, uses best F1 threshold from labeled test data.')
    return parser.parse_args()


def mvtec_parent_path(data_path):
    path = Path(data_path).expanduser().resolve()
    if path.name == 'mvtec_anomaly_detection':
        return str(path.parent)
    return str(path)


def choose_threshold(labels, scores, manual_threshold):
    if manual_threshold is not None:
        return float(manual_threshold), 'manual'

    labels = np.asarray(labels).astype(int)
    scores = np.asarray(scores, dtype=np.float64)
    if np.unique(labels).size < 2:
        raise ValueError(
            'Cannot infer threshold because test labels contain only one class. '
            'Pass --threshold explicitly for unlabeled or one-class test data.'
        )

    candidates = np.unique(scores)
    if candidates.size == 0:
        raise ValueError('Cannot infer threshold from empty scores.')

    best_threshold = float(candidates[0])
    best_f1 = -1.0
    for threshold in candidates:
        predictions = scores > threshold
        true_positive = np.logical_and(predictions, labels == 1).sum()
        false_positive = np.logical_and(predictions, labels == 0).sum()
        false_negative = np.logical_and(~predictions, labels == 1).sum()
        precision = (
            true_positive / (true_positive + false_positive)
            if true_positive + false_positive else 0.0
        )
        recall = (
            true_positive / (true_positive + false_negative)
            if true_positive + false_negative else 0.0
        )
        f1_score = (
            2 * precision * recall / (precision + recall)
            if precision + recall else 0.0
        )
        if f1_score > best_f1:
            best_f1 = f1_score
            best_threshold = float(threshold)
    return best_threshold, 'test_f1'


def build_model(device):
    model = wide_resnet50_2(pretrained=True, progress=True).to(device)
    model.eval()
    return model


def extract_avgpool_features(model, loader, device):
    outputs = []

    def hook(module, input_tensor, output_tensor):
        outputs.append(output_tensor.detach().cpu())

    handle = model.avgpool.register_forward_hook(hook)
    try:
        for batch in tqdm(loader, desc='extract avgpool features'):
            x = batch[0].to(device)
            with torch.no_grad():
                model(x)
    finally:
        handle.remove()

    return torch.cat(outputs, dim=0)


def load_or_create_train_features(model, dataset, batch_size, device, cache_path):
    if os.path.exists(cache_path):
        with open(cache_path, 'rb') as f:
            return pickle.load(f)

    loader = DataLoader(dataset, batch_size=batch_size, pin_memory=True)
    features = extract_avgpool_features(model, loader, device)
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, 'wb') as f:
        pickle.dump(features, f)
    return features


def write_csv(output_csv, class_name, image_paths, labels, scores, threshold, threshold_source):
    os.makedirs(os.path.dirname(os.path.abspath(output_csv)), exist_ok=True)
    with open(output_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                'class_name',
                'filename',
                'anomaly_score',
                'threshold',
                'threshold_source',
                'pred_label',
                'gt_label',
                'image_path',
            ],
        )
        writer.writeheader()
        for image_path, label, score in zip(image_paths, labels, scores):
            score_value = float(score)
            writer.writerow({
                'class_name': class_name,
                'filename': os.path.basename(image_path),
                'anomaly_score': f'{score_value:.8f}',
                'threshold': f'{threshold:.8f}',
                'threshold_source': threshold_source,
                'pred_label': 'defect' if score_value > threshold else 'normal',
                'gt_label': int(label),
                'image_path': image_path,
            })


def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    root_path = mvtec_parent_path(args.data_path)

    train_dataset = mvtec.MVTecDataset(
        root_path=root_path,
        class_name=args.class_name,
        is_train=True,
    )
    test_dataset = mvtec.MVTecDataset(
        root_path=root_path,
        class_name=args.class_name,
        is_train=False,
    )
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, pin_memory=True)

    model = build_model(device)
    cache_path = os.path.join(
        args.save_path,
        'temp_image',
        f'train_{args.class_name}_avgpool.pkl',
    )
    train_features = load_or_create_train_features(
        model, train_dataset, args.batch_size, device, cache_path
    )
    test_features = extract_avgpool_features(model, test_loader, device)

    dist_matrix = calc_dist_matrix(
        torch.flatten(test_features, 1),
        torch.flatten(train_features, 1),
    )
    topk_values, _ = torch.topk(
        dist_matrix,
        k=args.top_k,
        dim=1,
        largest=False,
    )
    scores = torch.mean(topk_values, 1).cpu().detach().numpy()
    labels = np.asarray(test_dataset.y, dtype=int)
    threshold, threshold_source = choose_threshold(labels, scores, args.threshold)

    output_csv = args.output_csv or os.path.join(
        args.save_path,
        f'image_scores_{args.class_name}.csv',
    )
    write_csv(
        output_csv,
        args.class_name,
        test_dataset.x,
        labels,
        scores,
        threshold,
        threshold_source,
    )
    print(f'Wrote {len(scores)} rows to {output_csv}')
    print(f'Threshold: {threshold:.8f} ({threshold_source})')


if __name__ == '__main__':
    main()
