import argparse
import csv
import os
import pickle
from collections import OrderedDict
from pathlib import Path
from random import sample

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter
from scipy.spatial.distance import mahalanobis
from torch.utils.data import DataLoader
from torchvision.models import resnet18, wide_resnet50_2
from tqdm import tqdm

import datasets.mvtec as mvtec
from main import embedding_concat


def default_mvtec_path():
    return str(Path(__file__).resolve().parents[2] / 'mvtec_anomaly_detection')


def parse_args():
    parser = argparse.ArgumentParser(
        'Export PaDiM image-level anomaly scores to CSV'
    )
    parser.add_argument('--data_path', default=default_mvtec_path(),
                        help='Path to mvtec_anomaly_detection.')
    parser.add_argument('--class_name', default='bottle', choices=mvtec.CLASS_NAMES)
    parser.add_argument('--save_path', default='./mvtec_result')
    parser.add_argument('--output_csv', default=None)
    parser.add_argument('--arch', default='resnet18', choices=['resnet18', 'wide_resnet50_2'])
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--threshold', type=float, default=None,
                        help='Manual anomaly threshold. If omitted, uses best F1 threshold from labeled test data.')
    return parser.parse_args()


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


def build_model_and_index(arch, device):
    if arch == 'resnet18':
        model = resnet18(pretrained=True, progress=True).to(device)
        total_dims = 448
        selected_dims = 100
    else:
        model = wide_resnet50_2(pretrained=True, progress=True).to(device)
        total_dims = 1792
        selected_dims = 550

    model.eval()
    np.random.seed(1024)
    torch.manual_seed(1024)
    idx = torch.tensor(sample(range(0, total_dims), selected_dims))
    return model, idx


def extract_layer_features(model, loader, device):
    outputs = OrderedDict([('layer1', []), ('layer2', []), ('layer3', [])])

    def make_hook(name):
        def hook(module, input_tensor, output_tensor):
            outputs[name].append(output_tensor.detach().cpu())
        return hook

    handles = [
        model.layer1[-1].register_forward_hook(make_hook('layer1')),
        model.layer2[-1].register_forward_hook(make_hook('layer2')),
        model.layer3[-1].register_forward_hook(make_hook('layer3')),
    ]
    try:
        for batch in tqdm(loader, desc='extract layer features'):
            x = batch[0].to(device)
            with torch.no_grad():
                model(x)
    finally:
        for handle in handles:
            handle.remove()

    for key in outputs:
        outputs[key] = torch.cat(outputs[key], dim=0)
    return outputs


def merge_selected_embedding(outputs, idx):
    embeddings = outputs['layer1']
    for layer_name in ['layer2', 'layer3']:
        embeddings = embedding_concat(embeddings, outputs[layer_name])
    return torch.index_select(embeddings, 1, idx)


def load_or_create_distribution(model, dataset, batch_size, idx, device, cache_path):
    if os.path.exists(cache_path):
        with open(cache_path, 'rb') as f:
            return pickle.load(f)

    loader = DataLoader(dataset, batch_size=batch_size, pin_memory=True)
    outputs = extract_layer_features(model, loader, device)
    embeddings = merge_selected_embedding(outputs, idx)
    batch_size_value, channels, height, width = embeddings.size()
    embeddings = embeddings.view(batch_size_value, channels, height * width)

    mean = torch.mean(embeddings, dim=0).numpy()
    cov = torch.zeros(channels, channels, height * width).numpy()
    identity = np.identity(channels)
    for i in range(height * width):
        cov[:, :, i] = np.cov(embeddings[:, :, i].numpy(), rowvar=False) + 0.01 * identity

    distribution = [mean, cov]
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, 'wb') as f:
        pickle.dump(distribution, f)
    return distribution


def compute_image_scores(model, dataset, batch_size, idx, device, distribution):
    loader = DataLoader(dataset, batch_size=batch_size, pin_memory=True)
    outputs = extract_layer_features(model, loader, device)
    embeddings = merge_selected_embedding(outputs, idx)
    batch_size_value, channels, height, width = embeddings.size()
    embeddings = embeddings.view(batch_size_value, channels, height * width).numpy()

    mean, cov = distribution
    score_maps = []
    for i in range(height * width):
        cov_inv = np.linalg.inv(cov[:, :, i])
        distances = [
            mahalanobis(sample_vector[:, i], mean[:, i], cov_inv)
            for sample_vector in embeddings
        ]
        score_maps.append(distances)

    score_maps = np.asarray(score_maps).T.reshape(batch_size_value, height, width)
    score_maps = torch.tensor(score_maps)
    score_maps = F.interpolate(
        score_maps.unsqueeze(1),
        size=dataset.resize,
        mode='bilinear',
        align_corners=False,
    ).squeeze().numpy()

    for i in range(score_maps.shape[0]):
        score_maps[i] = gaussian_filter(score_maps[i], sigma=4)

    min_score = float(score_maps.min())
    max_score = float(score_maps.max())
    if max_score == min_score:
        normalized = np.zeros_like(score_maps)
    else:
        normalized = (score_maps - min_score) / (max_score - min_score)
    return normalized.reshape(normalized.shape[0], -1).max(axis=1)


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
    model, idx = build_model_and_index(args.arch, device)

    train_dataset = mvtec.MVTecDataset(
        args.data_path,
        class_name=args.class_name,
        is_train=True,
    )
    test_dataset = mvtec.MVTecDataset(
        args.data_path,
        class_name=args.class_name,
        is_train=False,
    )

    cache_path = os.path.join(
        args.save_path,
        f'temp_{args.arch}',
        f'train_{args.class_name}.pkl',
    )
    distribution = load_or_create_distribution(
        model,
        train_dataset,
        args.batch_size,
        idx,
        device,
        cache_path,
    )
    scores = compute_image_scores(
        model,
        test_dataset,
        args.batch_size,
        idx,
        device,
        distribution,
    )
    labels = np.asarray(test_dataset.y, dtype=int)
    threshold, threshold_source = choose_threshold(labels, scores, args.threshold)

    output_csv = args.output_csv or os.path.join(
        args.save_path,
        f'image_scores_{args.class_name}_{args.arch}.csv',
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
