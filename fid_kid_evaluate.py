import argparse
import os
import random

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
import torchvision.transforms as Transforms
from torchvision.models.inception import inception_v3
from scipy import linalg


# ─────────────────────────────────────────────────────────
# Argument parsing
# ─────────────────────────────────────────────────────────

def get_opt():
    parser = argparse.ArgumentParser(
        description="Evaluate virtual try-on results using FID and KID metrics."
    )
    parser.add_argument(
        '--predict_dir',
        default='./result/lightweight/output/',
        help='Directory containing generated/predicted images'
    )
    parser.add_argument(
        '--ground_truth_dir',
        default='./data/test/image',
        help='Directory containing ground-truth images'
    )
    parser.add_argument(
        '--resolution',
        type=int,
        default=1024,
        choices=[256, 512, 1024],
        help='Image resolution (must match generator output)'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=32,
        help='Batch size for InceptionV3 feature extraction'
    )
    parser.add_argument(
        '--kid_subsets',
        type=int,
        default=100,
        help='Number of random subsets for KID estimation'
    )
    parser.add_argument(
        '--kid_subset_size',
        type=int,
        default=1000,
        help='Size of each random subset for KID estimation'
    )
    parser.add_argument(
        '--gpu_ids',
        default='0',
        help='GPU id(s) to use, e.g. "0". Pass "" for CPU.'
    )
    opt = parser.parse_args()
    return opt


# ─────────────────────────────────────────────────────────
# Device helper
# ─────────────────────────────────────────────────────────

def get_device(gpu_ids: str) -> torch.device:
    if gpu_ids and torch.cuda.is_available():
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_ids
        device = torch.device("cuda")
        print(f"[Device] Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("[Device] Using CPU")
    return device


# ─────────────────────────────────────────────────────────
# Image transform (InceptionV3 expects 299×299, [-1, 1])
# ─────────────────────────────────────────────────────────

def get_inception_transform():
    return Transforms.Compose([
        Transforms.Resize((299, 299)),
        Transforms.ToTensor(),
        Transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ])


# ─────────────────────────────────────────────────────────
# Feature extraction
# ─────────────────────────────────────────────────────────

def extract_inception_features(image_paths: list, model, transform, device, batch_size: int) -> np.ndarray:
    """
    Run images through InceptionV3 (up to the pool3 layer) and return
    a (N, 2048) numpy array of activations.
    """
    model.eval()
    all_features = []

    for start in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[start: start + batch_size]
        imgs = []
        for p in batch_paths:
            img = Image.open(p).convert('RGB')
            imgs.append(transform(img))
        batch = torch.stack(imgs, dim=0).to(device)   # .to(device) ← GPU-aware

        with torch.no_grad():
            # InceptionV3 forward until avgpool (pool3 = 2048-d)
            feats = inception_forward_pool3(model, batch)  # (B, 2048)
        all_features.append(feats.cpu().numpy())

        print(f"  Extracted features: {min(start + batch_size, len(image_paths))}/{len(image_paths)}", end='\r')

    print()
    return np.concatenate(all_features, axis=0)  # (N, 2048)


def inception_forward_pool3(model, x):
    """
    Run InceptionV3 forward pass and return pool3 (2048-d) features
    without the classification head.
    Works with torchvision's InceptionV3 (transform_input=False).
    """
    # Upsample if needed
    if x.shape[-1] != 299 or x.shape[-2] != 299:
        x = F.interpolate(x, size=(299, 299), mode='bilinear', align_corners=False)

    # InceptionV3 internal layers (torchvision implementation)
    x = model.Conv2d_1a_3x3(x)
    x = model.Conv2d_2a_3x3(x)
    x = model.Conv2d_2b_3x3(x)
    x = F.max_pool2d(x, kernel_size=3, stride=2)
    x = model.Conv2d_3b_1x1(x)
    x = model.Conv2d_4a_3x3(x)
    x = F.max_pool2d(x, kernel_size=3, stride=2)
    x = model.Mixed_5b(x)
    x = model.Mixed_5c(x)
    x = model.Mixed_5d(x)
    x = model.Mixed_6a(x)
    x = model.Mixed_6b(x)
    x = model.Mixed_6c(x)
    x = model.Mixed_6d(x)
    x = model.Mixed_6e(x)
    x = model.Mixed_7a(x)
    x = model.Mixed_7b(x)
    x = model.Mixed_7c(x)
    x = F.adaptive_avg_pool2d(x, (1, 1))   # pool3
    x = x.view(x.size(0), -1)              # (B, 2048)
    return x


# ─────────────────────────────────────────────────────────
# FID
# ─────────────────────────────────────────────────────────

def compute_statistics(features: np.ndarray):
    """Return (mu, sigma) of feature distribution."""
    mu = np.mean(features, axis=0)
    sigma = np.cov(features, rowvar=False)
    return mu, sigma


def compute_fid(mu1, sigma1, mu2, sigma2, eps: float = 1e-6) -> float:
    """
    Fréchet Inception Distance.
    Formula: ||mu1 - mu2||^2 + Tr(sigma1 + sigma2 - 2*sqrt(sigma1*sigma2))
    """
    diff = mu1 - mu2
    # Product of covariances → matrix square root
    covmean, _ = linalg.sqrtm(sigma1 @ sigma2, disp=False)
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset) @ (sigma2 + offset))

    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            raise ValueError(f"Imaginary component of covmean too large: {np.max(np.abs(covmean.imag))}")
        covmean = covmean.real

    fid = diff @ diff + np.trace(sigma1) + np.trace(sigma2) - 2 * np.trace(covmean)
    return float(fid)


# ─────────────────────────────────────────────────────────
# KID
# ─────────────────────────────────────────────────────────

def polynomial_mmd(X: np.ndarray, Y: np.ndarray, degree: int = 3,
                   gamma: float = None, coef0: float = 1.0) -> float:
    """
    Maximum Mean Discrepancy with a polynomial kernel.
    X, Y: (n, d) feature matrices.
    """
    n = X.shape[0]
    m = Y.shape[0]
    if gamma is None:
        gamma = 1.0 / X.shape[1]

    def poly_kernel(A, B):
        return (gamma * (A @ B.T) + coef0) ** degree

    K_XX = poly_kernel(X, X)
    K_YY = poly_kernel(Y, Y)
    K_XY = poly_kernel(X, Y)

    mmd = (
        (K_XX.sum() - np.trace(K_XX)) / (n * (n - 1))
        + (K_YY.sum() - np.trace(K_YY)) / (m * (m - 1))
        - 2 * K_XY.mean()
    )
    return float(mmd)


def compute_kid(real_feats: np.ndarray, fake_feats: np.ndarray,
                n_subsets: int = 100, subset_size: int = 1000) -> tuple:
    """
    Kernel Inception Distance via random subset MMD averaging.
    Returns (kid_mean, kid_std).
    """
    subset_size = min(subset_size, len(real_feats), len(fake_feats))
    mmds = []
    for _ in range(n_subsets):
        real_idx = np.random.choice(len(real_feats), subset_size, replace=False)
        fake_idx = np.random.choice(len(fake_feats), subset_size, replace=False)
        mmd = polynomial_mmd(real_feats[real_idx], fake_feats[fake_idx])
        mmds.append(mmd)
    return float(np.mean(mmds)), float(np.std(mmds))


# ─────────────────────────────────────────────────────────
# Main evaluation function
# ─────────────────────────────────────────────────────────

def Evaluation(opt):
    device = get_device(opt.gpu_ids)

    # ── collect paired file lists ──────────────────────────
    pred_files = sorted([f for f in os.listdir(opt.predict_dir) if f.endswith(('.jpg', '.png'))])
    if len(pred_files) == 0:
        raise FileNotFoundError(f"No images found in: {opt.predict_dir}")

    pred_paths = []
    gt_paths   = []

    for img_pred in pred_files:
        # Predicted filename pattern: <person>_<cloth>.png  (same as evaluate.py)
        gt_name = img_pred.split('_')[0] + '_00.jpg'
        gt_path = os.path.join(opt.ground_truth_dir, gt_name)
        if not os.path.exists(gt_path):
            print(f"[WARN] Ground-truth not found, skipping: {gt_path}")
            continue
        pred_paths.append(os.path.join(opt.predict_dir, img_pred))
        gt_paths.append(gt_path)

    print(f"[Info] Matched pairs: {len(pred_paths)}")

    # ── apply resolution scaling to GT (same as evaluate.py) ──
    transform = get_inception_transform()

    # ── build InceptionV3 (pool3 features, no head) ──────────
    print("[Model] Loading InceptionV3 ...")
    inception = inception_v3(weights='IMAGENET1K_V1', transform_input=False)
    inception = inception.to(device)        # .to(device) ── GPU-aware
    inception.eval()

    # ── extract features ──────────────────────────────────────
    def load_resized(path):
        """Load image and apply optional resolution resize (mirrors evaluate.py logic)."""
        img = Image.open(path).convert('RGB')
        if opt.resolution != 1024:
            if opt.resolution == 512:
                img = img.resize((384, 512), Image.BILINEAR)
            elif opt.resolution == 256:
                img = img.resize((192, 256), Image.BILINEAR)
        return img

    # Build path lists with resize applied before transform
    print("[Features] Extracting real (GT) features ...")
    real_features = _extract_features_from_paths(gt_paths,   inception, transform, device, opt.batch_size, load_resized)
    print("[Features] Extracting fake (predicted) features ...")
    fake_features = _extract_features_from_paths(pred_paths, inception, transform, device, opt.batch_size, load_resized)

    # ── FID ──────────────────────────────────────────────────
    print("[Metric] Computing FID ...")
    mu_r, sigma_r = compute_statistics(real_features)
    mu_f, sigma_f = compute_statistics(fake_features)
    fid_score = compute_fid(mu_r, sigma_r, mu_f, sigma_f)

    # ── KID ──────────────────────────────────────────────────
    print("[Metric] Computing KID ...")
    kid_mean, kid_std = compute_kid(
        real_features, fake_features,
        n_subsets=opt.kid_subsets,
        subset_size=opt.kid_subset_size,
    )

    # ── save results ─────────────────────────────────────────
    result_path = os.path.join(opt.predict_dir, 'fid_kid_eval.txt')
    with open(result_path, 'a') as f:
        f.write(f"Pairs evaluated : {len(pred_paths)}\n")
        f.write(f"FID             : {fid_score:.4f}\n")
        f.write(f"KID mean        : {kid_mean:.6f}\n")
        f.write(f"KID std         : {kid_std:.6f}\n")
    print(f"[Saved] Results written to {result_path}")

    return fid_score, kid_mean, kid_std


def _extract_features_from_paths(paths, model, transform, device, batch_size, load_fn):
    """Helper: load images via load_fn, apply transform, extract InceptionV3 pool3 feats."""
    all_features = []
    for start in range(0, len(paths), batch_size):
        batch_paths = paths[start: start + batch_size]
        imgs = []
        for p in batch_paths:
            img = load_fn(p)
            imgs.append(transform(img))
        batch = torch.stack(imgs, dim=0).to(device)   # .to(device) ── GPU-aware

        with torch.no_grad():
            feats = inception_forward_pool3(model, batch)  # (B, 2048)
        all_features.append(feats.cpu().numpy())

        done = min(start + batch_size, len(paths))
        print(f"  {done}/{len(paths)}", end='\r')

    print()
    return np.concatenate(all_features, axis=0)


# ─────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────

def main():
    opt = get_opt()
    print(opt)
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_ids

    fid, kid_mean, kid_std = Evaluation(opt)

    print("=" * 45)
    print(f"  FID       : {fid:.4f}")
    print(f"  KID mean  : {kid_mean:.6f}")
    print(f"  KID std   : {kid_std:.6f}")
    print("=" * 45)


if __name__ == '__main__':
    main()
