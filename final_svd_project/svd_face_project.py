#!/usr/bin/env python3
"""
Final project code for:
Beyond Compression: Using SVD to Reconstruct, Denoise, and Compare Grayscale Face Images

This script loads the AT&T face dataset, performs truncated-SVD reconstruction,
adds Gaussian noise for denoising experiments, and evaluates nearest-neighbor
face matching with Frobenius/Euclidean distance.
"""
from __future__ import annotations

# Limit BLAS threads before importing NumPy to avoid very slow SVD runs on some systems.
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import argparse
import shutil
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

SUPPORTED_EXTENSIONS = {".pgm", ".png", ".jpg", ".jpeg", ".bmp", ".gif"}
DEFAULT_K_VALUES = [5, 10, 20, 30]
DEFAULT_NOISE_SIGMAS = [10.0, 20.0]


@dataclass
class Dataset:
    images: np.ndarray
    labels: np.ndarray
    paths: List[Path]


@dataclass
class PreparedDataset:
    root: Path
    cleanup_path: Path | None = None


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parent
    default_zip = project_root / "att_faces_dataset.zip"
    default_data = str(default_zip) if default_zip.exists() else None

    parser = argparse.ArgumentParser(
        description="SVD-based reconstruction, denoising, and comparison for grayscale face images."
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default=default_data,
        help="Path to the extracted dataset folder or the dataset .zip file.",
    )
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--k_values", type=int, nargs="+", default=DEFAULT_K_VALUES)
    parser.add_argument("--noise_sigmas", type=float, nargs="+", default=DEFAULT_NOISE_SIGMAS)
    parser.add_argument("--train_per_subject", type=int, default=6)
    parser.add_argument("--denoise_trials", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--max_subjects",
        type=int,
        default=0,
        help="Use only the first N subjects for a faster test run. Use 0 for all subjects.",
    )
    parser.add_argument(
        "--max_images_per_subject",
        type=int,
        default=0,
        help="Use only the first N images per subject. Use 0 for all images.",
    )
    parser.add_argument(
        "--resize_width",
        type=int,
        default=48,
        help="Resize width for faster SVD. Use 0 to keep original size.",
    )
    parser.add_argument(
        "--resize_height",
        type=int,
        default=48,
        help="Resize height for faster SVD. Use 0 to keep original size.",
    )
    return parser.parse_args()


def prepare_dataset(dataset_path: Path, extraction_dir: Path) -> PreparedDataset:
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset path not found: {dataset_path}. Put att_faces_dataset.zip next to the script or pass --dataset_path."
        )

    if dataset_path.is_dir():
        return PreparedDataset(root=dataset_path)

    if dataset_path.suffix.lower() != ".zip":
        raise ValueError("dataset_path must be a directory or a .zip file.")

    extraction_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(dataset_path, "r") as zf:
        zf.extractall(extraction_dir)
    return PreparedDataset(root=extraction_dir, cleanup_path=extraction_dir)


def read_grayscale_image(path: Path, resize_width: int = 0, resize_height: int = 0) -> np.ndarray:
    img = Image.open(path).convert("L")
    if resize_width > 0 and resize_height > 0:
        img = img.resize((resize_width, resize_height))
    return np.asarray(img, dtype=np.float64)


def subject_from_path(path: Path) -> int:
    folder = path.parent.name
    digits = "".join(ch for ch in folder if ch.isdigit())
    if not digits:
        raise ValueError(f"Could not infer subject id from folder name: {folder}")
    return int(digits)


def sort_key(path: Path) -> tuple[int, int]:
    stem_digits = "".join(ch for ch in path.stem if ch.isdigit())
    image_num = int(stem_digits) if stem_digits else 0
    return subject_from_path(path), image_num


def load_face_dataset(
    dataset_dir: Path,
    max_subjects: int = 0,
    max_images_per_subject: int = 0,
    resize_width: int = 0,
    resize_height: int = 0,
) -> Dataset:
    paths = [p for p in dataset_dir.rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS]
    if not paths:
        raise FileNotFoundError(f"No supported image files found under {dataset_dir}")

    paths = sorted(paths, key=sort_key)
    unique_subjects = sorted({subject_from_path(p) for p in paths})

    if max_subjects > 0:
        allowed_subjects = set(unique_subjects[:max_subjects])
        paths = [p for p in paths if subject_from_path(p) in allowed_subjects]

    if max_images_per_subject > 0:
        counts: dict[int, int] = {}
        filtered: list[Path] = []
        for path in paths:
            sid = subject_from_path(path)
            counts.setdefault(sid, 0)
            if counts[sid] < max_images_per_subject:
                filtered.append(path)
                counts[sid] += 1
        paths = filtered

    images: list[np.ndarray] = []
    labels: list[int] = []
    shape: tuple[int, int] | None = None

    for path in paths:
        arr = read_grayscale_image(path, resize_width=resize_width, resize_height=resize_height)
        if shape is None:
            shape = arr.shape
        elif arr.shape != shape:
            raise ValueError(f"All images must have the same size. Found {arr.shape} and {shape}.")
        images.append(arr)
        labels.append(subject_from_path(path))

    return Dataset(np.stack(images, axis=0), np.asarray(labels, dtype=int), paths)


def split_train_test_by_subject(
    labels: np.ndarray,
    train_per_subject: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    train_indices: list[int] = []
    test_indices: list[int] = []

    for label in np.unique(labels):
        idx = np.where(labels == label)[0]
        idx = rng.permutation(idx)
        if len(idx) <= train_per_subject:
            raise ValueError(
                f"Subject {label} has only {len(idx)} images, but train_per_subject={train_per_subject}."
            )
        train_indices.extend(idx[:train_per_subject])
        test_indices.extend(idx[train_per_subject:])

    return np.asarray(sorted(train_indices)), np.asarray(sorted(test_indices))


def svd_factors(image: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    return np.linalg.svd(image, full_matrices=False)


def precompute_svd(images: np.ndarray) -> list[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    return [svd_factors(img) for img in images]


def reconstruct_from_factors(factors: Tuple[np.ndarray, np.ndarray, np.ndarray], k: int) -> np.ndarray:
    U, s, Vt = factors
    k = int(np.clip(k, 1, len(s)))
    return (U[:, :k] * s[:k]) @ Vt[:k, :]


def reconstruct_many_from_factors(
    factor_list: Sequence[Tuple[np.ndarray, np.ndarray, np.ndarray]],
    k: int,
) -> np.ndarray:
    return np.stack([reconstruct_from_factors(factors, k) for factors in factor_list], axis=0)


def relative_frobenius_error(reference: np.ndarray, estimate: np.ndarray) -> float:
    denom = np.linalg.norm(reference, ord="fro")
    if denom == 0:
        return float(np.linalg.norm(reference - estimate, ord="fro"))
    return float(np.linalg.norm(reference - estimate, ord="fro") / denom)


def compression_ratio(image_shape: Tuple[int, int], k: int) -> float:
    m, n = image_shape
    original_storage = m * n
    svd_storage = k * (m + n + 1)
    return float(original_storage / svd_storage)


def energy_retained_from_factors(factors: Tuple[np.ndarray, np.ndarray, np.ndarray], k: int) -> float:
    _, s, _ = factors
    k = int(np.clip(k, 1, len(s)))
    return float(np.sum(s[:k] ** 2) / np.sum(s ** 2))


def add_gaussian_noise(image: np.ndarray, sigma: float, rng: np.random.Generator) -> np.ndarray:
    noisy = image + rng.normal(0.0, sigma, size=image.shape)
    return np.clip(noisy, 0.0, 255.0)


def nearest_neighbor_predict(query: np.ndarray, candidates: np.ndarray, candidate_labels: np.ndarray) -> int:
    flattened_candidates = candidates.reshape(candidates.shape[0], -1)
    flattened_query = query.reshape(1, -1)
    distances = np.linalg.norm(flattened_candidates - flattened_query, axis=1)
    return int(candidate_labels[np.argmin(distances)])


def evaluate_reconstruction(images: np.ndarray, k_values: Sequence[int]) -> pd.DataFrame:
    rows: list[dict[str, float]] = []
    shape = images[0].shape
    factor_list = precompute_svd(images)

    for k in k_values:
        recon_images = reconstruct_many_from_factors(factor_list, k)
        errors = [relative_frobenius_error(img, rec) for img, rec in zip(images, recon_images)]
        energies = [energy_retained_from_factors(factors, k) for factors in factor_list]
        rows.append(
            {
                "k": k,
                "avg_relative_fro_error": float(np.mean(errors)),
                "std_relative_fro_error": float(np.std(errors)),
                "avg_energy_retained": float(np.mean(energies)),
                "compression_ratio": compression_ratio(shape, k),
            }
        )

    return pd.DataFrame(rows)


def evaluate_denoising(
    images: np.ndarray,
    k_values: Sequence[int],
    noise_sigmas: Sequence[float],
    trials: int,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows: list[dict[str, float]] = []

    accum: dict[tuple[float, int], dict[str, list[float]]] = {
        (sigma, k): {"noisy": [], "denoised": [], "improvement": []}
        for sigma in noise_sigmas for k in k_values
    }

    for sigma in noise_sigmas:
        for _ in range(trials):
            for image in images:
                noisy = add_gaussian_noise(image, sigma=sigma, rng=rng)
                noisy_err = relative_frobenius_error(image, noisy)
                factors = svd_factors(noisy)
                for k in k_values:
                    denoised = reconstruct_from_factors(factors, k)
                    denoised_err = relative_frobenius_error(image, denoised)
                    accum[(sigma, k)]["noisy"].append(noisy_err)
                    accum[(sigma, k)]["denoised"].append(denoised_err)
                    accum[(sigma, k)]["improvement"].append(noisy_err - denoised_err)

    for sigma in noise_sigmas:
        for k in k_values:
            cell = accum[(sigma, k)]
            rows.append(
                {
                    "sigma": sigma,
                    "k": k,
                    "avg_noisy_error": float(np.mean(cell["noisy"])),
                    "avg_denoised_error": float(np.mean(cell["denoised"])),
                    "avg_error_reduction": float(np.mean(cell["improvement"])),
                }
            )

    return pd.DataFrame(rows)


def evaluate_matching(
    images: np.ndarray,
    labels: np.ndarray,
    k_values: Sequence[int],
    noise_sigmas: Sequence[float],
    train_per_subject: int,
    seed: int,
) -> pd.DataFrame:
    train_idx, test_idx = split_train_test_by_subject(labels, train_per_subject, seed)
    train_images, test_images = images[train_idx], images[test_idx]
    train_labels, test_labels = labels[train_idx], labels[test_idx]
    rows: list[dict[str, float]] = []

    clean_preds = [nearest_neighbor_predict(query, train_images, train_labels) for query in test_images]
    rows.append(
        {
            "mode": "clean_baseline",
            "sigma": 0.0,
            "k": np.nan,
            "accuracy": float(np.mean(np.asarray(clean_preds) == test_labels)),
        }
    )

    train_factor_list = precompute_svd(train_images)
    train_low_rank_by_k = {k: reconstruct_many_from_factors(train_factor_list, k) for k in k_values}

    rng = np.random.default_rng(seed + 999)
    for sigma in noise_sigmas:
        noisy_test = np.stack([add_gaussian_noise(img, sigma=sigma, rng=rng) for img in test_images], axis=0)
        noisy_preds = [nearest_neighbor_predict(query, train_images, train_labels) for query in noisy_test]
        rows.append(
            {
                "mode": "noisy_baseline",
                "sigma": sigma,
                "k": np.nan,
                "accuracy": float(np.mean(np.asarray(noisy_preds) == test_labels)),
            }
        )

        noisy_factor_list = precompute_svd(noisy_test)
        for k in k_values:
            test_denoised = reconstruct_many_from_factors(noisy_factor_list, k)
            train_low_rank = train_low_rank_by_k[k]
            preds = [nearest_neighbor_predict(query, train_low_rank, train_labels) for query in test_denoised]
            rows.append(
                {
                    "mode": "svd_denoised_match",
                    "sigma": sigma,
                    "k": float(k),
                    "accuracy": float(np.mean(np.asarray(preds) == test_labels)),
                }
            )

    return pd.DataFrame(rows)


def save_sample_panel(images: np.ndarray, k_values: Sequence[int], sigma: float, outpath: Path, seed: int) -> None:
    rng = np.random.default_rng(seed)
    original = images[0]
    noisy = add_gaussian_noise(original, sigma=sigma, rng=rng)
    noisy_factors = svd_factors(noisy)

    cols = 2 + len(k_values)
    fig, axes = plt.subplots(1, cols, figsize=(3 * cols, 3))
    axes[0].imshow(original, cmap="gray", vmin=0, vmax=255)
    axes[0].set_title("Original")
    axes[0].axis("off")

    axes[1].imshow(noisy, cmap="gray", vmin=0, vmax=255)
    axes[1].set_title(f"Noisy\nσ={sigma:g}")
    axes[1].axis("off")

    for i, k in enumerate(k_values, start=2):
        denoised = reconstruct_from_factors(noisy_factors, k)
        axes[i].imshow(denoised, cmap="gray", vmin=0, vmax=255)
        axes[i].set_title(f"Rank {k}")
        axes[i].axis("off")

    fig.tight_layout()
    fig.savefig(outpath, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_reconstruction_metrics(df: pd.DataFrame, outpath: Path) -> None:
    fig, ax1 = plt.subplots(figsize=(7, 5))
    ax1.plot(df["k"], df["avg_relative_fro_error"], marker="o", label="Avg relative Fro error")
    ax1.set_xlabel("Rank k")
    ax1.set_ylabel("Relative error")

    ax2 = ax1.twinx()
    ax2.plot(df["k"], df["compression_ratio"], marker="s", linestyle="--", label="Compression ratio")
    ax2.set_ylabel("Compression ratio")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="best")
    ax1.set_title("Reconstruction Error vs Compression Ratio")
    fig.tight_layout()
    fig.savefig(outpath, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_energy_curve(df: pd.DataFrame, outpath: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(df["k"], df["avg_energy_retained"], marker="o")
    ax.set_xlabel("Rank k")
    ax.set_ylabel("Average energy retained")
    ax.set_title("Average Energy Retained by Rank-k Approximation")
    fig.tight_layout()
    fig.savefig(outpath, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_denoising(df: pd.DataFrame, outpath: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    for sigma in sorted(df["sigma"].unique()):
        sub = df[df["sigma"] == sigma].sort_values("k")
        ax.plot(sub["k"], sub["avg_error_reduction"], marker="o", label=f"σ={sigma:g}")
    ax.set_xlabel("Rank k")
    ax.set_ylabel("Average error reduction")
    ax.set_title("How Much SVD Denoising Helps")
    ax.legend()
    fig.tight_layout()
    fig.savefig(outpath, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_matching(df: pd.DataFrame, outpath: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    clean = df[df["mode"] == "clean_baseline"]
    if not clean.empty:
        ax.axhline(clean["accuracy"].iloc[0], linestyle="--", label="Clean baseline")

    noisy = df[df["mode"] == "noisy_baseline"]
    if not noisy.empty:
        ax.plot(noisy["sigma"], noisy["accuracy"], marker="s", label="Noisy baseline")

    den = df[df["mode"] == "svd_denoised_match"]
    for sigma in sorted(den["sigma"].dropna().unique()):
        sub = den[den["sigma"] == sigma].sort_values("k")
        ax.plot(sub["k"], sub["accuracy"], marker="o", label=f"SVD denoised (σ={sigma:g})")

    ax.set_xlabel("Rank k (or sigma for noisy baseline)")
    ax.set_ylabel("Matching accuracy")
    ax.set_title("Identity Matching Accuracy")
    ax.legend()
    fig.tight_layout()
    fig.savefig(outpath, dpi=200, bbox_inches="tight")
    plt.close(fig)


def write_summary(reconstruction_df: pd.DataFrame, denoising_df: pd.DataFrame, matching_df: pd.DataFrame, outpath: Path) -> None:
    best_recon = reconstruction_df.sort_values("avg_relative_fro_error").iloc[0]
    best_denoise = denoising_df.sort_values("avg_error_reduction", ascending=False).iloc[0]
    svd_matching = matching_df[matching_df["mode"] == "svd_denoised_match"]
    best_match = None if svd_matching.empty else svd_matching.sort_values("accuracy", ascending=False).iloc[0]
    clean = matching_df[matching_df["mode"] == "clean_baseline"]
    clean_acc = None if clean.empty else clean["accuracy"].iloc[0]

    with open(outpath, "w", encoding="utf-8") as file:
        file.write("SVD Face Project Summary\n")
        file.write("========================\n\n")
        file.write("Best reconstruction setting\n")
        file.write("---------------------------\n")
        file.write(f"k = {int(best_recon['k'])}\n")
        file.write(f"Average relative Frobenius error = {best_recon['avg_relative_fro_error']:.4f}\n")
        file.write(f"Average energy retained = {best_recon['avg_energy_retained']:.4f}\n")
        file.write(f"Compression ratio = {best_recon['compression_ratio']:.4f}\n\n")

        file.write("Best denoising setting\n")
        file.write("----------------------\n")
        file.write(f"sigma = {best_denoise['sigma']}\n")
        file.write(f"k = {int(best_denoise['k'])}\n")
        file.write(f"Average noisy error = {best_denoise['avg_noisy_error']:.4f}\n")
        file.write(f"Average denoised error = {best_denoise['avg_denoised_error']:.4f}\n")
        file.write(f"Average error reduction = {best_denoise['avg_error_reduction']:.4f}\n\n")

        file.write("Best matching setting\n")
        file.write("---------------------\n")
        if clean_acc is not None:
            file.write(f"Clean baseline accuracy = {clean_acc:.4f}\n")
        if best_match is not None:
            file.write(f"Best SVD matching sigma = {best_match['sigma']}\n")
            file.write(f"Best SVD matching k = {int(best_match['k'])}\n")
            file.write(f"Best SVD matching accuracy = {best_match['accuracy']:.4f}\n")


def save_config(args: argparse.Namespace, dataset: Dataset, outpath: Path) -> None:
    config = {
        "dataset_path": str(args.dataset_path),
        "num_images": int(len(dataset.images)),
        "num_subjects": int(len(np.unique(dataset.labels))),
        "image_shape": tuple(int(v) for v in dataset.images[0].shape),
        "k_values": list(args.k_values),
        "noise_sigmas": list(args.noise_sigmas),
        "train_per_subject": int(args.train_per_subject),
        "denoise_trials": int(args.denoise_trials),
        "seed": int(args.seed),
        "max_subjects": int(args.max_subjects),
        "max_images_per_subject": int(args.max_images_per_subject),
        "resize_width": int(args.resize_width),
        "resize_height": int(args.resize_height),
    }
    pd.Series(config).to_json(outpath, indent=2)


def clean_temp_path(path: Path | None) -> None:
    if path is not None and path.exists():
        shutil.rmtree(path, ignore_errors=True)


def main() -> None:
    args = parse_args()
    if args.dataset_path is None:
        raise ValueError("No dataset path provided. Put att_faces_dataset.zip next to the script or pass --dataset_path.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    prepared: PreparedDataset | None = None
    try:
        prepared = prepare_dataset(Path(args.dataset_path), output_dir / "_extracted_dataset")
        dataset = load_face_dataset(
            prepared.root,
            max_subjects=args.max_subjects,
            max_images_per_subject=args.max_images_per_subject,
            resize_width=args.resize_width,
            resize_height=args.resize_height,
        )

        print(f"Loaded {len(dataset.images)} images from {prepared.root}")
        print(f"Image shape: {dataset.images[0].shape}")
        print(f"Subjects: {len(np.unique(dataset.labels))}")

        reconstruction_df = evaluate_reconstruction(dataset.images, args.k_values)
        denoising_df = evaluate_denoising(
            dataset.images, args.k_values, args.noise_sigmas, args.denoise_trials, args.seed
        )
        matching_df = evaluate_matching(
            dataset.images, dataset.labels, args.k_values, args.noise_sigmas, args.train_per_subject, args.seed
        )

        reconstruction_df.to_csv(output_dir / "reconstruction_metrics.csv", index=False)
        denoising_df.to_csv(output_dir / "denoising_metrics.csv", index=False)
        matching_df.to_csv(output_dir / "matching_metrics.csv", index=False)

        save_sample_panel(dataset.images, args.k_values[:4], args.noise_sigmas[0], output_dir / "sample_reconstruction_panel.png", args.seed)
        plot_reconstruction_metrics(reconstruction_df, output_dir / "reconstruction_vs_compression.png")
        plot_energy_curve(reconstruction_df, output_dir / "energy_retained.png")
        plot_denoising(denoising_df, output_dir / "denoising_improvement.png")
        plot_matching(matching_df, output_dir / "matching_accuracy.png")
        write_summary(reconstruction_df, denoising_df, matching_df, output_dir / "summary.txt")
        save_config(args, dataset, output_dir / "run_config.json")

        print("\nSaved outputs to:", output_dir.resolve())
    finally:
        clean_temp_path(None if prepared is None else prepared.cleanup_path)


if __name__ == "__main__":
    main()
