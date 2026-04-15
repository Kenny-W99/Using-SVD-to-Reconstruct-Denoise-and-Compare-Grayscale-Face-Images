# Beyond Compression: Using SVD to Reconstruct, Denoise, and Compare Grayscale Face Images

This project uses the AT&T Database of Faces to study three questions:

1. How does truncated SVD reconstruct grayscale face images at different ranks `k`?
2. Can low-rank SVD help remove Gaussian noise from face images?
3. Can reconstructed images still be used for simple identity matching?

## Project structure

- `svd_face_project.py` — main runnable script
- `att_faces_dataset.zip` — dataset zip used in the project
- `requirements.txt` — required Python packages
- `outputs/` — generated CSV files, plots, and summary text

## Methods used

- **SVD / low-rank approximation** for image reconstruction
- **Frobenius norm** for reconstruction and denoising error
- **Gaussian noise** for probability-based perturbation
- **Nearest-neighbor matching** with Euclidean distance for image comparison

## How to run

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the full project:

```bash
./run_project.sh
```

Equivalent Python command:

```bash
python svd_face_project.py --dataset_path att_faces_dataset.zip --output_dir outputs
```

## Optional faster test run

```bash
python svd_face_project.py \
  --dataset_path att_faces_dataset.zip \
  --output_dir outputs_test \
  --max_subjects 10 \
  --resize_width 48 \
  --resize_height 48
```

## Main outputs

The script saves:

- `reconstruction_metrics.csv`
- `denoising_metrics.csv`
- `matching_metrics.csv`
- `sample_reconstruction_panel.png`
- `reconstruction_vs_compression.png`
- `energy_retained.png`
- `denoising_improvement.png`
- `matching_accuracy.png`
- `summary.txt`
- `run_config.json`

## Notes

- The script automatically extracts the dataset zip to a temporary folder inside the output directory.
- Thread counts for NumPy/BLAS are capped inside the script so SVD runs do not become extremely slow on some machines.
- Default settings resize images to `48 x 48` for faster execution. Use `--resize_width 0 --resize_height 0` if you want to keep the original resolution.


If SVD runs very slowly on your machine, use `./run_project.sh` because it sets safer BLAS thread limits before Python starts.

The original dataset images are stored in .pgm format, which is a standard grayscale image format.
The project code reads these files directly and processes them as image matrices.