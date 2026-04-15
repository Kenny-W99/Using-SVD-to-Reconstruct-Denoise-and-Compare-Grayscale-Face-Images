# Result notes from a full 40-subject run

Using the uploaded AT&T face dataset, resized to `48 x 48`, with `k = [5, 10, 20, 30]`, `sigma = [10, 20]`, and `train_per_subject = 6`:

- Best reconstruction quality happened at **k = 30**
- Average relative Frobenius reconstruction error at `k = 30` was about **0.0084**
- Best denoising improvement happened at **sigma = 20**, **k = 10**
- Clean nearest-neighbor matching accuracy was about **0.9813**
- Best SVD-based noisy matching accuracy also reached about **0.9813**

These numbers are stored in `outputs/summary.txt` and the CSV files.
