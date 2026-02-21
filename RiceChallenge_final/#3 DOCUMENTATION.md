# Solution Documentation

## Overview and Objectives
This solution trains and runs a multi-task computer vision model that predicts 15 rice quality indicators from images. The model estimates grain counts (including sub-categories) and size/color measurements directly from tiled image inputs. The objective is to provide accurate predictions that match the competition evaluation (MAE across all targets).

Expected outcomes:
- A reproducible training pipeline that produces a checkpoint.
- An inference script that generates a competition-ready submission CSV.
- A data pipeline that relies only on competition-provided datasets.

## Mobile-Friendly Inference Rationale
The inference pipeline is designed to be mobile-friendly because it runs a single forward pass on a fixed number of tiles per image (8x6 grid) without requiring heavy post-processing or multi-stage ensembles. The ConvNeXt-Small variant was chosen to keep parameter count and memory usage lower, and only one model is used during inference to reduce RAM/VRAM demands. This keeps memory usage predictable and allows the pipeline to be exported or optimized for edge devices. The checkpoint can be deployed with a lightweight inference script that performs only tiling, resizing, normalization, and a single model call per image, which suits mobile or embedded deployments.

## Architecture Diagram
```
                +---------------------+
                |  Data Sources       |
                |  - Train.csv        |
                |  - Test.csv         |
                |  - Image zips       |
                +----------+----------+
                           |
                           v
                 +--------------------+
                 | ETL / Preprocess   |
                 | - Download images  |
                 | - Tiling (8x6)     |
                 | - Resize/Normalize |
                 +----------+---------+
                           |
                           v
                 +--------------------+
                 | Model Training     |
                 | - ConvNeXt backbone|
                 | - Multi-scale heads|
                 | - MAE metrics      |
                 +----------+---------+
                           |
                           v
                 +--------------------+
                 | Checkpoint (.pth)  |
                 +----------+---------+
                           |
                           v
                 +--------------------+
                 | Inference          |
                 | - Load checkpoint  |
                 | - Predict targets  |
                 | - Write submission |
                 +--------------------+
```

## ETL Process

### Extract
- **Data sources**: `Data/Train.csv`, `Data/Test.csv`, and image zips hosted at `https://storage.googleapis.com/unido-afririce/`.
- **Extraction method**: `download_unido_images.py` downloads and extracts image zips into `Data/images/`.
- **Considerations**: image data is large; downloads are one-time and cached locally.

### Transform
- **Tiling**: each image is split into an 8x6 non-overlapping grid.
- **Resize**: each tile is resized to `512x512`.
- **Normalize**: standard image normalization via Albumentations.
- **Augmentations** (train only): horizontal/vertical flips and random rotations.
- **Target scaling**: count targets are scaled by `SCALE=100.0` during training.

### Load
- **Storage**: local filesystem under `Data/`.
- **Access**: CSVs are read into pandas; images are loaded with PIL and cached in memory for training.
- **Optimization**: training caches images to speed up epochs.

## Data Modeling
- **Model**: ConvNeXt-Small backbone (timm) with multi-scale decoder heads for counts and a pooled head for measures.
- **Targets**: 9 count targets and 6 measurement targets.
- **Losses**:
  - Weighted Huber-like loss for count targets.
  - L1 loss for measurement targets.
  - Consistency loss enforcing sum of sub-counts equals total.
- **Implementation details and design decisions**:
  - **Fixed 8x6 tiling**: Chosen to avoid overlap and double counting while keeping a constant number of tiles per image for stable batching and memory usage.
  - **Multi-scale decoding (1/16 + 1/32)**: Combines mid- and low-resolution features to balance spatial detail (counts) and global context (measures).
  - **Meta conditioning**: A learned projection of rice type is injected into the decoder to let the model adapt per class without separate specialists.
  - **Count scaling (`SCALE=100.0`)**: Stabilizes gradients for large count values and improves optimizer behavior at small batch sizes.
  - **Weighted Huber-like loss**: Emphasizes harder count categories while limiting the impact of outliers.
  - **Consistency constraint**: Encourages the model to respect known label structure (Broken + Long + Medium = Total).
  - **Train-time caching**: Preloads images to reduce I/O stalls during long training runs.
- **Hyperparameters**:
  - `BATCH_SIZE=2`, `GRAD_ACCUM=4`, `EPOCHS=200`
  - `LR=4e-5`, `WEIGHT_DECAY=0.05`
  - `TILE_SIZE=512`, `GRID=8x6`
- **Evaluation metric**: Mean Absolute Error (MAE) across all targets.
- **Validation**: holdout split using `train_test_split` with stratification on `Comment`.

## Inference
- **Script**: `submit.py`
- **Inputs**: `Data/Test.csv` and images in `Data/images/images/`.
- **Outputs**: `submission.csv` with required columns and ordering.
- **Versioning**: checkpoint file `ultimate_tiled_multitask.pth` referenced by filename.
- **Post-processing**: class-specific zeroing for certain rice types.
- **Deployment**: run on a single GPU/CPU instance; can be containerized for batch scoring.
- **Update strategy**: store checkpoints with version tags and record training config and scores; retrain when new labeled data is available or MAE degrades.

## Run Time
Run time depends on hardware and dataset size:
- Setup used for the reported run time:
  - GPU: RTX 5090 32GB
  - CPU: AMD Ryzen 9 9900X 12-Core processor
  - RAM: 64GB
- `download_unido_images.py`: download + unzip (network bound).
- `train_all_specialists.py`: about 20 hours for a full run in this setup.
- `submit.py`: minutes on GPU; longer on CPU.

## Data Volume and Frequency
- **Volume**: multi-zip image dataset; large downloads and significant disk usage.
- **Frequency**: one-time download; re-download only if the dataset is updated.

## Performance Metrics
- **Primary metric**: MAE (competition metric).
- **Scores**: fill in your public and private leaderboard scores here:
  - Public score: 0.928982656
  - Private score: 0.925934544
- **Other metrics**: per-target MAE reported during training.

## Error Handling and Logging
- **Downloads**: basic error handling for HTTP errors and Google Drive virus scan pages.
- **Training/inference**: errors surface via exceptions; progress and epoch metrics are logged using `rich`.

## Potential Issues and Best Practices
- **Potential issues**: network failures during download; CPU-only training is very slow; cached images can increase memory usage.
- **Best practices**: verify dataset integrity after download; version checkpoints (date + metric); monitor MAE drift between runs.

## Maintenance and Monitoring
- **Monitoring**: track MAE per epoch and compare against best checkpoint.
- **Retraining**: re-run `train_all_specialists.py` when new data or improved hyperparameters are available.
- **Scaling**: enable multi-GPU or mixed-precision training for faster iterations.
- **Lifecycle**: store checkpoints with version tags and keep a changelog for improvements.

## Usage Instructions
1. Install dependencies: `pip install -r requirements.txt`
2. Download images: `python download_unido_images.py`
3. Train model: `python train_all_specialists.py`
4. Download checkpoint (optional): `python download_checkpoint.py`
5. Generate submission: `python submit.py`

## Notes
- All preprocessing is performed in code; no external preprocessing is required.
- Seeds are set in both training and inference for reproducibility.
