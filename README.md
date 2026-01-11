# Camera_Trap_Analyzer (Animal Presence Detection)

This project implements a simple computer vision pipeline to detect whether an animal is present in camera trap images.
It uses the Caltech Camera Traps dataset metadata and streams images on demand (no full dataset download required).

## Features
- Downloads Caltech Camera Traps metadata (COCO Camera Traps JSON) automatically
- Fetches camera trap images from the public Azure bucket and caches them locally
- CV pipeline:
  - grayscale conversion
  - background subtraction (MOG2)
  - thresholding + morphological cleanup
  - foreground-area rule to classify **animal present** vs **empty**
- Saves visual overlays and a CSV report
- Computes evaluation metrics: accuracy, precision, recall, and confusion matrix

## Setup
### 1. Create and activate a virtual environment
### 2. Install dependencies
### 3. Run "python src/main.py"
The script will:
- download metadata into data/meta/
- download only the images it needs into data/cache/
- save results to: outputs/overlays/ (images with foreground mask overlay + text) AND outputs/reports/results.csv (per-image predictions)

It also prints accuracy, precision, recall, and the confusion matrix.

## Notes
The dataset images are not included in this repository. They are fetched from the public Caltech Camera Traps storage and cached locally in data/.
You can tune the detector by editing parameters in src/main.py: frames_per_seq, area_thresh, and MOG2 settings.
This repository contains only code. Dataset licensing is handled by the dataset providers.
