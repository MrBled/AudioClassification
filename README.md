# Acoustic Scene Classification (FCNN, PyTorch)

This project implements an acoustic scene classification model using a Fully Convolutional Neural Network (FCNN).
The model classifies 10-second audio recordings into one of several real-world acoustic scenes.

---

## Dataset

TUT Acoustic Scenes 2017 Development

---

## Input Features

- Audio is loaded as mono at 44.1 kHz.
- Features:
  - Log-Mel spectrogram (128 mel bins)
  - Δ (delta) and ΔΔ (delta–delta)
- Features are stacked to form 3 channels: [log-mel, delta, delta-delta]

Input tensor shape (channels-first):

| Dimension | Meaning                     |
|----------:|-----------------------------|
|    **B**  | batch size                  |
|    **C**  | channels = 3                |
|    **T**  | time frames (~500 for 10 s) |
|    **F**  | frequency bins = 128        |

PyTorch: (B, C, T, F)

---

## Output

- [beach, bus, cafe/restaurant, car, city_center, forest_path, grocery_store, home, library, metro_station, office, park, residential_area, train, tram]
- Labels are integers from 0 to 14.
- Output tensor shape: (B, num_classes)

---

## Current Architecture

The FCNN consists of convolutional blocks with batch normalisation, ReLU activations, dropout, max pooling, and channel attention.

| Stage              | Operations                                  | Output shape (example T=500, F=128) |
|--------------------|---------------------------------------------|-------------------------------------|
| Input              | Features                                    | (B, 3, 500, 128)                    |
| ConvLayer1         | 2 × Conv + BN + ReLU; MaxPool(2×2)          | (B, 144, 125, 32)                   |
| ConvLayer2         | 2 × Conv + BN + ReLU; MaxPool(2×2)          | (B, 288, 62, 16)                    |
| ConvLayer3         | 4 × [Conv + BN + ReLU + Dropout]; MaxPool   | (B, 576, 31, 8)                     |
| 1×1 Conv + BN      | Project to num_classes channels              | (B, num_classes, 31, 8)             |
| Channel Attention  | Squeeze-and-Excitation weighting            | (B, num_classes, 31, 8)             |
| Global Avg Pool    | Pool over time and frequency                | (B, num_classes, 1, 1)              |
| Flatten / Softmax  | Class probabilities                         | (B, num_classes)                    |

---

## Training

- Loss: CrossEntropyLoss
- Optimiser: AdamW
- Scheduler: Cosine annealing
- Split: 90% train / 10% validation
- Mixed precision training supported (torch.cuda.amp)

---

## Example Command

python train.py \
  --root /data/clement/data/acoustic_scenes/TUT-acoustic-scenes-2017-development \
  --epochs 30 \
  --batch_size 32 \
  --workers 4
