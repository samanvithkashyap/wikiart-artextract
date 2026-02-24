# WikiArt Style Classification — ArtExtract GSoC 2026

This repository contains my baseline implementation for the **HumanAI ArtExtract GSoC 2026 task** — classifying paintings by artistic style using a CRNN architecture on the WikiArt dataset.

## What I Built

A CNN + BiGRU model (EfficientNet-B3 backbone + 2-layer Bidirectional GRU) trained on 81,444 paintings across 29 artistic styles. Style classification benefits from a sequential approach because brushstroke and texture patterns span spatially across the canvas — the BiGRU captures these left-to-right and right-to-left relationships across vertical column features extracted by EfficientNet.

Also implemented an outlier detection pipeline using per-image cross-entropy loss + confidence scoring to identify paintings that don't visually fit their assigned style label. Found interesting cases like Jackson Pollock drip paintings confusing the model completely, and a potential Hopper mislabeling in the dataset.

**Training results:** Loss dropped from 1.69 → 0.64 over 5 epochs on a Kaggle T4 GPU.

## Future Plan — Multi-Task Architecture

The next version will extend this to a multi-head model predicting Style, Artist, and Genre simultaneously:

- Shared EfficientNet-B3 backbone
- Style head: BiGRU (spatial sequence matters for brushstroke patterns)
- Artist head: Global Average Pooling → FC layers
- Genre head: Global Average Pooling → FC layers
- Combined loss: λ₁·L_style + λ₂·L_artist + λ₃·L_genre

Two-phase training — freeze backbone first, then fine-tune last 3 blocks at lower LR.

## Dataset

WikiArt — 81,444 images, 29 styles, 195 artists, 10 genres.
