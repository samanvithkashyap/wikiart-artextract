# WikiArt Style Classification — ArtExtract GSoC 2026

## Why This Project

There's something almost mystical about the idea that a painting can hide another reality beneath it — that under the visible brushstrokes of a finished work, there might be a completely different composition, a discarded idea, or an earlier version the artist painted over. The ArtExtract project is ultimately about building systems that can see what humans miss in art. That idea hooked me immediately.

This repository is my baseline implementation for the **HumanAI ArtExtract GSoC 2026 task** — the foundation I'm building toward that larger goal. Before you can find hidden paintings or detect anomalies in art, you need a model that genuinely understands artistic style. That's what this is.

---

## What I Built

A CRNN (Convolutional-Recurrent Neural Network) trained on 81,444 paintings across 29 artistic styles from the WikiArt dataset.

**Why a CRNN and not just a CNN?**

Style isn't a local feature. When you look at an Impressionist painting, what tells you it's Impressionist isn't any single patch — it's the way loose brushstrokes repeat and flow across the entire canvas. A plain CNN sees small regions independently and misses that spatial continuity. The BiGRU fixes this by treating the image as a sequence of vertical column features extracted by EfficientNet, capturing left-to-right and right-to-left relationships across the full width of the canvas. That sequential context is exactly what style recognition needs.

**Why EfficientNet?**

EfficientNet scales width, depth, and resolution together (compound scaling) instead of just stacking more layers. This gives richer, more expressive feature maps with fewer parameters than ResNet or VGG. For art images where texture and fine detail matter, that efficiency translates directly to better feature extraction without the computational overhead.

**Architecture:**
- EfficientNet-B3 backbone → 1536-channel feature maps
- Reshape to column sequence → 2-layer Bidirectional GRU (hidden=256)
- FC(512 → 29 styles)

**Training results:** Loss dropped from 1.69 → 0.64 over 5 epochs on a Kaggle T4 GPU.

---

## Outlier Detection

Also implemented an outlier detection pipeline using per-image cross-entropy loss + confidence scoring to identify paintings that don't visually fit their assigned style label.

The logic: high loss + high confidence = genuine outlier. The model is certain it belongs somewhere else. High loss + low confidence = just a hard example.

**Interesting findings:**
- Jackson Pollock drip paintings (labeled Abstract Expressionism) confused the model completely — predicted Baroque at 99% confidence. Makes sense: drip paintings look unlike anything else in the dataset.
- Hilma af Klint (labeled Symbolism) predicted Minimalism — she painted abstract geometric works in 1906, decades before abstraction existed as a movement. The label is arguably wrong.
- Edward Hopper (labeled New Realism) predicted Realism — likely a dataset mislabeling, Hopper is commonly miscategorized in WikiArt.

![Outlier Visualization](outliers_visualization.png)

---

## Future Plan — Multi-Task Architecture

The next version extends this to a multi-head model predicting Style, Artist, and Genre simultaneously:

- Shared EfficientNet-B3 backbone
- Style head: BiGRU (spatial sequence matters for brushstroke patterns)
- Artist head: Global Average Pooling → FC layers
- Genre head: Global Average Pooling → FC layers
- Combined loss: λ₁·L_style + λ₂·L_artist + λ₃·L_genre

Two-phase training — freeze backbone first, then fine-tune last 3 blocks at lower LR.

---

## Notebook

Full implementation on Kaggle: [View Notebook](https://www.kaggle.com/code/samanvithkashyap/train-initial-architechure?scriptVersionId=299832708)

---

## Dataset

WikiArt — 81,444 images, 29 styles, 195 artists, 10 genres.
