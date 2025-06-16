# 🌥️ Multi-Attention Enhancement Thin Cloud Removal using GAN

This project introduces a GAN-based deep learning model for removing thin clouds from satellite imagery. The architecture, **MAE-CG (Multi-Attention Enhancement Conditional GAN)**, incorporates two attention mechanisms:

- 🔍 `CBAM` – Convolutional Block Attention Module for channel and spatial refinement  
- 🌐 `GLAM` – Global and Local Attention Module for improved feature enhancement

**Paper Published in IEEE:** [Link to Paper](https://ieeexplore.ieee.org/abstract/document/10984085)  

## 🧪 Dataset
- **RICE-1**: Cloudy and cloud-free satellite image pairs used for training and evaluation.

## 📊 Performance Comparison

| Model              | PSNR  | SSIM  | MSE   | RMSE |
|--------------------|-------|-------|-------|------|
| DHI                | 15.01 | 0.805 | 0.24  | 0.48 |
| AOD-Net            | 18.76 | 0.814 | 0.36  | 0.60 |
| DehazeNet          | 23.14 | 0.838 | 0.15  | 0.38 |
| Cycle-Dehaze       | 23.73 | 0.851 | 0.22  | 0.46 |
| **MAE-CG (Proposed)** | **24.80** | **0.862** | **0.16** | **0.40** |

## 🛠️ Tech Stack
- `Python`
- `TensorFlow / Keras`
- `OpenCV`
- `NumPy`
- `Matplotlib`

## 🚀 Features
- ✅ GAN-based thin cloud removal
- ✅ Dual attention module integration (CBAM + GLAM)
- ✅ Stable training with loss tracking
- ✅ Visual and quantitative comparisons with other methods

---




---
