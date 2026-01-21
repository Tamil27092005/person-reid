# 🎯 Commercial Person Re-Identification System
### Cross-Camera Tracking for Retail Analytics & Surveillance
A production-ready Person Re-Identification system that tracks individuals across multiple non-overlapping cameras using deep metric learning. Built with OSNet architecture + ArcFace loss, achieving state-of-the-art performance for retail and surveillance applications.

Person Re-Identification (ReID) is a critical computer vision task that matches individuals across different cameras and viewpoints. This project addresses the challenge of tracking people in retail environments where traditional single-camera systems fail due to:

- **Non-overlapping camera views** (entrance, aisles, checkout zones)
- **Appearance variations** (lighting, pose, occlusion)
- **Scale challenges** (distance, resolution differences)
- **Real-time requirements** (low latency for analytics)

Our system combines:

1. **OSNet x0.75** - Efficient omni-scale feature learning backbone (2.4M parameters)
2. **ArcFace Loss** - Angular margin loss for discriminative embeddings (scale=30, margin=0.3)
3. **Triplet Loss** - Hard negative mining for robust feature separation (margin=0.3)
4. **Multi-Dataset Training** - Trained on 256k images across 4 major datasets (Market-1501, DukeMTMC-reID, CUHK03, LAST)

**Result**: 85.98% Rank-1 accuracy on unseen test data, enabling reliable cross-camera tracking for retail analytics including visitor counting, dwell time measurement, and customer journey mapping.

---

## 👥 Collaborators

## 🚀 Key Features

### 1. 🎯 State-of-the-Art Performance

- **85.98% Rank-1 Accuracy** on test set (1,555 persons, 3,110 queries)
- **62.44% mAP** (Mean Average Precision)
- **92.78% Rank-5** (correct person in top-5 matches)
- **Zero-shot generalization** to unseen camera networks

### 2. 🏗️ Production-Ready Architecture

- **Lightweight Model**: OSNet x0.75 with only 2.4M parameters
- **Real-time Inference**: 120 FPS on NVIDIA GTX 1060+
- **512-dim Embeddings**: L2-normalized feature vectors for fast similarity search
- **GPU Optimized**: Mixed precision training with automatic gradient scaling

### 3. 🌍 Multi-Dataset Training

Trained on **4 major benchmark datasets** for robust generalization:

| Dataset | Images | Persons | Cameras | Environment |
|---------|--------|---------|---------|-------------|
| **Market-1501** | 29,419 | 1,501 | 6 | Outdoor retail |
| **DukeMTMC-reID** | 36,411 | 1,404 | 8 | Indoor/outdoor campus |
| **CUHK03** | 14,097 | 1,467 | 10 pairs | Indoor CCTV |
| **LAST** | 176,426 | 10,808 | Multiple | Large-scale diverse |
| **Total** | **256,353** | **15,588** | **30+** | Mixed scenarios |

### 4. 📊 Flexible Deployment Modes

python
# High Recall (Visitor Counting)
reid = ReIDInference(threshold=0.50)  # F1=94.1%, Recall=95.1%

# Balanced (Customer Journey)
reid = ReIDInference(threshold=0.65)  # F1=79.6%, Precision=98.3%

# High Precision (VIP Recognition)
reid = ReIDInference(threshold=0.80)  # F1=47.5%, Precision=98.6%
