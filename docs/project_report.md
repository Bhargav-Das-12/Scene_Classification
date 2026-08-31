# Scene Classification: Project Report

**[← Back to Home](./index.md)**

## 1. Introduction

Classifying real-world scenes presents a challenging computer vision problem because many environments have strong semantic overlap. For example, scenes such as a bedroom and a hotel room may contain visually similar objects and layouts. This project, focuses on building an efficient, accurate, and deployable scene classification system capable of categorizing images into **314 distinct scene categories**.

Two fundamentally different deep learning architectures were evaluated:

- **MobileNetV2** — a lightweight Convolutional Neural Network (CNN).
- **DINOv2 (ViT-S/14)** — a self-supervised Vision Transformer.

The models were evaluated using classification accuracy, Cross-Entropy Loss, computational cost, parameter count, and inference latency. The main objective was to study the trade-off between classification performance and computational efficiency.

### Project Objective

The primary objective is to quantify the **accuracy–efficiency trade-off** between a lightweight CNN and a Vision Transformer and determine their suitability for different deployment requirements.

---

## 2. Project Tasks

1. **Dataset Preparation** — Organization of a curated Kaggle subset of the MIT Places365 dataset.
2. **Architecture Selection** — Comparison of MobileNetV2 and DINOv2 (ViT-S/14).
3. **Evaluation Strategy** — Top-1/Top-5 accuracy, Cross-Entropy Loss, FLOPs, parameter count, and latency.
4. **MobileNetV2 Training** — Fine-tuning the later portion of the CNN backbone with a custom classification head.
5. **DINOv2 Training** — Training a custom MLP classification head with the Transformer backbone frozen.
6. **Model Analysis** — Comparison of accuracy, training dynamics, computational cost, and inference efficiency.
7. **Deployment** — Development of a full-stack React (Vite) and FastAPI application for side-by-side model inference.

---

## 3. Methodology

### 3.1 Dataset

This project uses a curated subset of the original **MIT Places365** dataset for efficient experimentation across **314 distinct scene categories**.

- **Raw Dataset Source:** [Kaggle Places365 Dataset](https://www.kaggle.com/datasets/pankajkumar2002/places365)

- **Processed Dataset:** [Organized Dataset – Google Drive](https://drive.google.com/file/d/1Hh9pWRZChD8zzoSHDsptuxQzO1qCtkSA/view?usp=drive_link)

  _The processed dataset contains images organized into PyTorch-ready class directories for the corresponding train, validation, and test splits._

### 3.2 Preprocessing

Training images were processed using data augmentation and normalization techniques, including:

- `RandomResizedCrop`
- `RandomRotation`
- `ColorJitter`
- ImageNet normalization

For model inference, uploaded images follow the same spatial input convention used by the trained models:

`RGB → Resize(256 × 256) → CenterCrop(224 × 224) → ImageNet Normalization`

### 3.3 Model Architectures

The project evaluates two contrasting approaches to feature extraction and classification: a lightweight CNN designed for computational efficiency and a self-supervised Vision Transformer designed to provide rich visual representations.

#### MobileNetV2

MobileNetV2 was selected as the lightweight CNN baseline.

- The initial CNN backbone was frozen.
- The final five feature blocks, `features[14:]`, were unfrozen so that the feature extractor could adapt to the Places365 scene categories.
- A custom classifier head with **Dropout (`p = 0.5`)** was attached.
- The classifier produces predictions for **314 scene classes**.
- Training used **Cross-Entropy Loss**.

#### DINOv2 (ViT-S/14)

DINOv2 was selected as the Vision Transformer architecture.

- The complete Transformer backbone was **frozen** and used as a feature extractor.
- A custom two-layer MLP classification head was trained.
- Head structure: `Linear → ReLU → Dropout → Linear`
- The final layer predicts the **314 scene classes**.
- Training used **Cross-Entropy Loss**.

Because the DINOv2 backbone remained frozen, the classification head learned rapidly and validation performance began to deteriorate relatively early during training.

### 3.4 Training Dynamics

#### MobileNetV2

MobileNetV2 followed a smoother convergence pattern. Training continued for **12 epochs**, with validation accuracy gradually improving before reaching its peak performance.

#### DINOv2

DINOv2 reached its strongest validation performance early in training. The best validation accuracy occurred at approximately **Epoch 3**, while validation loss subsequently increased as training loss continued to decrease. Early stopping was triggered at **Epoch 7**.

The best-performing checkpoint was therefore used rather than simply using the final training epoch.

### 3.5 Training Curves

#### MobileNetV2

![MobileNetV2 Training Results](/results/mobile_net_main.png)

#### DINOv2

![DINOv2 Training Results](/results/dinov2_main.png)

The training curves show steady improvement in MobileNetV2 training accuracy over the 12 epochs, while DINOv2 shows rapid early validation improvement followed by increasing validation loss, supporting the use of early stopping.

---

## 4. Pipeline Architecture & Deployment

The final system uses a dual-model inference pipeline in which an uploaded image is processed once and evaluated by both trained models.

### 4.1 Data Ingestion & Preprocessing

An image uploaded through the React frontend is sent to the FastAPI backend. The backend converts the image to RGB, resizes it to **256 × 256 pixels**, center-crops it to **224 × 224 pixels**, and applies ImageNet normalization.

### 4.2 Dual-Model Inference

The preprocessed tensor is passed through both:

- **MobileNetV2:** Fine-tuned lightweight CNN.
- **DINOv2:** Frozen ViT-S/14 backbone with a custom MLP classification head.

Each model produces logits for the **314 scene categories**.

### 4.3 Confidence Scoring & Output

The logits produced by each model are converted into probability distributions using Softmax. `torch.topk` is then used to extract the five highest-scoring predictions.

The resulting Top-5 predictions and confidence scores are returned to the React frontend as a structured JSON response.

### 4.4 Pipeline Data Flow

```text
                 ┌───────────────────────┐
                 │    Local Image Upload │
                 │        React UI       │
                 └───────────┬───────────┘
                             │
                             ▼
                 ┌───────────────────────┐
                 │    FastAPI Backend    │
                 │       /predict        │
                 └───────────┬───────────┘
                             │
                             ▼
                 ┌───────────────────────┐
                 │  PyTorch Preprocessing│
                 │   RGB Conversion      │
                 │   Resize 256 × 256    │
                 │   CenterCrop 224 ×224 │
                 │   ImageNet Normalize  │
                 └───────────┬───────────┘
                             │
                    ┌────────┴────────┐
                    ▼                 ▼
          ┌────────────────┐   ┌────────────────────┐
          │  MobileNetV2   │   │ DINOv2 (ViT-S/14)  │
          │ Fine-tuned CNN │   │ Frozen Transformer │
          └───────┬────────┘   └──────────┬─────────┘
                  │                       │
                  └──────────┬────────────┘
                             ▼
                 ┌───────────────────────┐
                 │ Softmax + torch.topk  │
                 │    Top-5 Extraction   │
                 └───────────┬───────────┘
                             │
                             ▼
                 ┌───────────────────────┐
                 │   Top-5 Predictions   │
                 │   Confidence Scores   │
                 └───────────┬───────────┘
                             │
                             ▼
                 ┌───────────────────────┐
                 │ React Comparison      │
                 │ Dashboard             │
                 └───────────────────────┘
```

### 4.5 Web Application

The trained models were integrated into a full-stack web application for interactive comparison.

**Backend — FastAPI**

- Loads the trained PyTorch model weights during application startup.
- Keeps the models available for inference.
- Receives uploaded images and performs preprocessing.
- Runs both classifiers.
- Calculates Top-5 predictions and confidence scores.
- Returns the results as JSON.

**Frontend — React & Vite**

- Provides a local image-upload interface.
- Sends images to the FastAPI backend.
- Displays MobileNetV2 and DINOv2 predictions side by side.
- Presents Top-5 scene predictions and confidence scores for comparison.

---

## 5. Evaluation Methodology

The models were evaluated using classification performance metrics together with computational profiling.

### Top-1 Accuracy

Percentage of images for which the highest-probability prediction is the correct class.

### Top-5 Accuracy

Percentage of images for which the correct class appears among the five highest-probability predictions. This is particularly useful for semantically ambiguous scenes.

### Cross-Entropy Loss

Measures the quality of the predicted class probability distribution during training.

### Inference Latency

Average inference time per image after GPU warm-up.

### FLOPs

Theoretical computational cost required for a standard inference pass.

---

## 6. Results

### 6.1 Training and Validation Results

#### MobileNetV2

| Metric             | Training | Validation | Final Test |
| ------------------ | -------: | ---------: | ---------: |
| Cross-Entropy Loss |   2.7287 |     3.1151 |          — |
| Top-1 Accuracy     |   47.95% |     37.39% | **37.72%** |
| Top-5 Accuracy     |        — |          — | **71.25%** |

#### DINOv2

| Metric             | Training | Validation | Final Test |
| ------------------ | -------: | ---------: | ---------: |
| Cross-Entropy Loss |   1.8801 |     2.7748 |          — |
| Top-1 Accuracy     |   49.04% |     43.68% | **43.18%** |
| Top-5 Accuracy     |        — |     76.65% | **76.31%** |

The validation evaluation for DINOv2 covered **7,298 images**, while the final test evaluation covered **7,302 images**.

### 6.2 Final Test Accuracy

| Model                 | Top-1 Test Accuracy | Top-5 Test Accuracy |
| --------------------- | ------------------: | ------------------: |
| **MobileNetV2**       |              37.72% |              71.25% |
| **DINOv2 (ViT-S/14)** |          **43.18%** |          **76.31%** |

DINOv2 achieved **5.46 percentage points higher Top-1 accuracy** and **5.06 percentage points higher Top-5 accuracy** than MobileNetV2 on the final test set.

### 6.3 Computational Efficiency

| Metric             |         MobileNetV2 |     DINOv2 (ViT-S/14) |
| ------------------ | ------------------: | --------------------: |
| Parameter Count    | 2,626,106 (~2.62 M) | 21,878,714 (~21.87 M) |
| Computational Cost |    **0.653 GFLOPs** |         11.050 GFLOPs |
| Average Latency    |   **6.41 ms/image** |         7.87 ms/image |

DINOv2 required approximately **16.9× more theoretical compute** than MobileNetV2:

`11.050 / 0.653 ≈ 16.9`

Despite this large difference in theoretical computational cost, the measured GPU latency difference was relatively small.

---

## 7. Final Analysis

The experiments demonstrate a clear accuracy–efficiency trade-off between the two architectures.

DINOv2 achieved higher final test performance, with **43.18% Top-1** and **76.31% Top-5** accuracy, compared with **37.72% Top-1** and **71.25% Top-5** for MobileNetV2.

However, DINOv2 required substantially more computational resources, with approximately **21.87 M parameters** and **11.050 GFLOPs**, compared with **2.62 M parameters** and **0.653 GFLOPs** for MobileNetV2. This corresponds to approximately **16.9× higher theoretical computational cost** for DINOv2.

The relatively small difference in measured GPU latency despite the large FLOPs difference also demonstrates that theoretical computational complexity does not directly translate to practical inference latency.

---

## 8. Conclusion

The project developed and evaluated a complete **314-class scene classification pipeline** using two contrasting deep learning architectures.

**DINOv2 (ViT-S/14)** achieved the strongest classification performance, reaching **43.18% Top-1** and **76.31% Top-5** accuracy on the final test set. Its stronger performance demonstrates the effectiveness of rich pretrained Transformer representations for complex scene recognition.

**MobileNetV2** achieved **37.72% Top-1** and **71.25% Top-5** accuracy while requiring only **0.653 GFLOPs** and approximately **2.62 million parameters**.

Overall, the project demonstrates an end-to-end workflow covering **dataset preparation, model selection, transfer learning, training, evaluation, computational profiling, and full-stack deployment**. The results highlight the practical trade-off between classification performance and computational efficiency when selecting a model for real-world scene classification.
