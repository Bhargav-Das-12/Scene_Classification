# Scene Classification

_This project was developed during an internship at EICT IIT Guwahati under the guidance of Prithvijit Guha Sir._

![Python](https://img.shields.io/badge/Python-3.11-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-green.svg)
![React](https://img.shields.io/badge/React-Vite-61DAFB.svg)

An advanced deep learning system designed for multi-model scene classification across **314 distinct environments**, using a curated subset of the MIT Places365 dataset.

The project benchmarks a lightweight Convolutional Neural Network, **MobileNetV2**, against a Vision Transformer, **DINOv2 (ViT-S/14)**. The system also provides a full-stack web application for real-time dual-model inference and side-by-side comparison of scene predictions.

---

## Project Resources

- **[Read the Full Project Report Here](./project_report.md)** - Detailed methodology, results, and pipeline.
- **Progress Report – Notion:** [View Progress Report on Notion](https://app.notion.com/p/Scene-Classification-38594c0acda780a2afbfce1794ce6a36?source=copy_link)
- **[View Source Code on GitHub](https://github.com/Bhargav-Das-12/Scene_Classification)** - Access the complete repository, scripts, and datasets.
- **Video Demonstration:**
<br>
<iframe width="800" height="450" src="https://www.youtube.com/embed/Jz3pot37TLA?si=ny3WboK9Q2OmzGEG" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>

---

## Overview

The project evaluates two fundamentally different approaches to scene classification:

- **MobileNetV2** — Lightweight CNN optimized for computational efficiency.
- **DINOv2 (ViT-S/14)** — Vision Transformer used as a powerful visual feature extractor with a custom classification head.

Both models are trained/evaluated on the same **314-class scene classification task** and integrated into a common inference pipeline.

---

## Dataset

The project uses a curated subset of **MIT Places365** containing **314 scene categories**.

- **Raw Dataset:** [Kaggle Places365 Dataset](https://www.kaggle.com/datasets/pankajkumar2002/places365)
- **Processed Dataset:** [Organized Dataset – Google Drive](https://drive.google.com/file/d/1Hh9pWRZChD8zzoSHDsptuxQzO1qCtkSA/view?usp=drive_link)

_The processed dataset is organized into PyTorch-ready class directories for the train, validation, and test splits._

---

## Final Results

The models were evaluated on a held-out test set of **7,302 images** across 314 scene categories.

### Classification Performance

| Model                 | Top-1 Accuracy | Top-5 Accuracy |
| --------------------- | -------------: | -------------: |
| **MobileNetV2**       |     **37.72%** |     **71.25%** |
| **DINOv2 (ViT-S/14)** |     **43.18%** |     **76.31%** |

### Computational Efficiency

| Metric              |       MobileNetV2 | DINOv2 (ViT-S/14) |
| ------------------- | ----------------: | ----------------: |
| **Parameters**      |           ~2.62 M |          ~21.88 M |
| **FLOPs**           |  **0.653 GFLOPs** |     11.050 GFLOPs |
| **Average Latency** | **6.41 ms/image** |     7.87 ms/image |

DINOv2 achieves higher classification accuracy, while MobileNetV2 provides a substantially lower computational cost and smaller model footprint.

---

## Getting Started

### Prerequisites

Make sure the following are installed:

- Python 3.11
- pip
- Node.js 16+
- npm
- Git

### 1. Clone the Repository

```bash
git clone https://github.com/YourUsername/Scene-Classification.git
cd Scene-Classification
```

> Replace `YourUsername` with the actual GitHub username/repository URL.

### 2. Create the Python Environment

```bash
python -m venv venv
```

#### Windows PowerShell

```powershell
.\venv\Scripts\Activate.ps1
```

#### macOS/Linux

```bash
source venv/bin/activate
```

Install the required Python packages:

```bash
pip install -r requirements.txt
```

### 3. Backend Setup

Navigate to the backend:

```bash
cd backend
```

Make sure the trained `.pth` model weights are present in the `backend` directory.

Start the FastAPI server:

```bash
uvicorn main:app --reload
```

The backend will be available at:

```text
http://127.0.0.1:8000
```

The models are initialized during application startup.

### 4. Frontend Setup

Open a separate terminal and navigate to the frontend:

```bash
cd frontend
```

Install the dependencies:

```bash
npm install
```

Start the development server:

```bash
npm run dev
```

The React application will normally be available at:

```text
http://localhost:5173
```

---

## Project Structure

```text
Scene_Classification/
│
├── backend/
│   ├── main.py
│   ├── best_scene_classifier.pth
│   └── mobilenet_scene_classifier-final.pth
│
├── frontend/
│   ├── src/
│   │   └── App.jsx
│   └── package.json
│
├── notebooks/
│   ├── MobileNet_model_training-final.ipynb
│   └── DINOV2_model_training.ipynb
│
├── results/
│   ├── mobile_net_main.png
│   └── dinov2_main.png
│
├── Scene_Classification_Progress_Report.pdf
├── README.md
├── requirements.txt
├── .gitignore
└── LICENSE
```

### Directory Description

| Directory/File                             | Description                                                              |
| ------------------------------------------ | ------------------------------------------------------------------------ |
| `backend/`                                 | FastAPI backend and PyTorch inference pipeline                           |
| `frontend/`                                | React/Vite web application                                               |
| `notebooks/`                               | Dataset preparation, model training, evaluation, and profiling notebooks |
| `results/`                                 | Training accuracy and loss curves                                        |
| `*.pth`                                    | Trained model weights                                                    |
| `Scene_Classification_Progress_Report.pdf` | Detailed project progress report                                         |
| `requirements.txt`                         | Python dependencies                                                      |
| `.gitignore`                               | Files excluded from version control                                      |

---

## Author

**[Bhargav Das](https://github.com/Bhargav-Das-12)**

---

## License

MIT License.
