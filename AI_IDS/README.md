# AI-Powered Intrusion Detection System (AI-IDS)

This project implements a hybrid Machine Learning and Deep Learning-based Intrusion Detection System (IDS) using the NSL-KDD dataset. Models include Random Forest, LSTM, CNN, and an Ensemble model.

## 📁 Project Structure
- `src/` — modular Python scripts for preprocessing, training, and evaluation
- `notebooks/` — exploratory notebook (original `AI_IDS.ipynb`)
- `models/` — optional saved models (not included)
- `data/` — NSL-KDD dataset (not included in repo, see instructions)
- `requirements.txt` — dependencies list
- `.gitignore` — excludes temporary and environment files

## 🚀 How to Run

1. Clone the repo:
   ```bash
   git clone https://github.com/your-username/ai-ids.git
   cd ai-ids
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run:
   ```bash
   python src/train_models.py
   python src/evaluate_models.py
   ```

## 🧪 Models Used
- Random Forest
- Long Short-Term Memory (LSTM)
- Convolutional Neural Network (CNN)
- Ensemble (Voting-based)

## 📊 Metrics
- F1-Score
- ROC-AUC
- Confusion Matrix

## 📁 Dataset
The NSL-KDD dataset is used for training and evaluation. Download it from:
https://github.com/defcom17/NSL_KDD

## 📜 Paper
This code supports the IEEE paper:
> “AI-Powered Intrusion Detection System Using Machine Learning and Deep Learning Techniques” — Tejaswi Vemuri (2025)

## 🧠 Author
Tejaswi Vemuri — [LinkedIn](https://linkedin.com) | [GitHub](https://github.com/tejaswivemuri)

## 📄 License
MIT License
