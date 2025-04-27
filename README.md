
# AI-Powered Intrusion Detection System (AI-IDS)

This project implements an AI-powered Intrusion Detection System (AI-IDS) that identifies and mitigates network security threats using both traditional Machine Learning (ML) and Deep Learning (DL) techniques.
The models are trained and evaluated on the NSL-KDD dataset.

## 📂 Project Structure

```
ai-ids/
├── notebooks/
│   └── AI_IDS_full_script.ipynb
├── src/
│   ├── main.py
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── evaluation.py
│   ├── utils.py
│   └── models/
│       ├── random_forest.py
│       ├── lstm.py
│       ├── cnn.py
│       └── ensemble.py
├── README.md
├── requirements.txt
```

## ⚡ Features

- Train and evaluate Random Forest, LSTM, CNN, and an Ensemble model.
- Confusion matrix and evaluation metrics for each model.
- Modular code structure for easy extension.

## 🏗️ Models Used

| Model | Purpose |
|-------------------|---------|
| Random Forest | Baseline classical machine learning |
| LSTM (Bi-directional) | Sequence modeling of network traffic |
| CNN (1D) | Spatial pattern recognition in network features |
| Ensemble Voting | Combines all models for robust detection |

## 🧹 Installation

1. Clone the repository:
```bash
git clone https://github.com/your-username/ai-ids.git
cd ai-ids
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## 🛠️ Usage

Run the full training and evaluation pipeline:
```bash
python src/main.py
```

Make sure to update the dataset paths (`train_path`, `test_path`) inside `main.py`.

## 📈 Evaluation Metrics

- Confusion Matrix
- F1-Score
- ROC-AUC
- Precision, Recall

## 📊 Dataset

- NSL-KDD Dataset
- [Download link](https://www.unb.ca/cic/datasets/nsl.html)

## 🔥 Future Work

- Expand from binary to multi-class intrusion detection.
- Integrate Transformer-based models.
- Implement adversarial robustness testing.
- Deploy the AI-IDS as a cloud-native service.

## 🙌 Acknowledgments

- NSL-KDD Dataset creators
- Research papers cited for AI-IDS model inspiration

## 🧠 Author
Tejaswi Vemuri — [LinkedIn](https://linkedin.com/in/tejaswi-vemuri-69677a152) | [GitHub](https://github.com/tejaswivemuri)

