# Group 7

## 📌 Project Overview
This project predicts **road accident severity** using UK smart-city traffic datasets.  
We follow the Data Mining course project requirements and implement:

- A simple baseline model (`Logistic Regression`, `Decision Tree`, `Random Forest`, `KNN`)
- An advanced extension (`Gradient Boosting`, `XGBoost`)
- Full evaluation: accuracy metrics, model comparison chart, modular pipeline

This repository includes code, dataset handling, model training scripts, evaluation tools, and a final report-ready output.

---

## 📁 Repository Structure

```
project/
├── data/                          # data folder
|   └── link.md                    # raw data's link
|
├── notebooks/                     # Jupyter notebooks folder
│   └── placeholder.ipynb          # Placeholder notebook (replace with actual analysis)
|
├── reports/                       # figures & contribution_log
|   ├── figures/                   # dataset & features & results diagrams
|   └── contribution_log.md        # three weeks' contribution
|                         
├── src/
│   ├── __init__.py                # Package initializer
|   ├── baseline.py                # Basic and Advanced ML models (Logistic, DT, RF, GBDT, XGBoost)
|   ├── data_process.py            # Data loading, cleaning, encoding, feature selection
|   ├── evaluate.py                # Model evaluation and accuracy comparison plots
|   ├── figure.py                  # Advanced visualization (efficiency plot, feature lollipop)
|   ├── train.py                   # Full training pipeline (final reproducible version)
|   ├── train_fast.py              # Lightweight training pipeline (5% sampling for fast testing)
|   └── README.md                  # Description of all modules in this folder
|
README.md                          # Main project description and instructions. You are here*
|
requirements.txt                   # Python package dependencies
|
run.sh                             # Shell script for one-click pipeline execution
```

---

## 🚀 Quickstart

### 1. Install dependencies

pip install -r requirements.txt

### 2. Run full pipeline

bash run.sh

This will automatically:

Load and clean the dataset

Train both baseline and advanced models

Evaluate accuracy

Save a comparison chart to reports/figures/accuracy_comparison.png

---

## 👥 Group Members
- 张靖琳 - Project Lead
- 费力勤 - Data Lead
- 惠志文 - Modeling Lead  
- 缪健铭 - Evaluation Lead  
- 宋知恒 - Product/Comms Lead  
  
---

## 📜 License
MIT License
