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

data/
├── raw/ # Original CSV dataset (not tracked by Git)
├── processed/ # Cleaned / transformed data (optional)

notebooks/
├── placeholder.ipynb # Initial exploration / prototyping

src/
├── baseline.py # Basic ML models (Logistic, DT, RF, KNN)
├── advanced.py # Advanced ML models (GBDT, XGBoost)
├── data_utils.py # Data loading, cleaning, feature extraction
├── train.py # Train models, scaling, pipeline logic
├── evaluate.py # Accuracy computation, bar chart generation
├── main.py # Master script: run full pipeline
├── init.py # Package marker

reports/
├── figures/ # Accuracy comparison figure saved here

requirements.txt # Python dependencies
run.sh # One-click shell script to run pipeline
README.md # This file
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
- 缪健铭 - Evaluation Lead  
- 惠志文 - Modeling Lead  
- 张靖琳 - Project Lead 
- 宋知恒 - Evaluation Lead  
- 费力勤 - Product/Comms Lead  

---

## 📜 License
MIT License
