#!/bin/bash

echo "==============================="
echo "🚀 Starting full ML pipeline..."
echo "==============================="

# 1) Baseline Models
echo ""
echo "▶ Running baseline models (LR, DT, RF, GBDT, XGBoost)..."
python src/baseline.py

# 2) Data Processing (if needed)
# echo ""
# echo "▶ Running data processing..."
# python src/data_process.py

# 3) Full Training Pipeline (Final Model)
echo ""
echo "▶ Training final model (full pipeline)..."
python src/train.py

# 4) Fast Training (5% sampling for debugging)
echo ""
echo "▶ Running fast training mode (for quick checks)..."
python src/train_fast.py

# 5) Evaluation
echo ""
echo "▶ Evaluating model performance..."
python src/evaluate.py

# 6) Generating Figures
echo ""
echo "▶ Generating all plots and figures..."
python src/figure.py

echo ""
echo "==============================="
echo "🎉 Pipeline completed successfully!"
echo "==============================="


