#!/bin/bash
echo "Running baseline..."
python src/baseline.py

echo "Training LSTM..."
python src/train.py

echo "Evaluating..."
python src/evaluate.py
