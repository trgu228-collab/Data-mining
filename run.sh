#!/bin/bash
set -e  # Stop the script if any command fails

echo "========================================="
echo "   G7 Traffic Accident Severity Pipeline   "
echo "========================================="

# Activate conda if running locally (ignored on GitHub)
# source ~/anaconda3/etc/profile.d/conda.sh
# conda activate data-mining

echo ""
echo "Step 1/4: Running Baseline Models..."
python src/baseline.py

echo ""
echo "Step 2/4: Running Advanced Models..."
python src/advanced.py

echo ""
echo "Step 3/4: Training Final Model..."
python src/train.py

echo ""
echo "Step 4/4: Evaluating Models..."
python src/evaluate.py

echo ""
echo "========================================="
echo "Pipeline Complete! All results generated."
echo "========================================="

