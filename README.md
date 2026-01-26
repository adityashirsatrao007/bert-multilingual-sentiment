# Multilingual Product Sentiment Analysis with BERT

## Installation

To reproduce our results or use the model for inference:

1.  **Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

Run the evaluation script to test the model on validation data and generate performance plots:

```bash
python evaluate_model.py
```

## Results

# Multilingual Product Sentiment Analysis with BERT

## Installation

To reproduce our results or use the model for inference:

1.  **Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

Run the evaluation script to test the model on validation data and generate performance plots:

```bash
python evaluate_model.py
```

## Results

Our evaluation on the test set demonstrates strong performance:

- **Accuracy**: 90.00%
- **F1 Score (Weighted)**: 0.8571

### Visualization

The evaluation script automatically generates:

- `confusion_matrix.png`: Detailed breakdown of class-wise predictions.
- `roc_curve.png`: One-vs-Rest ROC curves for multi-class performance analysis.
- `precision_recall_curve.png`: Precision vs Recall trade-off analysis for each class.
- `class_performance.png`: Bar chart comparing Precision, Recall, and F1 across all classes.
- `confidence_histogram.png`: Distribution of model prediction confidence.
- `metrics_summary.png`: Summary of overall model performance metrics.
