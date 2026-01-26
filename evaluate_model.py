"""
Multilingual Sentiment Analysis Evaluation Framework.

This script loads our fine-tuned BERT model and evaluates its performance
on the validation dataset. It generates key metrics (Accuracy, F1)
and a suite of research-grade visualization plots for publication.

The model architecture is based on 'bert-base-multilingual-uncased',
optimized for 5-class sentiment classification.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay, f1_score, accuracy_score,
    roc_curve, auc, precision_recall_fscore_support, precision_recall_curve, average_precision_score
)
from sklearn.preprocessing import label_binarize

# ==========================================
# 1. Validation Data
# ==========================================
# Representative samples from our test set
VALIDATION_DATA = [
    ("I absolutely love this product, it works perfectly!", 5),
    ("The item arrived broken and customer service was rude.", 1),
    ("It's okay, does what it says but nothing special.", 3),
    ("Good value for money, but delivery was slow.", 4),
    ("Terrible. Do not buy.", 1),
    ("Excellent quality and fast shipping.", 5),
    ("Not bad, but I've seen better.", 3),
    ("Complete waste of money.", 1),
    ("Highly recommended!", 5),
    ("Average performance.", 3)
]

# Sentiment Classes: 1 to 5 stars
CLASSES = [1, 2, 3, 4, 5]
MODEL_PATH = "./model"

def main():
    """
    Main execution pipeline for model evaluation.
    """
    print(f"Loading our fine-tuned model from {MODEL_PATH}...")
    try:
        # Load the custom fine-tuned weights
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
    except Exception as e:
        print(f"Error loading model weights: {e}")
        return

    print("Model initialized. Starting inference on validation set...")

    y_true = []
    y_pred = []
    y_prob = []

    # Inference Loop
    for text, label in VALIDATION_DATA:
        # Tokenize using our model's vocabulary
        inputs = tokenizer(text, return_tensors="pt")
        
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            # Calculate probabilities for analysis
            probs = torch.nn.functional.softmax(logits, dim=1).numpy()[0]
            pred_idx = np.argmax(probs)
            pred_label = pred_idx + 1

        y_true.append(label)
        y_pred.append(pred_label)
        y_prob.append(probs)

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_prob = np.array(y_prob)

    # ==========================================
    # 2. Performance Metrics
    # ==========================================
    acc = accuracy_score(y_true, y_pred)
    precision_w, recall_w, f1_w, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted', zero_division=0)
    
    print("-" * 30)
    print(f"Validation Accuracy: {acc:.4f}")
    print(f"Weighted Precision:  {precision_w:.4f}")
    print(f"Weighted Recall:     {recall_w:.4f}")
    print(f"Weighted F1 Score:   {f1_w:.4f}")
    print("-" * 30)

    # ==========================================
    # 3. Visualization Generation
    # ==========================================
    
    # --- A. Confusion Matrix ---
    print("Generating Confusion Matrix plot...")
    cm = confusion_matrix(y_true, y_pred, labels=CLASSES)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASSES)
    
    plt.figure(figsize=(8, 6))
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix: Predicted vs True Ratings")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png")
    plt.close()

    # --- B. ROC Curve ---
    print("Generating ROC Curve plot...")
    y_true_bin = label_binarize(y_true, classes=CLASSES)
    n_classes = len(CLASSES)

    plt.figure(figsize=(10, 8))
    for i in range(n_classes):
        if np.sum(y_true_bin[:, i]) == 0:
            continue
            
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_prob[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, label=f'Rating {CLASSES[i]} (AUC = {roc_auc:.2f})')

    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Multi-class ROC Analysis')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("roc_curve.png")
    plt.close()

    # --- C. Precision-Recall Curve ---
    print("Generating Precision-Recall Curve plot...")
    plt.figure(figsize=(10, 8))
    for i in range(n_classes):
        if np.sum(y_true_bin[:, i]) == 0:
            continue
        
        precision, recall, _ = precision_recall_curve(y_true_bin[:, i], y_prob[:, i])
        avg_precision = average_precision_score(y_true_bin[:, i], y_prob[:, i])
        plt.plot(recall, precision, lw=2, label=f'Rating {CLASSES[i]} (AP = {avg_precision:.2f})')

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve per Class')
    plt.legend(loc="lower left")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("precision_recall_curve.png")
    plt.close()

    # --- D. Class-wise Performance Bar Chart ---
    print("Generating Class-wise Performance chart...")
    precision_c, recall_c, f1_c, _ = precision_recall_fscore_support(y_true, y_pred, labels=CLASSES, average=None, zero_division=0)
    
    x = np.arange(len(CLASSES))
    width = 0.25

    plt.figure(figsize=(10, 6))
    plt.bar(x - width, precision_c, width, label='Precision')
    plt.bar(x, recall_c, width, label='Recall')
    plt.bar(x + width, f1_c, width, label='F1 Score')

    plt.ylabel('Score')
    plt.xlabel('Rating Class')
    plt.title('Performance Metrics by Class')
    plt.xticks(x, [str(c) for c in CLASSES])
    plt.ylim([0, 1.1])
    plt.legend()
    plt.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig("class_performance.png")
    plt.close()

    # --- E. Confidence Histogram ---
    print("Generating Confidence Histogram...")
    # Get the probability of the predicted class for each sample
    max_probs = np.max(y_prob, axis=1)
    
    plt.figure(figsize=(8, 6))
    plt.hist(max_probs, bins=10, range=(0, 1), color='skyblue', edgecolor='black')
    plt.xlabel('Prediction Confidence')
    plt.ylabel('Number of Samples')
    plt.title('Model Prediction Confidence Distribution')
    plt.xlim([0, 1])
    plt.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig("confidence_histogram.png")
    plt.close()

    # --- F. Overall Metrics Summary Chart ---
    print("Generating Overall Metrics Summary chart...")
    metrics = ['Accuracy', 'Weighted Precision', 'Weighted Recall', 'Weighted F1']
    values = [acc, precision_w, recall_w, f1_w]
    
    plt.figure(figsize=(8, 6))
    bars = plt.bar(metrics, values, color=['#4CAF50', '#2196F3', '#FF9800', '#F44336'])
    plt.ylim([0, 1.1])
    plt.title('Overall Model Performance Metrics')
    plt.ylabel('Score')
    
    # Add text labels on bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                 f'{height:.4f}',
                 ha='center', va='bottom')
                 
    plt.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig("metrics_summary.png")
    plt.close()

    print("Evaluation complete. All research plots saved to project root.")

if __name__ == "__main__":
    main()
