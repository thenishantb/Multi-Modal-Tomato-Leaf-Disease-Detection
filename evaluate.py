import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms
from transformers import CLIPTokenizer
from model_arch import MultiModalNet
from train import PlantDataset # Imports the dataset class from your train file
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize
import os

# --- CONFIGURATION ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def evaluate_model():
    print(f"📊 Running Real-World Evaluation on {DEVICE}...")
    
    # 1. Verification
    if not os.path.exists("data/val_split.csv") or not os.path.exists("models/plant_model.pth"):
        print("❌ Error: Missing val_split.csv or model. Please run train.py first.")
        return

    # 2. Load the Hidden Validation Data and Class Map
    df_val = pd.read_csv("data/val_split.csv")
    df_classes = pd.read_csv("data/classes.csv")
    label_map = {row['label']: row['id'] for _, row in df_classes.iterrows()}
    class_names = list(label_map.keys())
    num_classes = len(class_names)
    
    tfms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    
    # Load ONLY the hidden validation data
    val_ds = PlantDataset(df_val, tokenizer, tfms, label_map=label_map)
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False)
    
    # 3. Load Model
    model = MultiModalNet(num_classes=num_classes).to(DEVICE)
    model.load_state_dict(torch.load("models/plant_model.pth", map_location=DEVICE))
    model.eval()

    all_preds, all_labels, all_probs = [], [], []

    # 4. Run Inference on Unseen Data
    print("Testing on unseen validation dataset...")
    with torch.no_grad():
        for img, ids, mask, sens, lbl in val_loader:
            img, ids, mask, sens = img.to(DEVICE), ids.to(DEVICE), mask.to(DEVICE), sens.to(DEVICE)
            logits = model(img, ids, mask, sens)
            probs = torch.nn.functional.softmax(logits, dim=1)
            preds = logits.argmax(dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(lbl.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)

    # Clean the class names for the graphs and reports
    clean_names = [name.replace("Tomato___", "").replace("_", " ") for name in class_names]

    # ---------------------------------------------------------
    # Generate Output 1: Accurate Classification Report
    # ---------------------------------------------------------
    print("\n" + "="*60)
    print("ACCURATE CLASSIFICATION REPORT (Precision, Recall, F1)")
    print("="*60)
    print(classification_report(all_labels, all_preds, target_names=clean_names, zero_division=0))

    # ---------------------------------------------------------
    # Generate Output 2: Confusion Matrix Heatmap
    # ---------------------------------------------------------
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=clean_names, yticklabels=clean_names)
    plt.title('Confusion Matrix (Validation Data)')
    plt.ylabel('Actual Disease')
    plt.xlabel('Predicted Disease')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    plt.close()

    # ---------------------------------------------------------
    # Generate Output 3: Multi-Class ROC-AUC Curve
    # ---------------------------------------------------------
    bin_labels = label_binarize(all_labels, classes=range(num_classes))
    plt.figure(figsize=(10, 8))
    
    for i in range(num_classes):
        if np.sum(bin_labels[:, i]) > 0: # Ensure the class actually exists in the validation split
            fpr, tpr, _ = roc_curve(bin_labels[:, i], all_probs[:, i])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, lw=2, label=f'{clean_names[i]} (AUC = {roc_auc:.2f})')

    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC Curve)')
    plt.legend(loc="lower right", fontsize='small')
    plt.grid(alpha=0.3)
    plt.savefig('roc_curve.png')
    plt.close()

    print("\n✅ All evaluation metrics and graphs generated successfully:")
    print("- confusion_matrix.png")
    print("- roc_curve.png")

if __name__ == "__main__":
    evaluate_model()