import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from PIL import Image
from torchvision import transforms
from transformers import CLIPTokenizer
from sklearn.model_selection import train_test_split
from model_arch import MultiModalNet # Ensure this matches your architecture file name
import os
import matplotlib.pyplot as plt

# --- CONFIGURATION ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS = 10
BATCH_SIZE = 16
LR = 1e-4

# --- DATASET CLASS ---
class PlantDataset(Dataset):
    def __init__(self, dataframe, tokenizer, transform, label_map=None):
        self.df = dataframe.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.transform = transform
        
        # Create or use existing label map
        if label_map is None:
            self.label_map = {lbl: i for i, lbl in enumerate(self.df['label'].unique())}
        else:
            self.label_map = label_map
            
        self.class_names = list(self.label_map.keys())

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        try:
            img = Image.open(row['path']).convert("RGB")
        except:
            img = Image.new('RGB', (224, 224), color='black') # Fallback for corrupt images
            
        img = self.transform(img)
        text = str(row['text'])
        tokens = self.tokenizer(text, padding='max_length', max_length=77, truncation=True, return_tensors="pt")
        sensors = torch.tensor([row['temp'], row['hum'], row['rain']], dtype=torch.float32)
        label = torch.tensor(self.label_map[row['label']], dtype=torch.long)
        
        return img, tokens['input_ids'].squeeze(0), tokens['attention_mask'].squeeze(0), sensors, label

# --- MAIN TRAINING SCRIPT ---
def train():
    print(f"🚀 Training on {DEVICE}...")
    
    # 1. Load Data and Split it physically to prevent data leakage
    if not os.path.exists("data/train_data.csv"):
        print("❌ Error: 'data/train_data.csv' not found.")
        return

    df_full = pd.read_csv("data/train_data.csv")
    
    # Stratified split ensures equal class representation in both sets
    df_train, df_val = train_test_split(df_full, test_size=0.2, random_state=42, stratify=df_full['label'])
    
    # Save the validation set so evaluate.py can use it later
    df_val.to_csv("data/val_split.csv", index=False)
    print(f"Data Split: {len(df_train)} Training | {len(df_val)} Validation (Saved to data/val_split.csv)")
    
    # 2. Setup Transforms and Tokenizer
    tfms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    
    # 3. Create Datasets
    train_ds = PlantDataset(df_train, tokenizer, tfms)
    val_ds = PlantDataset(df_val, tokenizer, tfms, label_map=train_ds.label_map)
    
    # Save class map for the App and Evaluation script
    map_df = pd.DataFrame(list(train_ds.label_map.items()), columns=['label', 'id'])
    map_df.to_csv("data/classes.csv", index=False)
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    
    # 4. Initialize Model (Using weight decay to prevent overfitting)
    model = MultiModalNet(num_classes=len(train_ds.class_names)).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4) 
    criterion = nn.CrossEntropyLoss()
    
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    
    # 5. Training Loop
    for epoch in range(EPOCHS):
        # Training Phase
        model.train()
        train_loss, train_correct, train_total = 0, 0, 0
        for img, ids, mask, sens, lbl in train_loader:
            img, ids, mask, sens, lbl = img.to(DEVICE), ids.to(DEVICE), mask.to(DEVICE), sens.to(DEVICE), lbl.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(img, ids, mask, sens)
            loss = criterion(outputs, lbl)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * img.size(0)
            _, predicted = torch.max(outputs, 1)
            train_total += lbl.size(0)
            train_correct += (predicted == lbl).sum().item()
            
        # Validation Phase
        model.eval()
        val_loss, val_correct, val_total = 0, 0, 0
        with torch.no_grad():
            for img, ids, mask, sens, lbl in val_loader:
                img, ids, mask, sens, lbl = img.to(DEVICE), ids.to(DEVICE), mask.to(DEVICE), sens.to(DEVICE), lbl.to(DEVICE)
                outputs = model(img, ids, mask, sens)
                loss = criterion(outputs, lbl)
                
                val_loss += loss.item() * img.size(0)
                _, predicted = torch.max(outputs, 1)
                val_total += lbl.size(0)
                val_correct += (predicted == lbl).sum().item()
                
        # Record Metrics
        history['train_loss'].append(train_loss / train_total)
        history['val_loss'].append(val_loss / val_total)
        history['train_acc'].append(train_correct / train_total)
        history['val_acc'].append(val_correct / val_total)
        
        print(f"Epoch {epoch+1}/{EPOCHS} | Train Acc: {history['train_acc'][-1]:.4f} | Val Acc: {history['val_acc'][-1]:.4f}")
        
    # 6. Save Model
    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), "models/plant_model.pth")
    print("🎉 Model Saved to 'models/plant_model.pth'")

    # 7. Generate Accuracy & Loss Graphs
    epochs_range = range(1, EPOCHS + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    ax1.plot(epochs_range, history['train_acc'], label='Train Acc', marker='o')
    ax1.plot(epochs_range, history['val_acc'], label='Val Acc', marker='s')
    ax1.set_title('Training and Validation Accuracy')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True)

    ax2.plot(epochs_range, history['train_loss'], label='Train Loss', marker='o')
    ax2.plot(epochs_range, history['val_loss'], label='Val Loss', marker='s')
    ax2.set_title('Training and Validation Loss')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('accuracy_loss_graphs.png')
    print("✅ Saved 'accuracy_loss_graphs.png'")

if __name__ == "__main__":
    train()