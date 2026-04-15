import os
import pandas as pd
import random

# CONFIGURATION
IMAGE_DIR = "data/images" # Point to where you put the class folders
OUTPUT_FILE = "data/train_data.csv"

def get_sensor_data(label):
    """Simulates realistic weather based on disease type."""
    if "Late_blight" in label:
        return random.uniform(10, 20), random.uniform(80, 95), random.uniform(10, 50) # Cold & Wet
    elif "Spider_mites" in label:
        return random.uniform(30, 38), random.uniform(20, 40), 0.0 # Hot & Dry
    elif "Healthy" in label:
        return random.uniform(20, 28), random.uniform(40, 60), random.uniform(0, 5) # Ideal
    else:
        return random.uniform(22, 32), random.uniform(50, 75), random.uniform(0, 10) # Generic

data = []

# Scan folders
if not os.path.exists(IMAGE_DIR):
    print(f"❌ Error: '{IMAGE_DIR}' not found. Please download the dataset first.")
    exit()

classes = [d for d in os.listdir(IMAGE_DIR) if os.path.isdir(os.path.join(IMAGE_DIR, d))]
print(f"Found {len(classes)} classes: {classes}")

for label in classes:
    class_path = os.path.join(IMAGE_DIR, label)
    # Limit to 200 images per class for faster training demo
    for img_name in os.listdir(class_path)[:200]: 
        if img_name.lower().endswith(('.jpg', '.png', '.jpeg')):
            temp, hum, rain = get_sensor_data(label)
            
            data.append({
                "path": os.path.join(class_path, img_name),
                "label": label,
                "text": f"The leaves show signs of {label.replace('Tomato___', '').replace('_', ' ')}",
                "temp": round(temp, 1),
                "hum": round(hum, 1),
                "rain": round(rain, 1)
            })

# Save
df = pd.DataFrame(data)
df.to_csv(OUTPUT_FILE, index=False)
print(f"✅ Dataset generated: {OUTPUT_FILE} ({len(df)} samples)")