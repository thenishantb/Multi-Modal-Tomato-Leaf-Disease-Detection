import chainlit as cl
import torch
import pandas as pd
from PIL import Image
import io
from torchvision import transforms
from model_arch import MultiModalNet

# CONFIG
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Load Class Names
try:
    df_classes = pd.read_csv("data/classes.csv")
    ID_TO_LABEL = {row['id']: row['label'] for _, row in df_classes.iterrows()}
    NUM_CLASSES = len(ID_TO_LABEL)
except:
    print("⚠️ Warning: classes.csv not found. Did you run train.py?")
    ID_TO_LABEL = {0: "Unknown"}
    NUM_CLASSES = 10 

# KNOWLEDGE BASE
DISEASE_INFO = {
    "Healthy": "The plant looks healthy! Keep up the good work with regular watering and proper sunlight.",
    "Early blight": "Caused by a fungus. Symptoms include brown, bulls-eye shaped spots on lower leaves. Spreads in warm, wet weather. Remove infected leaves and avoid overhead watering.",
    "Late blight": "A severe disease caused by a water mold. Causes large, dark, water-soaked patches. Thrives in cool, wet conditions. Can destroy a crop quickly; apply fungicides early.",
    "Leaf Mold": "Appears as pale green or yellow spots on the upper leaf surface, with olive-green mold on the underside. Common in high humidity environments like greenhouses.",
    "Septoria leaf spot": "Fungal disease causing numerous small, circular spots with dark borders and gray centers. Prune lower leaves to improve air circulation.",
    "Spider mites Two-spotted spider mite": "Tiny pests that cause leaves to look stippled, yellow, and dry. Look for fine webbing. They thrive in hot, dry conditions. Use neem oil or miticides.",
    "Target Spot": "Fungal disease showing brown lesions with yellow halos. Often affects plants with poor air circulation. Ensure proper plant spacing.",
    "Tomato Yellow Leaf Curl Virus": "Transmitted by whiteflies. Leaves curl upward, turn yellow, and plant growth is stunted. Control whitefly populations to manage this.",
    "Tomato mosaic virus": "Causes mottled light and dark green patterns on leaves. It is highly contagious and spreads via contaminated tools or hands. Disinfect tools and wash hands frequently.",
    "Bacterial spot": "Small, dark, water-soaked spots on leaves and fruit. Spreads rapidly in warm, rainy weather. Avoid working with plants when they are wet."
}

@cl.on_chat_start
async def start():
    settings = await cl.ChatSettings([
        cl.input_widget.Slider(id="temp", label="Temperature (°C)", initial=25, min=0, max=50, step=1),
        cl.input_widget.Slider(id="hum", label="Humidity (%)", initial=60, min=0, max=100, step=5),
        cl.input_widget.Slider(id="rain", label="Rainfall (mm)", initial=0, min=0, max=100, step=5),
    ]).send()
    
    model = MultiModalNet(num_classes=NUM_CLASSES)
    try:
        model.load_state_dict(torch.load("models/plant_model.pth", map_location=DEVICE))
        print("✅ Model loaded successfully.")
    except:
        print("⚠️ Model file not found. Please run train.py first.")
    
    model.to(DEVICE).eval()
    cl.user_session.set("model", model)
    cl.user_session.set("sensors", [25, 60, 0]) 

    await cl.Message(
        content="""
        🌿 **Plant Doctor AI is Ready!**
        
        1. **Set Weather:** Use the sliders below to match current conditions.
        2. **Upload:** Drag & Drop a leaf photo.
        """
    ).send()

@cl.on_settings_update
async def update_sensors(settings):
    cl.user_session.set("sensors", [settings["temp"], settings["hum"], settings["rain"]])

@cl.on_message
async def main(message: cl.Message):
    model = cl.user_session.get("model")
    sensors = cl.user_session.get("sensors")
    
    if not message.elements:
        await cl.Message("📷 **Please upload an image.**").send()
        return
    
    image_bytes = None
    for el in message.elements:
        if "image" in el.mime:
            path = el.path if hasattr(el, 'path') else None
            if path: image_bytes = open(path, "rb").read()
            else: image_bytes = el.content
            break
            
    if not image_bytes: 
        await cl.Message("⚠️ Could not read image.").send()
        return

    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    
    tfms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    img_tensor = tfms(img).unsqueeze(0).to(DEVICE)
    sensor_tensor = torch.tensor(sensors, dtype=torch.float32).unsqueeze(0).to(DEVICE)
    
    dummy_ids = torch.zeros((1, 77), dtype=torch.long).to(DEVICE)
    dummy_mask = torch.ones((1, 77), dtype=torch.long).to(DEVICE)

    with torch.no_grad():
        logits = model(img_tensor, dummy_ids, dummy_mask, sensor_tensor)
        probabilities = torch.nn.functional.softmax(logits, dim=1)

    probs = probabilities.cpu().numpy()[0]

    # 🔥 SENSOR BOOSTING
    temp, hum, rain = sensors

    if hum > 80:
        for idx, label in ID_TO_LABEL.items():
            if "Late blight" in label:
                probs[idx] += 0.10

    if temp > 30 and hum < 40:
        for idx, label in ID_TO_LABEL.items():
            if "Spider mites" in label or "Two-spotted" in label:
                probs[idx] += 0.10

    probs = probs / probs.sum()

    # 🔥 TOP-2 PREDICTIONS
    top_indices = probs.argsort()[-2:][::-1]

    top1_idx = top_indices[0]
    top2_idx = top_indices[1]

    conf1 = probs[top1_idx]
    conf2 = probs[top2_idx]

    raw_label = ID_TO_LABEL.get(top1_idx, "Unknown")
    second_label = ID_TO_LABEL.get(top2_idx, "Unknown")

    clean_label = raw_label.replace("Tomato___", "").replace("_", " ").strip()
    second_clean = second_label.replace("Tomato___", "").replace("_", " ").strip()

    disease_description = DISEASE_INFO.get(clean_label, "No additional information is available.")
    second_description = DISEASE_INFO.get(second_clean, "")

    # 🔥 SMART RESPONSE
    if conf1 < 0.5:
        response = f"""
### ⚠️ Low Confidence Detection

🤖 The model is not fully confident.

### Possible Diseases:
1️⃣ **{clean_label}** ({conf1*100:.1f}%)
2️⃣ **{second_clean}** ({conf2*100:.1f}%)

📖 **About {clean_label}:**
{disease_description}

---

🌡️ **Weather Context Used:**
* Temp: {sensors[0]}°C | Humidity: {sensors[1]}% | Rain: {sensors[2]}mm

💡 *Tip: Please verify symptoms manually.*
"""
    else:
        response = f"""
### 🩺 Diagnosis: **{clean_label}**
*(Confidence: {conf1*100:.1f}%)*

---

📖 **About this condition:**
{disease_description}

---

🌡️ **Weather Context Used:**
* Temp: {sensors[0]}°C | Humidity: {sensors[1]}% | Rain: {sensors[2]}mm
"""

    await cl.Message(content=response).send()