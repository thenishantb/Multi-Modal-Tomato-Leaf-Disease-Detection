import torch
import torch.nn as nn
import torchvision.models as models
from transformers import CLIPTextModel

class MultiModalNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        
        # 1. Vision Encoder (ResNet50)
        # We use ResNet50 for image feature extraction
        self.vision = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.vision.fc = nn.Identity() # Remove the final classification layer
        # Output dim is 2048
        
        # 2. Text Encoder (CLIP)
        # We use CLIP to understand text context
        self.text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32")
        # Freeze weights to save memory and speed up training
        for param in self.text_encoder.parameters():
            param.requires_grad = False
        self.text_proj = nn.Linear(512, 512)
        
        # 3. Sensor Encoder (MLP)
        # Simple Neural Network for numbers (Temp, Hum, Rain)
        self.sensor_mlp = nn.Sequential(
            nn.Linear(3, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU()
        )
        
        # 4. Fusion Layer (Combine all 3)
        # 2048 (Image) + 512 (Text) + 128 (Sensor) = 2688
        self.fusion = nn.Sequential(
            nn.Linear(2048 + 512 + 128, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, num_classes)
        )
        
    def forward(self, img, input_ids, attn_mask, sensors):
        # Vision Stream
        v_feat = self.vision(img) # [Batch, 2048]
        
        # Text Stream
        t_out = self.text_encoder(input_ids=input_ids, attention_mask=attn_mask).last_hidden_state
        t_feat = self.text_proj(t_out[:, 0, :]) # Use [CLS] token
        
        # Sensor Stream
        s_feat = self.sensor_mlp(sensors)
        
        # Fuse
        fused = torch.cat([v_feat, t_feat, s_feat], dim=1)
        
        # Predict
        return self.fusion(fused)