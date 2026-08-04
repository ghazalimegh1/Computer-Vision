import torch
import torch.nn as nn
from Transformer import Encoder

class ViT(nn.Module):
    def __init__(self, image_size, patch_size, num_layers, d_model, num_heads, d_ff, n_classes):
        super().__init__()
        assert image_size % patch_size == 0
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2
        self.encoder = Encoder(num_layers, d_model, num_heads, d_ff)
        self.patch_embedding = nn.Linear(patch_size * patch_size * 3, d_model)
        self.classifier = nn.Linear(d_model, n_classes)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))  # Class token
        self.positional_encoding = nn.Parameter(torch.zeros(1, self.num_patches + 1, d_model))


    def forward(self, x):
        patches = self._create_patches(x)
        embedded_patches = self.patch_embedding(patches)
        embedded_patches = torch.cat([self.cls_token.expand(embedded_patches.size(0), -1, -1), embedded_patches], dim=1)
        embedded_patches += self.positional_encoding
        encoder_output = self.encoder.forward(embedded_patches)
        cls_token = encoder_output[:, 0, :]  # the class token representation
        logits = self.classifier(cls_token)
        return logits


    def _create_patches(self, x):
        batch_size, channels, height, width = x.size()
        patches = []
        for i in range(0, height, self.patch_size):
            for j in range(0, width, self.patch_size):
                patch = x[: ,:, i:i+self.patch_size, j:j+self.patch_size]
                patches.append(patch)
        patches = torch.stack(patches, dim=1)
        patches = patches.view(batch_size, self.num_patches, -1)  
        return patches

    def predict(self, x):
        with torch.no_grad():
            logits = self.forward(x)
            cls = torch.argmax(logits, dim=1)
            return cls