import torch
from torch import nn

from Transformer import Encoder

class DeiT(nn.Module):
    def __init__(self, image_size, patch_size, num_layers, d_model, num_heads, d_ff, n_classes, distillation=False, teacher_model=None):
        super().__init__()
        assert image_size % patch_size == 0
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2
        self.encoder = Encoder(num_layers, d_model, num_heads, d_ff)
        self.patch_embedding = nn.Linear(patch_size * patch_size * 3, d_model)
        self.cls_classifier = nn.Linear(d_model, n_classes)
        self.distillation_classifier = nn.Linear(d_model, n_classes) if distillation else None
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.distillation = distillation
        self.positional_encoding = nn.Parameter(torch.zeros(1, self.num_patches + 2 if self.distillation else self.num_patches + 1, d_model))
        self.distillation_token = nn.Parameter(torch.zeros(1, 1, d_model))  
        self.teacher_model = teacher_model
        if self.teacher_model is not None:
            self.teacher_model.eval()
            for param in self.teacher_model.parameters():
                param.requires_grad = False
        


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

    def forward(self, x):
        patches = self._create_patches(x)
        embedded_patches = self.patch_embedding(patches)
        if self.distillation:
            embedded_patches = torch.cat([self.cls_token.expand(embedded_patches.size(0), -1, -1),
                                           self.distillation_token.expand(embedded_patches.size(0), -1, -1),
                                             embedded_patches], dim=1)

        else:
            embedded_patches = torch.cat([self.cls_token.expand(embedded_patches.size(0), -1, -1), embedded_patches], dim=1)


        embedded_patches += self.positional_encoding

        encoder_output = self.encoder.forward(embedded_patches)
        cls_token = encoder_output[:, 0, :]
        distillation_token = encoder_output[:, 1, :] if self.distillation else None
        return cls_token, distillation_token


    def train_step(self, x, y, optimizer, hard_distillation=False, lambda_distillation=0.5, tau=1.0):
        optimizer.zero_grad()
        cls_token, distillation_token = self.forward(x)
        logits = self.cls_classifier(cls_token)
        if self.distillation:
            distillation_logits = self.distillation_classifier(distillation_token)
            if hard_distillation:
                with torch.no_grad():
                    teacher_logits = self.teacher_model(x)
                    if isinstance(teacher_logits, tuple):
                        teacher_logits = teacher_logits[0]

                    distillation_loss = nn.CrossEntropyLoss()(distillation_logits, torch.argmax(teacher_logits, dim=1))
            else:
                with torch.no_grad():
                    teacher_logits = self.teacher_model(x)
                    student = nn.functional.log_softmax(distillation_logits / tau, dim=1)
                    teacher = nn.functional.softmax(teacher_logits / tau, dim=1)
                    distillation_loss = nn.KLDivLoss()(student, teacher)

            if hard_distillation:
                loss = (nn.CrossEntropyLoss()(logits, y) + distillation_loss) / 2
            else:
                loss = (1 - lambda_distillation) * nn.CrossEntropyLoss()(logits, y) + lambda_distillation * (tau ** 2) * distillation_loss
        else:
            loss = nn.CrossEntropyLoss()(logits, y)
        loss.backward()
        optimizer.step()
        return loss.item(), logits


    def predict(self, x):
        with torch.no_grad():
            cls_token, distillation_token = self.forward(x)
            logits = self.cls_classifier(cls_token)
            if self.distillation:
                distillation_logits = self.distillation_classifier(distillation_token)
                pred = (logits + distillation_logits) / 2
            else:
                pred = logits
            cls = torch.argmax(pred, dim=1)
            return cls


