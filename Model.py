# ==================== CELL 6: MODEL ARCHITECTURE - OSNet WITH ARCFACE ====================

print("=" * 70)
print("CELL 6: DEFINING OSNet-x0.75 WITH ARCFACE HEAD")
print("=" * 70)

import torch
import torch.nn as nn
import math
import torchreid

# ==================== ARCFACE LOSS ====================

class ArcFaceLoss(nn.Module):
    """
    ArcFace Loss: Additive Angular Margin Loss
    Paper: https://arxiv.org/abs/1801.07698
    
    Improves feature discrimination by adding angular margin
    """
    def __init__(self, in_features, out_features, scale=30.0, margin=0.30, easy_margin=False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.scale = scale  # s in paper
        self.margin = margin  # m in paper
        self.easy_margin = easy_margin
        
        # Weight matrix for class centers
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)
        
        # Precompute cos(m) and sin(m) for efficiency
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.th = math.cos(math.pi - margin)
        self.mm = math.sin(math.pi - margin) * margin
    
    def forward(self, embeddings, labels):
        """
        Args:
            embeddings: [N, in_features] - L2 normalized embeddings
            labels: [N] - ground truth labels
        Returns:
            output: [N, out_features] - scaled logits with margin
        """
        # Normalize embeddings and weights
        embeddings = nn.functional.normalize(embeddings, p=2, dim=1)
        weight_norm = nn.functional.normalize(self.weight, p=2, dim=1)
        
        # Compute cosine similarity: cos(theta)
        cosine = nn.functional.linear(embeddings, weight_norm)
        cosine = cosine.clamp(-1.0, 1.0)  # Numerical stability
        
        # Compute sin(theta)
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        
        # Compute cos(theta + m) = cos(theta)cos(m) - sin(theta)sin(m)
        phi = cosine * self.cos_m - sine * self.sin_m
        
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        
        # Create one-hot encoded labels
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1).long(), 1)
        
        # Apply margin only to ground truth class
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        
        # Scale for convergence
        output *= self.scale
        
        return output


# ==================== OSNet + ARCFACE MODEL ====================

class OSNetWithArcFace(nn.Module):
    """OSNet backbone with ArcFace head for Person ReID"""
    
    def __init__(self, backbone_name, num_classes, arcface_scale=30.0, arcface_margin=0.30):
        super().__init__()
        
        print(f"\n🔨 Building {backbone_name} with ArcFace...")
        
        # Load OSNet backbone
        self.backbone = torchreid.models.build_model(
            name=backbone_name,
            num_classes=num_classes,
            pretrained=True,
            loss='softmax'
        )
        
        # Remove original classifier
        self.backbone.classifier = nn.Identity()
        
        # Feature dimension (512 for all OSNet variants)
        self.feat_dim = 512
        
        # BNNeck (Batch Normalization Neck) - standard in ReID
        self.bottleneck = nn.BatchNorm1d(self.feat_dim)
        self.bottleneck.bias.requires_grad_(False)
        
        # ArcFace head
        self.arcface = ArcFaceLoss(
            in_features=self.feat_dim,
            out_features=num_classes,
            scale=arcface_scale,
            margin=arcface_margin
        )
        
        # Initialize BNNeck
        nn.init.constant_(self.bottleneck.weight, 1)
        nn.init.constant_(self.bottleneck.bias, 0)
        
        print(f"   ✓ Backbone: {backbone_name}")
        print(f"   ✓ Feature dim: {self.feat_dim}")
        print(f"   ✓ ArcFace: scale={arcface_scale}, margin={arcface_margin}")
    
    def forward(self, x, labels=None):
        """
        Forward pass
        
        Args:
            x: [N, 3, H, W] input images
            labels: [N] ground truth labels (required during training)
        
        Returns:
            Training mode: (features, bn_features, arcface_logits)
            Eval mode: normalized features
        """
        # Extract features from backbone
        features = self.backbone(x)
        
        # Apply BNNeck
        bn_features = self.bottleneck(features)
        
        if self.training and labels is not None:
            # Training: return ArcFace logits
            arcface_logits = self.arcface(bn_features, labels)
            return features, bn_features, arcface_logits
        else:
            # Evaluation: return normalized features
            return nn.functional.normalize(bn_features, p=2, dim=1)
    
    def extract_features(self, x):
        """Extract L2-normalized features for ReID evaluation"""
        with torch.no_grad():
            features = self.backbone(x)
            bn_features = self.bottleneck(features)
            return nn.functional.normalize(bn_features, p=2, dim=1)


# ==================== CREATE MODEL ====================

# Configuration
MODEL_NAME = 'osnet_x0_75'
ARCFACE_SCALE = 30.0
ARCFACE_MARGIN = 0.30

print(f"\n{'='*70}")
print(f"Creating {MODEL_NAME.upper()} with ArcFace")
print(f"{'='*70}")

# Create model
model = OSNetWithArcFace(
    backbone_name=MODEL_NAME,
    num_classes=num_classes,
    arcface_scale=ARCFACE_SCALE,
    arcface_margin=ARCFACE_MARGIN
).to(device)

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"\n📊 Model Statistics:")
print(f"   Total parameters:     {total_params:,} ({total_params/1e6:.2f}M)")
print(f"   Trainable parameters: {trainable_params:,} ({trainable_params/1e6:.2f}M)")
print(f"   Model size (approx):  ~{total_params * 4 / 1024 / 1024:.1f} MB")

# Test forward pass
print(f"\n🧪 Testing forward pass...")
dummy_input = torch.randn(4, 3, IMG_HEIGHT, IMG_WIDTH).to(device)
dummy_labels = torch.randint(0, num_classes, (4,)).to(device)

model.train()
with torch.no_grad():
    features, bn_features, arcface_logits = model(dummy_input, dummy_labels)

print(f"   ✓ Input:          {dummy_input.shape}")
print(f"   ✓ Features:       {features.shape}")
print(f"   ✓ BN Features:    {bn_features.shape}")
print(f"   ✓ ArcFace Logits: {arcface_logits.shape}")

model.eval()
with torch.no_grad():
    eval_features = model(dummy_input)
print(f"   ✓ Eval Features:  {eval_features.shape} (L2-normalized)")

print(f"\n✅ Model ready for training!")
print("=" * 70)
