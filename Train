# ==================== CELL 7: OPTIMIZED TRAINING - OSNet-x0.75 WITH ARCFACE ====================

print("=" * 70)
print("CELL 7: OPTIMIZED TRAINING - OSNet-x0.75 WITH ARCFACE")
print("=" * 70)

import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from tqdm import tqdm
import time
import os
import json
from datetime import datetime

# ==================== OPTIMIZED CONFIGURATION ====================

EPOCHS = 30
LEARNING_RATE = 0.00035
WEIGHT_DECAY = 5e-4

# OPTIMIZATION 1: Larger batch size (fewer iterations)
BATCH_SIZE = 128  # Increased from 128 (try 192 or 160 if GPU OOM)

# OPTIMIZATION 2: More workers for faster data loading
NUM_WORKERS = 4  # Reduced for stability (was 8)

# OPTIMIZATION 3: Gradient accumulation (if needed for memory)
ACCUMULATION_STEPS = 1  # Set to 2 if using batch_size=128

# ✅ SAVE TO OUTPUT DIRECTORY FOR DIRECT DOWNLOAD
CHECKPOINT_DIR = '/kaggle/working'  # Direct output directory
SAVE_EVERY_N_EPOCHS = 1  # Save EVERY epoch for download

print(f"\n📋 Optimized Training Configuration:")
print(f"   Model:              {MODEL_NAME}")
print(f"   Epochs:             {EPOCHS}")
print(f"   Batch Size:         {BATCH_SIZE}")
print(f"   Num Workers:        {NUM_WORKERS}")
print(f"   Gradient Accum:     {ACCUMULATION_STEPS} steps")
print(f"   Learning Rate:      {LEARNING_RATE}")
print(f"   Weight Decay:       {WEIGHT_DECAY}")
print(f"   ArcFace Scale:      {ARCFACE_SCALE}")
print(f"   ArcFace Margin:     {ARCFACE_MARGIN}")
print(f"   Save Every:         {SAVE_EVERY_N_EPOCHS} epoch(s)")
print(f"   💾 Save Location:   {CHECKPOINT_DIR}/ (Direct download)")
print(f"   Device:             {device}")


# ==================== RECREATE OPTIMIZED DATALOADERS ====================

print(f"\n🔄 Creating optimized DataLoaders...")

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=True,
    drop_last=True,
    persistent_workers=False,  # Disabled for stability
    prefetch_factor=2  # Reduced for stability
)

val_query_loader = DataLoader(
    val_query_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=True,
    persistent_workers=False,
    prefetch_factor=2
)

val_gallery_loader = DataLoader(
    val_gallery_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=True,
    persistent_workers=False,
    prefetch_factor=2
)

print(f"   ✓ Train loader:        {len(train_loader):,} batches ({BATCH_SIZE} per batch)")
print(f"   ✓ Val Query loader:    {len(val_query_loader):,} batches")
print(f"   ✓ Val Gallery loader:  {len(val_gallery_loader):,} batches")


# ==================== OPTIMIZER & SCHEDULER ====================

print(f"\n⚙️  Setting up training components...")

# Optimizer
optimizer = optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY
)

# Learning rate scheduler
scheduler = optim.lr_scheduler.MultiStepLR(
    optimizer,
    milestones=[int(EPOCHS * 0.4), int(EPOCHS * 0.7)],  # Decay at epochs 12, 21
    gamma=0.1
)

# Mixed precision scaler
scaler = GradScaler()

# Loss function
criterion = nn.CrossEntropyLoss()

print(f"   ✓ Optimizer:  Adam (lr={LEARNING_RATE}, wd={WEIGHT_DECAY})")
print(f"   ✓ Scheduler:  MultiStepLR (decay at epochs {int(EPOCHS*0.4)}, {int(EPOCHS*0.7)})")
print(f"   ✓ Loss:       CrossEntropy + ArcFace")
print(f"   ✓ Mixed Precision: Enabled (AMP)")


# ==================== OPTIMIZED TRAINING FUNCTION ====================

def train_one_epoch(model, train_loader, criterion, optimizer, scaler, device, epoch):
    """Optimized training function with gradient accumulation"""
    model.train()
    
    total_loss = 0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc=f"   Epoch {epoch}/{EPOCHS}", ncols=120)
    
    for batch_idx, (imgs, labels, _) in enumerate(pbar):
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        
        # Mixed precision forward pass
        with autocast():
            features, bn_features, arcface_logits = model(imgs, labels)
            loss = criterion(arcface_logits, labels)
            
            # Normalize loss for gradient accumulation
            if ACCUMULATION_STEPS > 1:
                loss = loss / ACCUMULATION_STEPS
        
        # Backward pass
        scaler.scale(loss).backward()
        
        # Update weights every N accumulation steps
        if (batch_idx + 1) % ACCUMULATION_STEPS == 0 or (batch_idx + 1) == len(train_loader):
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
        
        # Statistics
        actual_loss = loss.item() * ACCUMULATION_STEPS if ACCUMULATION_STEPS > 1 else loss.item()
        total_loss += actual_loss
        
        _, predicted = arcface_logits.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        # Update progress bar
        pbar.set_postfix({
            'Loss': f'{actual_loss:.4f}',
            'Acc': f'{100.*correct/total:.2f}%',
            'LR': f'{optimizer.param_groups[0]["lr"]:.6f}'
        })
    
    avg_loss = total_loss / len(train_loader)
    accuracy = 100. * correct / total
    
    return avg_loss, accuracy


# ==================== VALIDATION FUNCTION ====================

def validate_epoch(model, query_loader, gallery_loader, device):
    """Fast validation with progress indicator"""
    model.eval()
    
    # Extract embeddings
    all_query_emb = []
    all_query_labels = []
    
    with torch.no_grad():
        for imgs, labels, _ in tqdm(query_loader, desc="   Val Query", leave=False, ncols=100):
            imgs = imgs.to(device, non_blocking=True)
            embeddings = model.extract_features(imgs)
            all_query_emb.append(embeddings.cpu())
            all_query_labels.extend(labels.numpy())
    
    query_emb = torch.cat(all_query_emb, dim=0).numpy()
    query_labels = np.array(all_query_labels)
    
    all_gallery_emb = []
    all_gallery_labels = []
    
    with torch.no_grad():
        for imgs, labels, _ in tqdm(gallery_loader, desc="   Val Gallery", leave=False, ncols=100):
            imgs = imgs.to(device, non_blocking=True)
            embeddings = model.extract_features(imgs)
            all_gallery_emb.append(embeddings.cpu())
            all_gallery_labels.extend(labels.numpy())
    
    gallery_emb = torch.cat(all_gallery_emb, dim=0).numpy()
    gallery_labels = np.array(all_gallery_labels)
    
    # Evaluate
    metrics = evaluate_reid(query_emb, query_labels, gallery_emb, gallery_labels)
    
    return metrics


# ==================== ✅ MODIFIED CHECKPOINT FUNCTION - SAVE TO /kaggle/working/ ====================

def save_checkpoint(model, optimizer, epoch, train_loss, train_acc, val_metrics, is_best=False):
    """Save model checkpoint to /kaggle/working/ for direct download"""
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_loss,
        'train_acc': train_acc,
        'val_mAP': val_metrics['mAP'],
        'val_rank1': val_metrics['Rank-1'],
        'val_rank5': val_metrics['Rank-5'],
        'val_rank10': val_metrics['Rank-10'],
        'model_name': MODEL_NAME,
        'num_classes': num_classes,
        'arcface_scale': ARCFACE_SCALE,
        'arcface_margin': ARCFACE_MARGIN,
    }
    
    # ✅ Save epoch checkpoint to /kaggle/working/
    epoch_filepath = f"{CHECKPOINT_DIR}/{MODEL_NAME}_epoch{epoch:02d}.pth"
    torch.save(checkpoint, epoch_filepath)
    file_size = os.path.getsize(epoch_filepath) / (1024 * 1024)  # MB
    print(f"      💾 Saved: {os.path.basename(epoch_filepath)} ({file_size:.1f} MB)")
    
    # ✅ Save best model separately
    if is_best:
        best_filepath = f"{CHECKPOINT_DIR}/{MODEL_NAME}_BEST.pth"
        torch.save(checkpoint, best_filepath)
        print(f"      🏆 BEST: {os.path.basename(best_filepath)} ({file_size:.1f} MB)")
    
    # ✅ Always save latest checkpoint (overwrite)
    latest_filepath = f"{CHECKPOINT_DIR}/{MODEL_NAME}_LATEST.pth"
    torch.save(checkpoint, latest_filepath)


# ==================== MAIN TRAINING LOOP ====================

print(f"\n{'='*70}")
print("🚀 STARTING OPTIMIZED TRAINING")
print(f"{'='*70}\n")

# Training history
history = {
    'train_loss': [],
    'train_acc': [],
    'val_mAP': [],
    'val_rank1': [],
    'val_rank5': [],
    'val_rank10': [],
    'learning_rates': [],
    'epochs': [],
    'epoch_times': []
}

best_mAP = 0
best_rank1 = 0
best_epoch = 0
start_time = time.time()

# Training loop
for epoch in range(1, EPOCHS + 1):
    epoch_start = time.time()
    
    print(f"\n{'='*70}")
    print(f"📊 Epoch {epoch}/{EPOCHS}")
    print(f"{'='*70}")
    
    # Train
    train_loss, train_acc = train_one_epoch(
        model, train_loader, criterion, optimizer, scaler, device, epoch
    )
    
    epoch_time = time.time() - epoch_start
    
    print(f"\n   📈 Train Results:")
    print(f"      Loss:       {train_loss:.4f}")
    print(f"      Accuracy:   {train_acc:.2f}%")
    print(f"      Time:       {epoch_time/60:.2f} min")
    
    # Validate
    print(f"\n   🧪 Validation:")
    val_metrics = validate_epoch(model, val_query_loader, val_gallery_loader, device)
    
    print(f"\n   📊 Validation Results:")
    print(f"      mAP:        {val_metrics['mAP']*100:.2f}%")
    print(f"      Rank-1:     {val_metrics['Rank-1']*100:.2f}%")
    print(f"      Rank-5:     {val_metrics['Rank-5']*100:.2f}%")
    print(f"      Rank-10:    {val_metrics['Rank-10']*100:.2f}%")
    
    # Save history
    history['train_loss'].append(train_loss)
    history['train_acc'].append(train_acc)
    history['val_mAP'].append(float(val_metrics['mAP']))
    history['val_rank1'].append(float(val_metrics['Rank-1']))
    history['val_rank5'].append(float(val_metrics['Rank-5']))
    history['val_rank10'].append(float(val_metrics['Rank-10']))
    history['learning_rates'].append(optimizer.param_groups[0]['lr'])
    history['epochs'].append(epoch)
    history['epoch_times'].append(epoch_time)
    
    # Check if best model
    is_best = val_metrics['mAP'] > best_mAP
    if is_best:
        best_mAP = val_metrics['mAP']
        best_rank1 = val_metrics['Rank-1']
        best_epoch = epoch
        print(f"\n   🏆 NEW BEST MODEL!")
        print(f"      Best mAP:   {best_mAP*100:.2f}%")
        print(f"      Best Rank-1: {best_rank1*100:.2f}%")
    
    # ✅ Save checkpoint EVERY epoch to /kaggle/working/
    print(f"\n   💾 Saving checkpoint to {CHECKPOINT_DIR}/...")
    save_checkpoint(model, optimizer, epoch, train_loss, train_acc, val_metrics, is_best)
    
    # Update learning rate
    old_lr = optimizer.param_groups[0]['lr']
    scheduler.step()
    new_lr = optimizer.param_groups[0]['lr']
    
    if old_lr != new_lr:
        print(f"\n   📉 Learning rate reduced: {old_lr:.6f} → {new_lr:.6f}")
    
    # Time estimates
    elapsed = time.time() - start_time
    avg_epoch_time = elapsed / epoch
    eta = avg_epoch_time * (EPOCHS - epoch)
    
    print(f"\n   ⏱️  Time:")
    print(f"      Epoch:      {epoch_time/60:.2f} min")
    print(f"      Elapsed:    {elapsed/60:.1f} min ({elapsed/3600:.2f}h)")
    print(f"      Remaining:  {eta/60:.1f} min ({eta/3600:.2f}h)")
    print(f"      Avg/epoch:  {avg_epoch_time/60:.2f} min")


# ==================== TRAINING COMPLETE ====================

total_time = time.time() - start_time

print(f"\n{'='*70}")
print("✅ TRAINING COMPLETE!")
print(f"{'='*70}")

print(f"\n📊 Final Summary:")
print(f"   Total time:         {total_time/60:.1f} min ({total_time/3600:.2f} hours)")
print(f"   Avg time/epoch:     {total_time/EPOCHS/60:.2f} min")
print(f"\n   Best Performance:")
print(f"      Epoch:           {best_epoch}")
print(f"      Val mAP:         {best_mAP*100:.2f}%")
print(f"      Val Rank-1:      {best_rank1*100:.2f}%")
print(f"\n   Final Performance:")
print(f"      Train Accuracy:  {history['train_acc'][-1]:.2f}%")
print(f"      Val mAP:         {history['val_mAP'][-1]*100:.2f}%")
print(f"      Val Rank-1:      {history['val_rank1'][-1]*100:.2f}%")

# Calculate improvement
initial_mAP = 0.1153  # From pretrained test
improvement = (best_mAP - initial_mAP) * 100

print(f"\n   📈 Improvement:")
print(f"      Pretrained mAP:  {initial_mAP*100:.2f}%")
print(f"      Trained mAP:     {best_mAP*100:.2f}%")
print(f"      Gain:            +{improvement:.2f}%")

# Save training history
history_path = f"{CHECKPOINT_DIR}/{MODEL_NAME}_training_history.json"
with open(history_path, 'w') as f:
    json.dump(history, f, indent=2)

print(f"\n💾 Saved Files in {CHECKPOINT_DIR}/ (Ready for Download):")
print(f"   📁 {MODEL_NAME}_BEST.pth                  ← Best model (Epoch {best_epoch}, mAP: {best_mAP*100:.2f}%)")
print(f"   📁 {MODEL_NAME}_LATEST.pth                ← Latest model (Epoch {EPOCHS})")
print(f"   📁 {MODEL_NAME}_epoch01.pth ... epoch{EPOCHS:02d}.pth  ← All epoch checkpoints")
print(f"   📁 {MODEL_NAME}_training_history.json     ← Training curves")

print(f"\n📥 HOW TO DOWNLOAD:")
print(f"   1. Click on folder icon (📁) on left sidebar")
print(f"   2. Navigate to /kaggle/working/")
print(f"   3. Click ⋮ (three dots) next to any .pth file")
print(f"   4. Select 'Download'")

print(f"\n🎯 Recommended Download:")
print(f"   → {MODEL_NAME}_BEST.pth (Best performance)")

print(f"\n{'='*70}")
print(f"🚀 Model ready for deployment!")
print(f"{'='*70}\n")
