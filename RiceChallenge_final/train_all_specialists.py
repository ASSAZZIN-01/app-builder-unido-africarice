"""
- Strategy: Precise 8x6 Grid (Non-overlapping for Counts) for NATIVE 4K resolution.
- Architecture: FPN-style Multi-Scale Backbone (ConvNeXt-Small).
- Fixes: Unified Gradient Flow (no detach), Sum-Consistency, and Overlap Correction.
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import GradScaler, autocast
import timm
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import train_test_split
from concurrent.futures import ThreadPoolExecutor
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, MofNCompleteColumn
from rich.table import Table
import random
import warnings

warnings.filterwarnings('ignore')
console = Console()

class Config:
    # Use Data folder relative to where the script is run
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(SCRIPT_DIR, 'Data')
    IMAGE_DIR = os.path.join(DATA_DIR, 'images', 'images')
    TRAIN_CSV = os.path.join(DATA_DIR, 'Train.csv')
    
    MODEL_NAME = 'convnext_small.fb_in22k_ft_in1k_384'
    TILE_SIZE = 512
    GRID_COLS = 8
    GRID_ROWS = 6
    N_TILES = GRID_COLS * GRID_ROWS
    
    BATCH_SIZE = 2
    GRAD_ACCUM = 4 
    EPOCHS = 200
    LR = 4e-5
    WEIGHT_DECAY = 0.05
    
    COUNT_COLS = ['Count', 'Broken_Count', 'Long_Count', 'Medium_Count', 'Black_Count',
                  'Chalky_Count', 'Red_Count', 'Yellow_Count', 'Green_Count']
    MEASURE_COLS = ['WK_Length_Average', 'WK_Width_Average', 'WK_LW_Ratio_Average',
                    'Average_L', 'Average_a', 'Average_b']
    
    # Sub-categories that MUST sum up to Total Count
    # Primary groups: Broken + Long + Medium = Total Count
    # Secondary groups (Overlapping properties): Black, Chalky, Red, Yellow, Green
    
    # Scale counts for stable optimization (counts can be large)
    SCALE = 100.0
    # Specific weights for each count category (Higher = More focus)
    # Order: ['Count', 'Broken_Count', 'Long_Count', 'Medium_Count', 'Black_Count', 'Chalky_Count', 'Red_Count', 'Yellow_Count', 'Green_Count']
    COUNT_WEIGHTS = [1.0, 1.5, 1.5, 0.5, 1.5, 2.0, 1.0, 1.0, 1.0]
    
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    SEED = 42

def set_seed(seed):
    """Set random seeds and deterministic flags for reproducible training."""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class TiledMultiTaskDataset(Dataset):
    def __init__(self, df, transform, measure_stats, cache=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.measure_stats = measure_stats
        self.cache = cache if cache else {}

    def __len__(self):
        """Return number of samples in the dataset."""
        return len(self.df)

    def get_tiles(self, image):
        """Split an image into a fixed 8x6 non-overlapping grid."""
        h, w, c = image.shape
        # Exact non-overlapping grid to prevent double counting
        # For 4000x3000 image, 8x6 grid gives 500x500 tiles
        step_h = h // Config.GRID_ROWS
        step_w = w // Config.GRID_COLS
        
        tiles = []
        for r in range(Config.GRID_ROWS):
            for c_idx in range(Config.GRID_COLS):
                y1 = r * step_h
                x1 = c_idx * step_w
                y2 = (r + 1) * step_h if r < Config.GRID_ROWS - 1 else h
                x2 = (c_idx + 1) * step_w if c_idx < Config.GRID_COLS - 1 else w
                
                tile = image[y1:y2, x1:x2]
                # Ensure each tile is exactly the same size for batching (resize if needed)
                tiles.append(tile)
        return tiles

    def __getitem__(self, idx):
        """Load, tile, and transform a sample with its targets and metadata."""
        row = self.df.iloc[idx]
        image = self.cache.get(idx)
        if image is None:
            image = np.array(Image.open(os.path.join(Config.IMAGE_DIR, f"{row['ID']}.png")).convert('RGB'))
            
        tiles = self.get_tiles(image)
        processed = [self.transform(image=t)['image'] for t in tiles]
        stack = torch.stack(processed)
        
        counts = torch.tensor(row[Config.COUNT_COLS].values.astype(np.float32), dtype=torch.float32)
        measures = torch.tensor((row[Config.MEASURE_COLS].values.astype(np.float32) - self.measure_stats[0]) / (self.measure_stats[1] + 1e-8), dtype=torch.float32)
        
        rice_type = {'Paddy': 0, 'White': 1, 'Brown': 2}.get(row['Comment'], 0)
        meta = torch.zeros(3)
        meta[rice_type] = 1.0
        
        return stack, meta, counts, measures, rice_type

class MultiScaleCSRDecoder(nn.Module):
    def __init__(self, in_channels_list):
        super().__init__()
        # Combine features from 1/16 and 1/32 resolutions
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
        # Merge layer for the two scales
        mid_ch = 128
        self.reduce_32 = nn.Sequential(nn.Conv2d(in_channels_list[-1], mid_ch, 1), nn.ReLU(inplace=True))
        self.reduce_16 = nn.Sequential(nn.Conv2d(in_channels_list[-2], mid_ch, 1), nn.ReLU(inplace=True))
        
        self.backend = nn.Sequential(
            nn.Conv2d(mid_ch * 2 + 32, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, kernel_size=1)
        )

    def forward(self, feats_16, feats_32, meta_map_16):
        """Decode multi-scale features into a density map for counts."""
        f32_up = self.up(self.reduce_32(feats_32))
        f16 = self.reduce_16(feats_16)
        
        # Ensure sizes match exactly (handling odd dimensions)
        if f32_up.shape != f16.shape:
            f32_up = F.interpolate(f32_up, size=f16.shape[2:], mode='bilinear', align_corners=True)
            
        combined = torch.cat([f16, f32_up, meta_map_16], dim=1)
        return self.backend(combined)

class UltimateSpecialist(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        self.backbone = timm.create_model(model_name, pretrained=True, features_only=True)
        self.backbone.set_grad_checkpointing(True)
        
        ch_list = self.backbone.feature_info.channels()
        # Using 1/16 (index -2) and 1/32 (index -1) features
        self.meta_proj = nn.Sequential(nn.Linear(3, 32), nn.LayerNorm(32), nn.GELU())
        
        self.count_heads = nn.ModuleList([MultiScaleCSRDecoder(ch_list) for _ in range(9)])
        
        # Measures head uses global context from 1/32 features
        self.measure_head = nn.Sequential(
            nn.Linear(ch_list[-1] + 32, 256), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(256, 6)
        )

    def forward(self, x, meta):
        """Run tiled forward pass and return count and measure predictions."""
        B, N, C, H_in, W_in = x.shape
        x_flat = x.view(B*N, C, H_in, W_in)
        
        all_feats = self.backbone(x_flat)
        f16 = all_feats[-2]
        f32 = all_feats[-1]
        
        m = self.meta_proj(meta) # (B, 32)
        m_flat = m.repeat_interleave(N, dim=0)
        
        # Meta map for the 1/16 resolution
        BH16, BW16 = f16.shape[2:]
        m_map16 = m_flat.view(B*N, 32, 1, 1).expand(-1, -1, BH16, BW16)
        
        # Predict Counts
        tile_counts = []
        for head in self.count_heads:
            densities = F.relu(head(f16, f32, m_map16))
            tile_sum = densities.sum(dim=(1,2,3)).view(B, N)
            tile_counts.append(tile_sum.sum(dim=1).unsqueeze(1))
        
        counts = torch.cat(tile_counts, dim=1) # (B, 9)
        
        # Predict Measures using GAP on f32
        m_map32 = m_flat.view(B*N, 32, 1, 1).expand(-1, -1, f32.shape[2], f32.shape[3])
        combined32 = torch.cat([f32, m_map32], dim=1)
        pool = F.adaptive_avg_pool2d(combined32.detach(), 1).view(B, N, -1).mean(dim=1)
        measures = self.measure_head(pool)
        
        return counts, measures

def preload_images(df):
    """Preload images into memory to speed up training epochs."""
    cache = {}
    def load(idx):
        """Load a single image by index for preloading."""
        return idx, np.array(Image.open(os.path.join(Config.IMAGE_DIR, f"{df.iloc[idx]['ID']}.png")).convert('RGB'))
    with Progress(SpinnerColumn(), TextColumn("Preloading..."), BarColumn(), MofNCompleteColumn(), console=console) as pbar:
        task = pbar.add_task("", total=len(df))
        with ThreadPoolExecutor(max_workers=16) as ex:
            for idx, img in ex.map(load, range(len(df))):
                cache[idx] = img
                pbar.update(task, advance=1)
    return cache

def main():
    """Train the model, track validation MAE, and save the best checkpoint."""
    set_seed(Config.SEED)
    df = pd.read_csv(Config.TRAIN_CSV)
    train_df, val_df = train_test_split(df, test_size=0.15, random_state=Config.SEED, stratify=df['Comment'])
    
    t_cache = preload_images(train_df)
    v_cache = preload_images(val_df)
    
    m_raw = train_df[Config.MEASURE_COLS].values.astype(np.float32)
    m_stats = (m_raw.mean(axis=0), m_raw.std(axis=0))
    
    tf = A.Compose([A.Resize(Config.TILE_SIZE, Config.TILE_SIZE), A.HorizontalFlip(), A.VerticalFlip(), A.RandomRotate90(), A.Normalize(), ToTensorV2()])
    v_tf = A.Compose([A.Resize(Config.TILE_SIZE, Config.TILE_SIZE), A.Normalize(), ToTensorV2()])
    
    train_loader = DataLoader(TiledMultiTaskDataset(train_df, tf, m_stats, t_cache), batch_size=Config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(TiledMultiTaskDataset(val_df, v_tf, m_stats, v_cache), batch_size=Config.BATCH_SIZE)
    
    model = UltimateSpecialist(Config.MODEL_NAME).to(Config.DEVICE)
    count_weights = torch.tensor(Config.COUNT_WEIGHTS, dtype=torch.float32).to(Config.DEVICE)
    count_weights = count_weights / count_weights.mean() # Normalize to average 1.0
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=Config.LR, weight_decay=Config.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=Config.EPOCHS)
    scaler = GradScaler()
    criterion = nn.L1Loss()
    huber = nn.HuberLoss(delta=10.0)
    
    console.print(f"\n[bold green]Starting MULTI-SCALE NATIVE 4K (8x6 Grid)...[/bold green]")
    best_mae = float('inf')
    
    for epoch in range(1, Config.EPOCHS + 1):
        model.train()
        with Progress(SpinnerColumn(), TextColumn(f"Epoch {epoch}"), BarColumn(), MofNCompleteColumn(), console=console, transient=True) as pbar:
            task = pbar.add_task("", total=len(train_loader))
            for i, (stack, meta, counts, measures, _) in enumerate(train_loader):
                stack, meta, counts, measures = stack.to(Config.DEVICE), meta.to(Config.DEVICE), counts.to(Config.DEVICE), measures.to(Config.DEVICE)
                with autocast():
                    p_c, p_m = model(stack, meta)
                    
                    # 1. Main Loss (Weighted Huber for stability and focus on difficult classes)
                    diff = torch.abs(p_c - counts * Config.SCALE)
                    # Huber-like penalty but with weights
                    loss_c_raw = torch.where(diff < 10.0, 0.5 * diff**2, 10.0 * (diff - 5.0))
                    loss_c = (loss_c_raw * count_weights).mean() / Config.SCALE
                    
                    loss_m = criterion(p_m, measures)
                    
                    # 2. Consistency Loss: Total Count must match sum of sub-counts
                    # Indices: Total(0), Broken(1), Long(2), Medium(3)
                    loss_consist = criterion(p_c[:,1:4].sum(dim=1), p_c[:,0]) / Config.SCALE
                    
                    loss = 1.5 * loss_c + 0.1 * loss_m + 0.5 * loss_consist
                    
                scaler.scale(loss).backward()
                if (i+1) % Config.GRAD_ACCUM == 0:
                    scaler.step(optimizer); scaler.update(); optimizer.zero_grad()
                pbar.update(task, advance=1)
        
        model.eval()
        c_errs, m_errs = [], []
        with torch.no_grad():
            for stack, meta, counts, measures, rt in val_loader:
                p_c, p_m = model(stack.to(Config.DEVICE), meta.to(Config.DEVICE))
                p_c = p_c.cpu().numpy() / Config.SCALE
                
                # Paddy/Brown Zeros: enforce known class constraints on count categories
                for j, r_type in enumerate(rt.numpy()):
                    if r_type == 0: # Paddy
                        for k, col in enumerate(Config.COUNT_COLS):
                            if col in ['Chalky_Count', 'Medium_Count', 'Yellow_Count', 'Green_Count']: p_c[j, k] = 0
                    if r_type == 2: # Brown
                        for k, col in enumerate(Config.COUNT_COLS):
                            if col == 'Green_Count': p_c[j, k] = 0
                
                c_errs.append(np.abs(p_c - counts.numpy()))
                m_errs.append(np.abs(p_m.cpu().numpy() * (m_stats[1] + 1e-8) + m_stats[0] - (measures.numpy() * (m_stats[1] + 1e-8) + m_stats[0])))
        
        all_c_errs, all_m_errs = np.concatenate(c_errs), np.concatenate(m_errs)
        mae_c, mae_m = np.mean(all_c_errs), np.mean(all_m_errs)
        total = (mae_c * 9 + mae_m * 6) / 15
        
        is_best = total < best_mae
        if is_best:
            best_mae = total
            torch.save({'model': model.state_dict(), 'm_stats': m_stats}, 'ultimate_tiled_multitask.pth')
            
        console.print(f"  Epoch {epoch:2d} | Count MAE: {mae_c:.2f} | Meas MAE: {mae_m:.4f} | Total: {total:.2f} {'★' if is_best else ''}")
        
        if epoch % 5 == 0 or is_best:
            table = Table(title=f"Detailed MAE - Epoch {epoch}")
            table.add_column("Category"); table.add_column("Variable"); table.add_column("MAE", justify="right")
            for name, val in zip(Config.COUNT_COLS, np.mean(all_c_errs, axis=0)):
                table.add_row("Count", name, f"{val:.4f}")
            for name, val in zip(Config.MEASURE_COLS, np.mean(all_m_errs, axis=0)):
                table.add_row("Measure", name, f"{val:.4f}")
            console.print(table)
            
        scheduler.step()

if __name__ == "__main__":
    main()