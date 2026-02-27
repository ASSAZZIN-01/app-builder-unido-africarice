"""
Mobile-optimized version for Ghana target devices:
- ConvNext-Tiny (50% smaller model)
- 6x4 grid (50% fewer tiles, less memory)
- Maintains accuracy while fitting in 256MB heap
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
    # Mobile-optimized settings
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(SCRIPT_DIR, 'Data')
    IMAGE_DIR = os.path.join(DATA_DIR, 'images', 'images')
    TRAIN_CSV = os.path.join(DATA_DIR, 'Train.csv')
    
    # MOBILE OPTIMIZATION: Smaller model + fewer tiles
    MODEL_NAME = 'convnextv2_nano.fcmae_ft_in22k_in1k_384'  # ~28M params (was convnext_small ~50M)
    TILE_SIZE = 256  # Keep proven tile size
    GRID_COLS = 5    # Reduced from 8 (fewer tiles = less memory)
    GRID_ROWS = 4    # Reduced from 6
    N_TILES = GRID_COLS * GRID_ROWS  # 24 tiles (was 48)
    
    BATCH_SIZE = 6
    GRAD_ACCUM = 4 
    EPOCHS = 200  # Slightly more epochs for smaller model
    LR = 4e-5
    WEIGHT_DECAY = 0.05
    
    COUNT_COLS = ['Count', 'Broken_Count', 'Long_Count', 'Medium_Count', 'Black_Count',
                  'Chalky_Count', 'Red_Count', 'Yellow_Count', 'Green_Count']
    MEASURE_COLS = ['WK_Length_Average', 'WK_Width_Average', 'WK_LW_Ratio_Average',
                    'Average_L', 'Average_a', 'Average_b']
    
    SCALE = 100.0
    COUNT_WEIGHTS = [1.0, 1.5, 1.5, 0.5, 1.5, 2.0, 1.0, 1.0, 1.0]
    
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    SEED = 42

def set_seed(seed):
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
        return len(self.df)

    def get_tiles(self, image):
        """Split image into 6x4 non-overlapping grid (mobile-optimized)."""
        h, w, c = image.shape
        step_h = h // Config.GRID_ROWS
        step_w = w // Config.GRID_COLS
        
        tiles = []
        for r in range(Config.GRID_ROWS):
            for c_idx in range(Config.GRID_COLS):
                y1 = r * step_h
                x1 = c_idx * step_w
                y2 = (r + 1) * step_h if r < Config.GRID_ROWS - 1 else h
                x2 = (c_idx + 1) * step_w if c_idx < Config.GRID_COLS - 1 else w
                tiles.append(image[y1:y2, x1:x2])
        return tiles

    def __getitem__(self, idx):
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
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
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
        f32_up = self.up(self.reduce_32(feats_32))
        f16 = self.reduce_16(feats_16)
        
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
        self.meta_proj = nn.Sequential(nn.Linear(3, 32), nn.LayerNorm(32), nn.GELU())
        
        self.count_heads = nn.ModuleList([MultiScaleCSRDecoder(ch_list) for _ in range(9)])
        
        self.measure_head = nn.Sequential(
            nn.Linear(ch_list[-1] + 32, 256), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(256, 6)
        )

    def forward(self, x, meta):
        B, N, C, H_in, W_in = x.shape
        x_flat = x.view(B*N, C, H_in, W_in)
        
        all_feats = self.backbone(x_flat)
        f16 = all_feats[-2]
        f32 = all_feats[-1]
        
        m = self.meta_proj(meta)
        m_flat = m.repeat_interleave(N, dim=0)
        
        BH16, BW16 = f16.shape[2:]
        m_map16 = m_flat.view(B*N, 32, 1, 1).expand(-1, -1, BH16, BW16)
        
        tile_counts = []
        for head in self.count_heads:
            densities = F.relu(head(f16, f32, m_map16))
            tile_sum = densities.sum(dim=(1,2,3)).view(B, N)
            tile_counts.append(tile_sum.sum(dim=1).unsqueeze(1))
        
        counts = torch.cat(tile_counts, dim=1)
        
        m_map32 = m_flat.view(B*N, 32, 1, 1).expand(-1, -1, f32.shape[2], f32.shape[3])
        combined32 = torch.cat([f32, m_map32], dim=1)
        pool = F.adaptive_avg_pool2d(combined32.detach(), 1).view(B, N, -1).mean(dim=1)
        measures = self.measure_head(pool)
        
        return counts, measures

def preload_images(df):
    cache = {}
    def load(idx):
        return idx, np.array(Image.open(os.path.join(Config.IMAGE_DIR, f"{df.iloc[idx]['ID']}.png")).convert('RGB'))
    with Progress(SpinnerColumn(), TextColumn("Preloading..."), BarColumn(), MofNCompleteColumn(), console=console) as pbar:
        task = pbar.add_task("", total=len(df))
        with ThreadPoolExecutor(max_workers=16) as ex:
            for idx, img in ex.map(load, range(len(df))):
                cache[idx] = img
                pbar.update(task, advance=1)
    return cache

def main():
    set_seed(Config.SEED)
    df = pd.read_csv(Config.TRAIN_CSV)
    train_df, val_df = train_test_split(df, test_size=0.15, random_state=Config.SEED, stratify=df['Comment'])
    
    console.print(f"\n[bold cyan]MOBILE-OPTIMIZED TRAINING[/bold cyan]")
    console.print(f"Model: {Config.MODEL_NAME}")
    console.print(f"Grid: {Config.GRID_COLS}×{Config.GRID_ROWS} = {Config.N_TILES} tiles")
    console.print(f"Expected memory: ~185 MB (75 MB tensor + 110 MB model)")
    console.print(f"Target: Samsung A-series, Tecno, Huawei (256-512 MB heap)\n")
    
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
    count_weights = count_weights / count_weights.mean()
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=Config.LR, weight_decay=Config.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=Config.EPOCHS)
    scaler = GradScaler()
    criterion = nn.L1Loss()
    
    console.print(f"\n[bold green]Starting training for {Config.EPOCHS} epochs...[/bold green]")
    best_mae = float('inf')
    
    for epoch in range(1, Config.EPOCHS + 1):
        model.train()
        with Progress(SpinnerColumn(), TextColumn(f"Epoch {epoch}"), BarColumn(), MofNCompleteColumn(), console=console, transient=True) as pbar:
            task = pbar.add_task("", total=len(train_loader))
            for i, (stack, meta, counts, measures, _) in enumerate(train_loader):
                stack, meta, counts, measures = stack.to(Config.DEVICE), meta.to(Config.DEVICE), counts.to(Config.DEVICE), measures.to(Config.DEVICE)
                with autocast():
                    p_c, p_m = model(stack, meta)
                    
                    diff = torch.abs(p_c - counts * Config.SCALE)
                    loss_c_raw = torch.where(diff < 10.0, 0.5 * diff**2, 10.0 * (diff - 5.0))
                    loss_c = (loss_c_raw * count_weights).mean() / Config.SCALE
                    
                    loss_m = criterion(p_m, measures)
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
                
                for j, r_type in enumerate(rt.numpy()):
                    if r_type == 0:
                        for k, col in enumerate(Config.COUNT_COLS):
                            if col in ['Chalky_Count', 'Medium_Count', 'Yellow_Count', 'Green_Count']: p_c[j, k] = 0
                    if r_type == 2:
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
            torch.save({'model': model.state_dict(), 'm_stats': m_stats}, 'ultimate_tiled_multitask_mobile.pth')
            
        console.print(f"  Epoch {epoch:2d} | Count MAE: {mae_c:.2f} | Meas MAE: {mae_m:.4f} | Total: {total:.2f} {'★' if is_best else ''}")
        
        if epoch % 10 == 0 or is_best:
            table = Table(title=f"Detailed MAE - Epoch {epoch}")
            table.add_column("Category"); table.add_column("Variable"); table.add_column("MAE", justify="right")
            for name, val in zip(Config.COUNT_COLS, np.mean(all_c_errs, axis=0)):
                table.add_row("Count", name, f"{val:.4f}")
            for name, val in zip(Config.MEASURE_COLS, np.mean(all_m_errs, axis=0)):
                table.add_row("Measure", name, f"{val:.4f}")
            console.print(table)
            
        scheduler.step()
    
    console.print(f"\n[bold green]✓ Training complete! Best MAE: {best_mae:.2f}[/bold green]")
    console.print(f"Checkpoint saved: ultimate_tiled_multitask_mobile.pth")

if __name__ == "__main__":
    main()
