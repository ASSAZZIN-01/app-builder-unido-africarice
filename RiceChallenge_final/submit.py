
import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm

# Constants from training script
class Config:
    # Use Data folder relative to where the script is run
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(SCRIPT_DIR, 'Data')
    IMAGE_DIR = os.path.join(DATA_DIR, 'images', 'images')
    TEST_CSV = os.path.join(DATA_DIR, 'Test.csv')
    CHECKPOINT = 'ultimate_tiled_multitask.pth'
    
    MODEL_NAME = 'convnext_small.fb_in22k_ft_in1k_384'
    TILE_SIZE = 512
    GRID_COLS = 8
    GRID_ROWS = 6
    N_TILES = GRID_COLS * GRID_ROWS
    
    COUNT_COLS = ['Count', 'Broken_Count', 'Long_Count', 'Medium_Count', 'Black_Count',
                  'Chalky_Count', 'Red_Count', 'Yellow_Count', 'Green_Count']
    MEASURE_COLS = ['WK_Length_Average', 'WK_Width_Average', 'WK_LW_Ratio_Average',
                    'Average_L', 'Average_a', 'Average_b']
    
    PADDY_ZERO = ['Chalky_Count', 'Medium_Count', 'Yellow_Count', 'Green_Count']
    BROWN_ZERO = ['Green_Count']
    
    SCALE = 100.0
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    SEED = 42

def set_seed(seed):
    """Set random seeds and deterministic flags for reproducible inference."""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class MultiScaleCSRDecoder(nn.Module):
    def __init__(self, in_channels_list):
        super().__init__()
        # Combine features from 1/16 and 1/32 resolutions
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
        """Decode multi-scale features into a density map per tile."""
        f32_up = self.up(self.reduce_32(feats_32))
        f16 = self.reduce_16(feats_16)

        if f32_up.shape != f16.shape:
            f32_up = F.interpolate(f32_up, size=f16.shape[2:], mode='bilinear', align_corners=True)

        combined = torch.cat([f16, f32_up, meta_map_16], dim=1)
        return self.backend(combined)

class UltimateSpecialist(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        self.backbone = timm.create_model(model_name, pretrained=False, features_only=True)
        ch_list = self.backbone.feature_info.channels()
        self.meta_proj = nn.Sequential(nn.Linear(3, 32), nn.LayerNorm(32), nn.GELU())
        self.count_heads = nn.ModuleList([MultiScaleCSRDecoder(ch_list) for _ in range(9)])
        self.measure_head = nn.Sequential(
            nn.Linear(ch_list[-1] + 32, 256), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(256, 6)
        )

    def forward(self, x, meta):
        """Run tiled inference and return count and measure predictions."""
        B, N, C, H_in, W_in = x.shape
        x_flat = x.view(B*N, C, H_in, W_in)
        all_feats = self.backbone(x_flat)
        f16 = all_feats[-2]
        f32 = all_feats[-1]
        m = self.meta_proj(meta)
        m_flat = m.repeat_interleave(N, dim=0)
        bh16, bw16 = f16.shape[2:]
        m_map16 = m_flat.view(B*N, 32, 1, 1).expand(-1, -1, bh16, bw16)
        
        tile_densities = []
        for head in self.count_heads:
            d = F.relu(head(f16, f32, m_map16))
            tile_sum = d.sum(dim=(1,2,3)).view(B, N)
            tile_sum = tile_sum.sum(dim=1).unsqueeze(1) # Sum N tiles -> (B, 1)
            tile_densities.append(tile_sum)
        
        counts = torch.cat(tile_densities, dim=1)
        m_map32 = m_flat.view(B*N, 32, 1, 1).expand(-1, -1, f32.shape[2], f32.shape[3])
        combined32 = torch.cat([f32, m_map32], dim=1)
        pool = F.adaptive_avg_pool2d(combined32, 1).view(B, N, -1).mean(dim=1)
        measures = self.measure_head(pool)
        return counts, measures

def get_tiles(image):
    """Split an image into a fixed 8x6 grid of tiles."""
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

def main():
    """Load checkpoint, run inference on test data, and write submission CSV."""
    set_seed(Config.SEED)
    checkpoint = torch.load(Config.CHECKPOINT, map_location='cpu', weights_only=False)
    model = UltimateSpecialist(Config.MODEL_NAME)
    model.load_state_dict(checkpoint['model'])
    model.to(Config.DEVICE)
    model.eval()
    
    m_stats = checkpoint['m_stats']
    test_df = pd.read_csv(Config.TEST_CSV)
    
    transform = A.Compose([A.Resize(Config.TILE_SIZE, Config.TILE_SIZE), A.Normalize(), ToTensorV2()])

    results = []
    
    with torch.no_grad():
        for _, row in tqdm(test_df.iterrows(), total=len(test_df)):
            img_path = os.path.join(Config.IMAGE_DIR, f"{row['ID']}.png")
            image = np.array(Image.open(img_path).convert('RGB'))
            tiles = get_tiles(image)
            
            # Rice type meta
            rice_type = {'Paddy': 0, 'White': 1, 'Brown': 2}.get(row['Comment'], 0)
            meta = torch.zeros(1, 3).to(Config.DEVICE)
            meta[0, rice_type] = 1.0
            
            processed = torch.stack([transform(image=t)['image'] for t in tiles]).unsqueeze(0).to(Config.DEVICE)
            p_c, p_m = model(processed, meta)
            
            # Undo training-time scaling/normalization for readable outputs
            p_c = p_c.cpu().numpy()[0] / Config.SCALE
            p_m = p_m.cpu().numpy()[0] * (m_stats[1] + 1e-8) + m_stats[0]
            
            # Post-processing: apply known class constraints for rice types
            for k, col in enumerate(Config.COUNT_COLS):
                if (rice_type == 0 and col in Config.PADDY_ZERO) or (rice_type == 2 and col in Config.BROWN_ZERO):
                    p_c[k] = 0
            
            # Round counts, keep measures as is
            res_row = {'ID': row['ID']}
            for i, col in enumerate(Config.COUNT_COLS):
                res_row[col] = max(0, int(round(p_c[i])))
            for i, col in enumerate(Config.MEASURE_COLS):
                res_row[col] = p_m[i]
                
            results.append(res_row)

    out_df = pd.DataFrame(results)
    cols = ['ID'] + Config.COUNT_COLS + Config.MEASURE_COLS
    out_df = out_df[cols]
    out_df.to_csv('submission.csv', index=False)
    print("Submission saved to submission.csv")

if __name__ == "__main__":
    main()
