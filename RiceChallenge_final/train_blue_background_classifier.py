"""
Blue Background Rice Classifier
Train a lightweight 3-class classifier:
  - Class 0: Blue background + rice present
  - Class 1: Non-blue background + rice present  
  - Class 2: Other (no rice or irrelevant)

Optimized for mobile deployment (TFLite export).
"""

import os
import numpy as np
import cv2
import shutil
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import torchvision.models as models
from torchvision import transforms, datasets
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import GroupKFold
from sklearn.metrics import precision_score, recall_score, f1_score
from pathlib import Path
from PIL import Image
import random
from joblib import Parallel, delayed
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, MofNCompleteColumn
from rich.table import Table
import warnings

warnings.filterwarnings('ignore')
console = Console()


class Config:
    # Data paths
    SCRIPT_DIR = Path(__file__).parent
    DATA_DIR = SCRIPT_DIR / 'training_data'
    ORIGINAL_IMAGES_DIR = Path('/workspace/app-builder-unido-africarice/RiceChallenge_final/Data/images/images')
    
    # Output
    OUTPUT_DIR = SCRIPT_DIR / 'blue_bg_classifier'
    MODEL_PATH = OUTPUT_DIR / 'blue_bg_classifier.pth'
    TFLITE_PATH = OUTPUT_DIR / 'blue_bg_classifier.tflite'
    
    # Model & training
    IMAGE_SIZE = 224  # Mobile-friendly size
    NUM_CLASSES = 3   # Blue+Rice, Non-Blue+Rice, Other
    BATCH_SIZE = 32
    EPOCHS = 50
    LR = 1e-3
    WEIGHT_DECAY = 1e-4
    
    # Dataset composition
    BLUE_RICE_COUNT = 2500      # Augmented from 1341 base images
    NONBLUE_RICE_COUNT = 1000   # Shuffled channels, also augmented
    OTHER_COUNT = 500           # Downloaded or synthetic

    # Cleanup
    CLEAN_PARTIAL_FOOD101 = True
    
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    SEED = 42


def set_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.enable_grad()


class BlueBackgroundDataset(Dataset):
    """Dataset with samples and their group IDs for GroupKFold."""
    
    def __init__(self, samples, transform=None):
        """
        samples: list of tuples (image_path, label, group_id)
        transform: albumentations compose
        """
        self.samples = samples
        self.transform = transform
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label, _ = self.samples[idx]
        
        # Load image
        image = cv2.imread(str(img_path))
        if image is None:
            # Fallback: create a placeholder
            image = np.random.randint(0, 255, (Config.IMAGE_SIZE, Config.IMAGE_SIZE, 3), dtype=np.uint8)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']
        else:
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        
        return image, label


def load_original_images():
    """Load all 1341 original images (all blue background + rice)."""
    console.print("[bold cyan]Loading original images...[/bold cyan]")
    images = []
    
    if not Config.ORIGINAL_IMAGES_DIR.exists():
        console.print(f"[yellow]Warning: {Config.ORIGINAL_IMAGES_DIR} not found[/yellow]")
        return images
    
    image_files = list(Config.ORIGINAL_IMAGES_DIR.glob('*.png'))
    console.print(f"Found {len(image_files)} images")
    
    return image_files


def augment_blue_rice(image_path, num_augmentations):
    """Generate augmented versions of a blue+rice image."""
    augmented = []
    
    try:
        img = cv2.imread(str(image_path))
        if img is None:
            return augmented
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    except Exception as e:
        console.print(f"[red]Error loading {image_path}: {e}[/red]")
        return augmented
    
    # Define augmentation pipeline (must be inside function for parallel processing)
    aug_pipeline = A.Compose([
        A.OneOf([
            A.HorizontalFlip(p=1),
            A.VerticalFlip(p=1),
        ], p=1.0),
        A.CLAHE(p=0.75),
        A.OneOf([
            A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.2, p=1),
            A.RandomGamma(p=1),
        ], p=1.0),
        # Diverse cropping: 60% probability with varied crop sizes
        # When p=0.6, 40% of images keep full context (no crop applied), 60% get cropping
        A.OneOf([
            A.RandomCrop(height=int(Config.IMAGE_SIZE*0.95), width=int(Config.IMAGE_SIZE*0.95), p=1),  # 95% = slight zoom
            A.CenterCrop(height=int(Config.IMAGE_SIZE*0.90), width=int(Config.IMAGE_SIZE*0.90), p=1),   # 90% = mild zoom
            A.RandomCrop(height=int(Config.IMAGE_SIZE*0.80), width=int(Config.IMAGE_SIZE*0.80), p=1),  # 80% = moderate zoom
            A.CenterCrop(height=int(Config.IMAGE_SIZE*0.70), width=int(Config.IMAGE_SIZE*0.70), p=1),   # 70% = strong zoom
        ], p=0.6),  # 60% get cropped, 40% keep full context
        A.Resize(Config.IMAGE_SIZE, Config.IMAGE_SIZE),
    ])
    
    for _ in range(num_augmentations):
        try:
            aug_img = aug_pipeline(image=img)['image']
            augmented.append(aug_img)
        except Exception as e:
            pass  # Silently skip errors in parallel workers
    
    return augmented


def shuffle_channels_for_nonblue(image_path, num_augmentations):
    """
    Generate non-blue background versions by shuffling channels.
    Shuffle RGB channels to move blue away from background.
    Then apply normal augmentations.
    """
    augmented = []
    
    try:
        img = cv2.imread(str(image_path))
        if img is None:
            return augmented
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    except Exception as e:
        return augmented
    
    # Channel shuffle options
    channel_shuffles = [
        [0, 1, 2],  # Original RGB
        [1, 0, 2],  # GRB
        [2, 1, 0],  # BGR (swap R and B)
        [1, 2, 0],  # GBR
        [2, 0, 1],  # BRG
    ]
    
    # Define augmentation pipeline inside function for parallel processing
    aug_pipeline = A.Compose([
        A.OneOf([
            A.HorizontalFlip(p=1),
            A.VerticalFlip(p=1),
        ], p=0.8),
        A.CLAHE(p=0.6),
        A.OneOf([
            A.RandomBrightnessContrast(brightness_limit=0.4, contrast_limit=0.3, p=1),
            A.RandomGamma(p=1),
        ], p=0.9),
        # Diverse cropping: 60% probability with varied crop sizes
        # When p=0.6, 40% of images keep full context, 60% get cropping
        A.OneOf([
            A.RandomCrop(height=int(Config.IMAGE_SIZE*0.95), width=int(Config.IMAGE_SIZE*0.95), p=1),  # 95% = slight zoom
            A.CenterCrop(height=int(Config.IMAGE_SIZE*0.90), width=int(Config.IMAGE_SIZE*0.90), p=1),   # 90% = mild zoom
            A.RandomCrop(height=int(Config.IMAGE_SIZE*0.80), width=int(Config.IMAGE_SIZE*0.80), p=1),  # 80% = moderate zoom
            A.CenterCrop(height=int(Config.IMAGE_SIZE*0.70), width=int(Config.IMAGE_SIZE*0.70), p=1),   # 70% = strong zoom
        ], p=0.6),  # 60% get cropped, 40% keep full context
        A.Resize(Config.IMAGE_SIZE, Config.IMAGE_SIZE),
    ])
    
    for _ in range(num_augmentations):
        shuffle = random.choice(channel_shuffles)
        shuffled = img[:, :, shuffle]
        
        try:
            aug_img = aug_pipeline(image=shuffled)['image']
            augmented.append(aug_img)
        except Exception as e:
            pass  # Silently skip errors in parallel workers
    
    return augmented


def download_other_class_images(count):
    """
    Download real-world 'other' class images from CIFAR-10 + CIFAR-100.
    Returns list of image arrays (non-rice, diverse objects/people/food).
    """
    other_images = []
    downloaded_dir = Config.DATA_DIR / 'downloaded_other'
    downloaded_dir.mkdir(parents=True, exist_ok=True)
    
    # Try CIFAR-10 first (easiest, ~50k diverse images)
    try:
        console.print("[cyan]Downloading CIFAR-10...[/cyan]")
        cifar_dir = Config.DATA_DIR / 'cifar10'
        cifar_dir.mkdir(parents=True, exist_ok=True)
        
        # Download CIFAR-10
        cifar_train = datasets.CIFAR10(root=str(cifar_dir), train=True, download=True)
        
        # Extract count/2 random images from CIFAR-10
        need = min(count // 2, len(cifar_train))
        indices = random.sample(range(len(cifar_train)), need)
        
        for idx in indices:
            img, _ = cifar_train[idx]
            img = np.array(img)
            # Upscale from 32×32 to IMAGE_SIZE
            img = cv2.resize(img, (Config.IMAGE_SIZE, Config.IMAGE_SIZE), interpolation=cv2.INTER_LINEAR)
            other_images.append(img)
        
        console.print(f"[green]✓ Extracted {len(other_images)} from CIFAR-10[/green]")
    except Exception as e:
        console.print(f"[yellow]CIFAR-10 download failed: {e}[/yellow]")
    
    # Supplement with CIFAR-100 if available, or use synthetic fallback
    if len(other_images) < count:
        remaining = count - len(other_images)
        console.print(f"[cyan]Need {remaining} more images; trying CIFAR-100...[/cyan]")
        
        try:
            cifar100_dir = Config.DATA_DIR / 'cifar100'
            cifar100_dir.mkdir(parents=True, exist_ok=True)
            
            cifar100_train = datasets.CIFAR100(
                root=str(cifar100_dir),
                train=True,
                download=True,
            )
            
            need = min(remaining, len(cifar100_train))
            indices = random.sample(range(len(cifar100_train)), need)
            
            for idx in indices:
                img, _ = cifar100_train[idx]
                img = np.array(img)
                img = cv2.resize(img, (Config.IMAGE_SIZE, Config.IMAGE_SIZE), interpolation=cv2.INTER_LINEAR)
                other_images.append(img)
            
            console.print(f"[green]✓ Extracted {need} from CIFAR-100[/green]")
        except Exception as e:
            console.print(f"[yellow]CIFAR-100 download failed: {e}. Using synthetic fallback.[/yellow]")
            
            # Fallback: generate synthetic if downloads fail
            remaining = count - len(other_images)
            for _ in range(remaining):
                img = np.random.randint(0, 255, (Config.IMAGE_SIZE, Config.IMAGE_SIZE, 3), dtype=np.uint8)
                
                # Add some structure
                if random.random() > 0.5:
                    for _ in range(random.randint(3, 8)):
                        cx = random.randint(0, Config.IMAGE_SIZE)
                        cy = random.randint(0, Config.IMAGE_SIZE)
                        radius = random.randint(20, 60)
                        color = tuple(random.randint(0, 255) for _ in range(3))
                        cv2.circle(img, (cx, cy), radius, color, -1)
                
                other_images.append(img.astype(np.uint8))
            
            console.print(f"[yellow]Generated {remaining} synthetic images as fallback[/yellow]")
    
    console.print(f"[green]✓ Total 'other' images: {len(other_images)}[/green]")
    return other_images[:count]


def prepare_training_data():
    """Prepare complete training dataset with augmentations and grouping."""
    console.print("\n[bold cyan]Preparing training data...[/bold cyan]")
    
    Config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load original images
    original_images = load_original_images()
    if not original_images:
        console.print("[red]No original images found! Using synthetic data only.[/red]")
        original_images = []
    
    samples = []  # (image_array, label, group_id)
    group_ids = []
    
    # Class 0: Blue + Rice (augmented from original 1341 images)
    console.print(f"\n[bold]Class 0: Blue + Rice[/bold] (target: {Config.BLUE_RICE_COUNT})")
    console.print("  Augmenting (parallel)...")
    
    # Prepare augmentation tasks
    aug_tasks = []
    for group_id, img_path in enumerate(original_images):
        aug_count = max(1, Config.BLUE_RICE_COUNT // len(original_images)) + (1 if group_id < Config.BLUE_RICE_COUNT % len(original_images) else 0)
        aug_tasks.append((img_path, aug_count, group_id))
    
    # Parallel augmentation
    def augment_task(img_path, aug_count, group_id):
        augmented_imgs = augment_blue_rice(img_path, aug_count)
        return [(aug_img, 0, group_id) for aug_img in augmented_imgs]
    
    results = Parallel(n_jobs=-1, backend='threading')(delayed(augment_task)(path, count, gid) for path, count, gid in aug_tasks)
    blue_rice_data = [item for sublist in results for item in sublist]
    console.print(f"  Generated: {len(blue_rice_data)} samples")
    
    # Class 1: Non-Blue + Rice (shuffled channels from subset of originals)
    console.print(f"\n[bold]Class 1: Non-Blue + Rice[/bold] (target: {Config.NONBLUE_RICE_COUNT})")
    subset = random.sample(original_images, min(len(original_images), 200))
    aug_per_image = max(1, Config.NONBLUE_RICE_COUNT // len(subset))
    console.print("  Shuffling channels (parallel)...")
    
    # Prepare shuffling tasks
    shuffle_tasks = []
    for group_id, img_path in enumerate(subset):
        shuffle_tasks.append((img_path, aug_per_image, len(original_images) + group_id))
    
    # Parallel shuffling
    def shuffle_task(img_path, aug_count, group_id):
        augmented_imgs = shuffle_channels_for_nonblue(img_path, aug_count)
        return [(aug_img, 1, group_id) for aug_img in augmented_imgs]
    
    results = Parallel(n_jobs=-1, backend='threading')(delayed(shuffle_task)(path, count, gid) for path, count, gid in shuffle_tasks)
    nonblue_rice_data = [item for sublist in results for item in sublist]
    console.print(f"  Generated: {len(nonblue_rice_data)} samples")
    
    # Class 2: Other (synthetic or downloaded)
    console.print(f"\n[bold]Class 2: Other[/bold] (target: {Config.OTHER_COUNT})")
    other_data = []
    other_imgs = download_other_class_images(Config.OTHER_COUNT)
    for idx, img in enumerate(other_imgs):
        other_data.append((img, 2, len(original_images) + len(subset) + idx))
    
    console.print(f"  Generated: {len(other_data)} synthetic samples")
    
    # Combine all and save to disk
    console.print("\n[bold cyan]Saving augmented images to disk...[/bold cyan]")
    all_data = blue_rice_data + nonblue_rice_data + other_data
    
    class_dirs = [
        Config.DATA_DIR / 'class_0_blue_rice',
        Config.DATA_DIR / 'class_1_nonblue_rice',
        Config.DATA_DIR / 'class_2_other',
    ]
    
    for class_dir in class_dirs:
        class_dir.mkdir(parents=True, exist_ok=True)
    
    samples_with_paths = []
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        console=console,
    ) as pbar:
        task = pbar.add_task("Saving...", total=len(all_data))
        
        def save_image(idx, img, label, group_id):
            img_path = class_dirs[label] / f"sample_{idx:05d}.png"
            cv2.imwrite(str(img_path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            pbar.update(task, advance=1)
            return (str(img_path), label, group_id)
        
        samples_with_paths = Parallel(n_jobs=-1, backend='threading')(
            delayed(save_image)(idx, img, label, group_id) 
            for idx, (img, label, group_id) in enumerate(all_data)
        )
    
    console.print(f"[green]✓ Total samples prepared: {len(samples_with_paths)}[/green]")
    
    return samples_with_paths


def cleanup_partial_food101():
    food_dir = Config.DATA_DIR / 'food101'
    if food_dir.exists():
        console.print("[yellow]Removing partial Food-101 download to free space...[/yellow]")
        shutil.rmtree(food_dir, ignore_errors=True)


class MobileNetV3Classifier(nn.Module):
    """Lightweight classifier based on MobileNetV3-Small."""
    
    def __init__(self, num_classes=3):
        super().__init__()
        self.backbone = models.mobilenet_v3_small(pretrained=True)
        
        # Replace classifier head
        in_features = self.backbone.classifier[0].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes),
        )
    
    def forward(self, x):
        return self.backbone(x)


def train_classifier(train_loader, val_loader):
    """Train the classifier with metrics tracking."""
    model = MobileNetV3Classifier(num_classes=Config.NUM_CLASSES).to(Config.DEVICE)
    optimizer = AdamW(model.parameters(), lr=Config.LR, weight_decay=Config.WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=Config.EPOCHS)
    criterion = nn.CrossEntropyLoss()
    
    console.print(f"\n[bold green]Training for {Config.EPOCHS} epochs[/bold green]")
    console.print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")
    
    best_f1 = 0
    
    for epoch in range(1, Config.EPOCHS + 1):
        # Training
        model.train()
        train_loss = 0
        with Progress(
            SpinnerColumn(),
            TextColumn(f"[cyan]Epoch {epoch}/{Config.EPOCHS}[/cyan]"),
            BarColumn(),
            MofNCompleteColumn(),
            console=console,
            transient=True,
        ) as pbar:
            task = pbar.add_task("", total=len(train_loader))
            
            for images, labels in train_loader:
                images, labels = images.to(Config.DEVICE), labels.to(Config.DEVICE)
                
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                pbar.update(task, advance=1)
        
        train_loss /= len(train_loader)
        
        # Validation
        model.eval()
        all_preds = []
        all_labels = []
        val_loss = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(Config.DEVICE), labels.to(Config.DEVICE)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
                preds = outputs.argmax(dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(labels.cpu().numpy())
        
        val_loss /= len(val_loader)
        
        # Compute metrics
        precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
        recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
        f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
        
        # Display results
        is_best = f1 > best_f1
        if is_best:
            best_f1 = f1
            torch.save(model.state_dict(), str(Config.MODEL_PATH))
        
        status = "[green]★[/green]" if is_best else " "
        console.print(
            f"Epoch {epoch:2d} | Train: {train_loss:.4f} | Val: {val_loss:.4f} | "
            f"Prec: {precision:.4f} | Rec: {recall:.4f} | F1: {f1:.4f} {status}"
        )
        
        scheduler.step()
        
        # Detailed metrics every 10 epochs
        if epoch % 10 == 0 or is_best:
            table = Table(title=f"Detailed Metrics - Epoch {epoch}")
            table.add_column("Metric")
            table.add_column("Value", justify="right")
            table.add_row("Train Loss", f"{train_loss:.4f}")
            table.add_row("Val Loss", f"{val_loss:.4f}")
            table.add_row("Precision", f"{precision:.4f}")
            table.add_row("Recall", f"{recall:.4f}")
            table.add_row("F1 Score", f"{f1:.4f}")
            console.print(table)
    
    console.print(f"\n[bold green]✓ Training complete! Best F1: {best_f1:.4f}[/bold green]")
    console.print(f"Checkpoint saved: {Config.MODEL_PATH}")
    
    return model


def export_to_tflite(model_path):
    """Export trained model to TFLite format."""
    console.print("\n[bold cyan]Exporting to TFLite...[/bold cyan]")
    
    try:
        import tensorflow as tf
    except ImportError:
        console.print("[yellow]Warning: TensorFlow not installed. Skipping TFLite export.[/yellow]")
        console.print("Install with: pip install tensorflow")
        return
    
    try:
        model = MobileNetV3Classifier(num_classes=Config.NUM_CLASSES)
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
        model.eval()
        
        # Convert to ONNX first
        dummy_input = torch.randn(1, 3, Config.IMAGE_SIZE, Config.IMAGE_SIZE)
        onnx_path = str(Config.OUTPUT_DIR / 'blue_bg_classifier.onnx')
        
        torch.onnx.export(
            model, dummy_input, onnx_path,
            input_names=['input'],
            output_names=['output'],
            opset_version=12,
        )
        console.print(f"[green]✓ ONNX exported: {onnx_path}[/green]")

        # Convert ONNX -> TFLite using onnx2tf
        try:
            import onnx2tf
            from google.protobuf import message
        except Exception as e:
            console.print(
                f"[yellow]onnx2tf import failed: {e}. Install: pip install onnx2tf[/yellow]"
            )
            return

        saved_model_dir = str(Config.OUTPUT_DIR / 'tf_saved_model')
        
        # Convert ONNX to TensorFlow SavedModel
        onnx2tf.convert(
            input_onnx_file_path=onnx_path,
            output_folder_path=saved_model_dir,
            non_verbose=True
        )

        # Convert SavedModel to TFLite
        converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS,
            tf.lite.OpsSet.SELECT_TF_OPS
        ]
        tflite_model = converter.convert()

        with open(Config.TFLITE_PATH, 'wb') as f:
            f.write(tflite_model)

        console.print(f"[green]✓ TFLite exported: {Config.TFLITE_PATH}[/green]")

    except Exception as e:
        console.print(f"[red]Export error: {e}[/red]")


def main():
    set_seed(Config.SEED)
    Config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if Config.CLEAN_PARTIAL_FOOD101:
        cleanup_partial_food101()
    
    console.print("[bold cyan]Blue Background Rice Classifier Training[/bold cyan]")
    console.print(f"Image size: {Config.IMAGE_SIZE}×{Config.IMAGE_SIZE}")
    console.print(f"Batch size: {Config.BATCH_SIZE}")
    console.print(f"Epochs: {Config.EPOCHS}\n")
    
    # Prepare data
    samples = prepare_training_data()
    
    if not samples:
        console.print("[red]No data prepared! Exiting.[/red]")
        return
    
    # Extract group IDs for GroupKFold
    X = np.arange(len(samples))
    y = np.array([label for _, label, _ in samples])
    groups = np.array([group_id for _, _, group_id in samples])
    
    # Split with GroupKFold
    console.print("\n[bold cyan]Splitting data with GroupKFold...[/bold cyan]")
    gkf = GroupKFold(n_splits=5)
    splits = list(gkf.split(X, y, groups))
    
    # Use first split (80/20)
    train_idx, val_idx = splits[0]
    train_samples = [samples[i] for i in train_idx]
    val_samples = [samples[i] for i in val_idx]
    
    console.print(f"Train samples: {len(train_samples)}")
    console.print(f"Val samples: {len(val_samples)}")
    
    # Transform pipeline
    train_transform = A.Compose([
        A.Resize(Config.IMAGE_SIZE, Config.IMAGE_SIZE),
        A.HorizontalFlip(p=0.3),
        A.RandomBrightnessContrast(p=0.2),
        A.Normalize(),
        ToTensorV2(),
    ])
    
    val_transform = A.Compose([
        A.Resize(Config.IMAGE_SIZE, Config.IMAGE_SIZE),
        A.Normalize(),
        ToTensorV2(),
    ])
    
    # Create datasets and loaders
    train_dataset = BlueBackgroundDataset(train_samples, transform=train_transform)
    val_dataset = BlueBackgroundDataset(val_samples, transform=val_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=4)
    
    # Train
    model = train_classifier(train_loader, val_loader)
    
    # Export
    export_to_tflite(str(Config.MODEL_PATH))
    
    console.print(f"\n[bold green]✓ Complete![/bold green]")
    console.print(f"Model: {Config.MODEL_PATH}")
    console.print(f"Ready to integrate into Flutter app")


if __name__ == "__main__":
    main()
