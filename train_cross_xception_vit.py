
import torch
import torch.nn as nn
import os
import argparse
import yaml
from tqdm import tqdm
import numpy as np
import collections
import math
from torch.optim import lr_scheduler
from torch.cuda.amp import GradScaler, autocast

# 引用定義的模型
from cross_xception_vit import CrossXceptionViT

from deepfakes_dataset import DeepFakesDataset
from utils import get_n_params, check_correct, shuffle_dataset

# --- 全域路徑設定 ---
BASE_DIR = '../prepared_dataset/' 
TRAINING_DIR = os.path.join(BASE_DIR, "training_set")
VALIDATION_DIR = os.path.join(BASE_DIR, "validation_set")
MODELS_PATH = "models"

def read_frames(video_path, num_frames_to_sample):
    """
    從影片資料夾讀取指定數量的影格路徑。
    """
    if "Original" in video_path:
        label = 0.
    else:
        label = 1.
        
    all_frames_paths = sorted([os.path.join(video_path, f) for f in os.listdir(video_path)])
    frames_number = len(all_frames_paths)
    
    if frames_number > num_frames_to_sample:
        # 如果總影格數大於要取樣的數量，則均勻取樣
        step = frames_number // num_frames_to_sample
        sampled_paths = all_frames_paths[::step][:num_frames_to_sample]
    else:
        # 如果不夠，則全部選取
        sampled_paths = all_frames_paths
        
    return [(path, label) for path in sampled_paths]

# --- 主程式區塊 ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="訓練 Cross-Xception-ViT 模型")
    parser.add_argument('--config', type=str, default='config_cross_xception_vit.yaml', help="設定檔路徑")
    parser.add_argument('--dataset', type=str, default='All', help="要使用的偽造資料集類型 (e.g., 'Deepfakes', 'FaceSwap', 'All')")
    parser.add_argument('--workers', default=8, type=int, help="DataLoader 的工作執行緒數量")
    parser.add_argument('--num_epochs', default=100, type=int, help="訓練的總 epoch 數")
    parser.add_argument('--patience', type=int, default=10, help="Early stopping 的耐心值")
    opt = parser.parse_args()
    print(opt)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"--- 使用設備: {device} ---")

    with open(opt.config, 'r') as ymlfile:
        config = yaml.safe_load(ymlfile)

    # --- 資料準備與動態 1:1 平衡邏輯 ---
    print("--- 準備資料集，強制 1:1 真假影像平衡 ---")
    frames_per_real_video = max(int(config['training']['frames-per-video']), 1)
    all_fake_types = [d for d in os.listdir(TRAINING_DIR) if os.path.isdir(os.path.join(TRAINING_DIR, d)) and d != 'Original']
    fake_folders_to_process = all_fake_types if opt.dataset == 'All' else [opt.dataset]
    
    num_real_train_videos = len(os.listdir(os.path.join(TRAINING_DIR, 'Original')))
    total_real_train_images = num_real_train_videos * frames_per_real_video
    num_fake_train_videos = sum(len(os.listdir(os.path.join(TRAINING_DIR, fake_type))) for fake_type in fake_folders_to_process)
    frames_per_fake_train = max(1, round(total_real_train_images / num_fake_train_videos)) if num_fake_train_videos > 0 else 0
    print(f"訓練集: 每個真實影片取 {frames_per_real_video} 幀。計算後，每個偽造影片取約 {frames_per_fake_train} 幀以達成平衡。")

    num_real_val_videos = len(os.listdir(os.path.join(VALIDATION_DIR, 'Original')))
    total_real_val_images = num_real_val_videos * frames_per_real_video
    num_fake_val_videos = sum(len(os.listdir(os.path.join(VALIDATION_DIR, fake_type))) for fake_type in fake_folders_to_process)
    frames_per_fake_val = max(1, round(total_real_val_images / num_fake_val_videos)) if num_fake_val_videos > 0 else 0
    print(f"驗證集: 每個真實影片取 {frames_per_real_video} 幀。計算後，每個偽造影片取約 {frames_per_fake_val} 幀以達成平衡。")
    
    all_data_paths = []
    video_folders_to_process = []
    
    for dataset_base_path in [TRAINING_DIR, VALIDATION_DIR]:
        for folder_type in ['Original'] + fake_folders_to_process:
            subfolder = os.path.join(dataset_base_path, folder_type)
            if os.path.exists(subfolder):
                video_folders_to_process.extend([os.path.join(subfolder, v) for v in os.listdir(subfolder) if os.path.isdir(os.path.join(subfolder, v))])

    for path in tqdm(video_folders_to_process, desc="正在收集所有影像路徑"):
        num_samples = frames_per_real_video if "Original" in path else (frames_per_fake_train if TRAINING_DIR in path else frames_per_fake_val)
        all_data_paths.extend(read_frames(path, num_samples))

    train_data_paths = [d for d in all_data_paths if TRAINING_DIR in d[0]]
    validation_data_paths = [d for d in all_data_paths if VALIDATION_DIR in d[0]]
    train_samples = len(train_data_paths)
    validation_samples = len(validation_data_paths)

    print(f"訓練影像總數: {train_samples}, 驗證影像總數: {validation_samples}")
    print("最終訓練集統計:", collections.Counter(p[1] for p in train_data_paths))
    print("最終驗證集統計:", collections.Counter(p[1] for p in validation_data_paths))

    # --- 模型、優化器、損失函數設定 ---
    model = CrossXceptionViT(config, pretrained=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['training']['lr'], weight_decay=config['training']['weight-decay'])
    scheduler = lr_scheduler.StepLR(optimizer, step_size=config['training']['step-size'], gamma=config['training']['gamma'])
    scaler = GradScaler(enabled=True) # 使用混合精度訓練
    loss_fn = torch.nn.BCEWithLogitsLoss()
    
    print("模型總參數數量:", get_n_params(model))

    # --- DataLoader 設定 ---
    train_dataset = DeepFakesDataset(train_data_paths, labels=None, image_size=config['model']['image-size'], mode='train')
    dl = torch.utils.data.DataLoader(train_dataset, batch_size=config['training']['bs'], shuffle=True, num_workers=opt.workers)
    validation_dataset = DeepFakesDataset(validation_data_paths, labels=None, image_size=config['model']['image-size'], mode='validation')
    val_dl = torch.utils.data.DataLoader(validation_dataset, batch_size=config['training']['bs'], shuffle=False, num_workers=opt.workers)
    
    # --- 訓練迴圈 ---
    model.to(device)
    not_improved_loss = 0
    previous_loss = math.inf
    
    for t in range(opt.num_epochs):
        if not_improved_loss >= opt.patience:
            print(f"Early stopping 在 epoch {t+1}，因為驗證損失已連續 {opt.patience} 個 epochs 未改善。")
            break
        
        # --- 訓練階段 ---
        total_loss, train_correct = 0, 0
        model.train()
        loop = tqdm(dl, desc=f"EPOCH #{t+1}/{opt.num_epochs} [訓練中]", leave=True)
        for index, (images, labels) in enumerate(loop):
            images = images.permute(0, 3, 1, 2).to(device)
            labels = labels.unsqueeze(1).to(device)
            
            with autocast(enabled=True):
                y_pred = model(images)
                loss = loss_fn(y_pred, labels)
            
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            total_loss += loss.item()
            corrects, _, _ = check_correct(y_pred.cpu(), labels.cpu())
            train_correct += corrects
            loop.set_postfix(loss=loss.item(), acc=f"{(train_correct / ((index + 1) * config['training']['bs'])):.4f}")

        # --- 驗證階段 ---
        total_val_loss, val_correct = 0, 0
        model.eval()
        with torch.no_grad():
            for val_images, val_labels in val_dl:
                val_images = val_images.permute(0, 3, 1, 2).to(device)
                val_labels = val_labels.unsqueeze(1).to(device)
                
                with autocast(enabled=True):
                    val_pred = model(val_images)
                    val_loss = loss_fn(val_pred, val_labels)
                
                total_val_loss += val_loss.item()
                corrects, _, _ = check_correct(val_pred.cpu(), val_labels.cpu())
                val_correct += corrects
        
        scheduler.step()
        
        avg_loss = total_loss / len(dl)
        avg_acc = train_correct / train_samples
        avg_val_loss = total_val_loss / len(val_dl)
        avg_val_acc = val_correct / validation_samples
        
        print(f"EPOCH #{t+1}/{opt.num_epochs} -> loss: {avg_loss:.4f}, acc: {avg_acc:.4f} || val_loss: {avg_val_loss:.4f}, val_acc: {avg_val_acc:.4f}")
        
        # --- 儲存最佳模型 ---
        if avg_val_loss < previous_loss:
            previous_loss = avg_val_loss
            not_improved_loss = 0
            if not os.path.exists(MODELS_PATH):
                os.makedirs(MODELS_PATH)
            model_save_path = os.path.join(MODELS_PATH, f"cross_xception_vit_best_{opt.dataset}.pth")
            torch.save(model.state_dict(), model_save_path)
            print(f"驗證損失改善，儲存最佳模型至 {model_save_path}")
        else:
            not_improved_loss += 1
            print(f"驗證損失未改善，計數: {not_improved_loss}/{opt.patience}")