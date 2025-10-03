
import torch
import os
import argparse
import yaml
from tqdm import tqdm
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from collections import defaultdict
import random # <--- 核心改動：導入 random 模組

# 引用我們自訂的模組
from cross_xception_vit import CrossXceptionViT
from deepfakes_dataset import DeepFakesDataset

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
    
    if frames_number == 0: return []

    if frames_number > num_frames_to_sample:
        step = frames_number // num_frames_to_sample
        sampled_paths = all_frames_paths[::step][:num_frames_to_sample]
    else:
        sampled_paths = all_frames_paths
        
    return [(path, label) for path in sampled_paths]

def print_metrics(title, labels, preds):
    """輔助函式，用於格式化輸出評估指標。"""
    accuracy = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary', zero_division=0)
    
    print(f"\n--- {title} ---")
    print(f"準確率 (Accuracy): {accuracy:.4f}")
    print(f"精確率 (Precision): {precision:.4f}")
    print(f"召回率 (Recall): {recall:.4f}")
    print(f"F1 分數 (F1-Score): {f1:.4f}")
    print("-" * (len(title) + 6))

# --- 主程式區塊 ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="詳細分類測試 Cross-Xception-ViT 模型 (平衡抽樣版)")
    # ... (其餘參數不變)
    parser.add_argument('--weights_path', type=str, default='models/cross_xception_vit_best_All.pth', help="訓練好的模型權重路徑")
    parser.add_argument('--config', type=str, default='config_cross_xception_vit.yaml', help="與模型匹配的設定檔路徑")
    parser.add_argument('--test_dir', type=str, default='../prepared_dataset/test_set', help="測試資料集的路徑")
    parser.add_argument('--frames_per_video', type=int, default=30, help="每個測試影片要取樣的影格數")
    opt = parser.parse_args()
    print(opt)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"--- 使用設備: {device} ---")

    with open(opt.config, 'r') as ymlfile:
        config = yaml.safe_load(ymlfile)

    print(f"--- 正在從 {opt.weights_path} 載入模型權重 ---")
    model = CrossXceptionViT(config, pretrained=False)
    model.load_state_dict(torch.load(opt.weights_path, map_location=device))
    model.to(device)
    model.eval()
    print("--- 模型載入成功 ---")

    # === 1. 收集所有影片路徑，並分類存放 ===
    video_to_method = {}
    all_video_paths = []
    fake_videos_by_method = defaultdict(list)

    real_videos_path = os.path.join(opt.test_dir, 'Original')
    for video_name in os.listdir(real_videos_path):
        path = os.path.join(real_videos_path, video_name)
        if os.path.isdir(path):
            video_to_method[path] = 'Original'
            all_video_paths.append(path)

    fake_method_folders = [f for f in os.listdir(opt.test_dir) if os.path.isdir(os.path.join(opt.test_dir, f)) and f != 'Original']
    for fake_method in fake_method_folders:
        method_path = os.path.join(opt.test_dir, fake_method)
        for video_name in os.listdir(method_path):
            path = os.path.join(method_path, video_name)
            if os.path.isdir(path):
                video_to_method[path] = fake_method
                all_video_paths.append(path)
                fake_videos_by_method[fake_method].append(path)

    # === 2. 一次性完成所有影片的預測 ===
    video_to_prediction = {}
    print(f"\n--- 開始對全部 {len(all_video_paths)} 個影片進行預測 ---")
    with torch.no_grad():
        for video_path in tqdm(all_video_paths, desc="總進度"):
            frames_data = read_frames(video_path, opt.frames_per_video)
            if not frames_data: continue

            frame_paths, _ = zip(*frames_data)
            
            video_dataset = DeepFakesDataset(list(zip(frame_paths, [0]*len(frame_paths))), labels=None, image_size=config['model']['image-size'], mode='validation')
            dl = torch.utils.data.DataLoader(video_dataset, batch_size=config['training']['bs'], shuffle=False, num_workers=4)

            video_predictions = []
            for images, _ in dl:
                images = images.permute(0, 3, 1, 2).to(device)
                preds = model(images)
                preds = torch.sigmoid(preds).squeeze().cpu().numpy()
                if preds.ndim == 0: preds = [preds.item()]
                video_predictions.extend(list(preds))
            
            if not video_predictions: continue
            video_to_prediction[video_path] = round(np.mean(video_predictions))

    # === 3. 從已儲存的預測結果中，分類計算各項指標 ===
    print("\n\n" + "="*25 + " 測 試 報 告 " + "="*25)

    original_paths = [path for path, method in video_to_method.items() if method == 'Original']
    
    # 計算單一偽造方法的性能 (這部分不變)
    for fake_method in fake_method_folders:
        fake_paths = fake_videos_by_method[fake_method]
        test_labels = [0] * len(original_paths) + [1] * len(fake_paths)
        test_preds = [video_to_prediction.get(p, 0) for p in original_paths] + \
                     [video_to_prediction.get(p, 0) for p in fake_paths]
        print_metrics(f"Original vs. {fake_method}", test_labels, test_preds)

    # === 4. 計算總體性能 (使用平衡抽樣) ===
    num_real_videos = len(original_paths)
    num_fake_methods = len(fake_method_folders)
    
    if num_fake_methods > 0:
        num_to_sample_per_method = num_real_videos // num_fake_methods
        sampled_fake_videos = []
        print("\n--- 總體性能平衡抽樣說明 ---")
        print(f"真實影片數: {num_real_videos}")
        print(f"將從 {num_fake_methods} 種偽造方法中，每種隨機抽取 {num_to_sample_per_method} 個影片進行 1:1 評估。")
        print("------------------------------")
        
        for method, videos in fake_videos_by_method.items():
            # 使用 random.sample 進行不重複抽樣
            sampled_videos = random.sample(videos, min(len(videos), num_to_sample_per_method))
            sampled_fake_videos.extend(sampled_videos)

        overall_labels = [0] * len(original_paths) + [1] * len(sampled_fake_videos)
        overall_preds = [video_to_prediction.get(p, 0) for p in original_paths] + \
                        [video_to_prediction.get(p, 0) for p in sampled_fake_videos]

        print_metrics("總體性能 (Original vs. ALL Fakes - 平衡)", overall_labels, overall_preds)