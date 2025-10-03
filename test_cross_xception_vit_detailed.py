
import torch
import os
import argparse
import yaml
from tqdm import tqdm
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

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
    
    if frames_number == 0:
        return []

    if frames_number > num_frames_to_sample:
        step = frames_number // num_frames_to_sample
        sampled_paths = all_frames_paths[::step][:num_frames_to_sample]
    else:
        sampled_paths = all_frames_paths
        
    return [(path, label) for path in sampled_paths]


def run_test(model, device, config, video_paths, labels, frames_per_video):
    """
    一個輔助函式，用來執行一輪完整的測試並回傳結果。
    """
    final_predictions = []
    with torch.no_grad():
        for video_path in tqdm(video_paths, desc="正在測試影片"):
            frames_data = read_frames(video_path, frames_per_video)
            if not frames_data: continue

            frame_paths, _ = zip(*frames_data)
            
            # ******** 核心修正：移除 'data_paths=' 關鍵字 ********
            # 第一個參數直接傳遞資料列表，作為位置參數
            video_dataset = DeepFakesDataset(
                list(zip(frame_paths, [0]*len(frame_paths))), 
                labels=None, 
                image_size=config['model']['image-size'], 
                mode='validation'
            )
            
            dl = torch.utils.data.DataLoader(video_dataset, batch_size=config['training']['bs'], shuffle=False, num_workers=4)

            video_predictions = []
            for images, _ in dl:
                images = images.permute(0, 3, 1, 2).to(device)
                preds = model(images)
                preds = torch.sigmoid(preds).squeeze().cpu().numpy()
                if preds.ndim == 0:
                    preds = [preds.item()]
                video_predictions.extend(list(preds))
            
            if not video_predictions: continue
            final_video_pred = np.mean(video_predictions)
            final_predictions.append(round(final_video_pred))
    
    if len(labels) != len(final_predictions):
        print(f"\n警告：標籤數量 ({len(labels)}) 與預測數量 ({len(final_predictions)}) 不符，此輪測試結果可能不準確。")
        return 0, 0, 0, 0

    accuracy = accuracy_score(labels, final_predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, final_predictions, average='binary', zero_division=0)
    
    return accuracy, precision, recall, f1

# --- 主程式區塊 ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="詳細分類測試 Cross-Xception-ViT 模型")
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

    real_videos_path = os.path.join(opt.test_dir, 'Original')
    real_video_files = [os.path.join(real_videos_path, v) for v in os.listdir(real_videos_path) if os.path.isdir(os.path.join(real_videos_path, v))]
    
    fake_method_folders = [f for f in os.listdir(opt.test_dir) if os.path.isdir(os.path.join(opt.test_dir, f)) and f != 'Original']
    
    all_fake_video_files = []
    
    for fake_method in fake_method_folders:
        print(f"\n{'='*20} 開始測試: Original vs. {fake_method} {'='*20}")
        
        current_fake_videos_path = os.path.join(opt.test_dir, fake_method)
        current_fake_video_files = [os.path.join(current_fake_videos_path, v) for v in os.listdir(current_fake_videos_path) if os.path.isdir(os.path.join(current_fake_videos_path, v))]
        all_fake_video_files.extend(current_fake_video_files)
        
        current_test_videos = real_video_files + current_fake_video_files
        current_test_labels = [0] * len(real_video_files) + [1] * len(current_fake_video_files)
        
        acc, pre, rec, f1 = run_test(model, device, config, current_test_videos, current_test_labels, opt.frames_per_video)
        
        print(f"--- 測試結果 (Original vs. {fake_method}) ---")
        print(f"準確率 (Accuracy): {acc:.4f}")
        print(f"精確率 (Precision): {pre:.4f}")
        print(f"召回率 (Recall): {rec:.4f}")
        print(f"F1 分數 (F1-Score): {f1:.4f}")
        print("-" * (48 + len(fake_method)))

    print(f"\n{'='*20} 開始測試: Original vs. ALL Fakes {'='*20}")
    
    overall_test_videos = real_video_files + all_fake_video_files
    overall_test_labels = [0] * len(real_video_files) + [1] * len(all_fake_video_files)

    acc, pre, rec, f1 = run_test(model, device, config, overall_test_videos, overall_test_labels, opt.frames_per_video)

    print(f"--- 總體測試結果 (Original vs. ALL Fakes) ---")
    print(f"準確率 (Accuracy): {acc:.4f}")
    print(f"精確率 (Precision): {pre:.4f}")
    print(f"召回率 (Recall): {rec:.4f}")
    print(f"F1 分數 (F1-Score): {f1:.4f}")
    print("-" * 54)