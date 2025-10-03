import torch
from torch.utils.data import Dataset
import cv2
import numpy as np
from albumentations import Compose, RandomBrightnessContrast, HorizontalFlip, FancyPCA, HueSaturationValue, ImageCompression, OneOf, ToFloat, ShiftScaleRotate, PadIfNeeded, GaussNoise, GaussianBlur
from transforms.albu import IsotropicResize

# --- 定義圖片的轉換/增強流程 ---
def create_transforms(image_size):
    return Compose([
        IsotropicResize(max_side=image_size, interpolation_down=cv2.INTER_AREA, interpolation_up=cv2.INTER_CUBIC),
        PadIfNeeded(min_height=image_size, min_width=image_size, border_mode=cv2.BORDER_CONSTANT),
        OneOf([
            RandomBrightnessContrast(),
            FancyPCA(),
            HueSaturationValue()
        ], p=0.7),
        HorizontalFlip(p=0.5),
        ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=10, border_mode=cv2.BORDER_CONSTANT, p=0.8),
        ImageCompression(quality_lower=60, quality_upper=100, p=0.5),
        GaussNoise(p=0.1),
        GaussianBlur(blur_limit=3, p=0.05),
        ToFloat(max_value=255)
    ])

def create_validation_transforms(image_size):
    return Compose([
        IsotropicResize(max_side=image_size, interpolation_down=cv2.INTER_AREA, interpolation_up=cv2.INTER_CUBIC),
        PadIfNeeded(min_height=image_size, min_width=image_size, border_mode=cv2.BORDER_CONSTANT),
        ToFloat(max_value=255)
    ])


class DeepFakesDataset(Dataset):
    def __init__(self, data, labels, image_size, mode='train'):
        """
        優化後的初始化方法。
        現在 self.data 儲存的是 (圖片路徑, 標籤) 的元組列表，而不是圖片本身。
        """
        self.data = data # 這是一個包含 (image_path, label) 的列表
        self.image_size = image_size

        # 根據是訓練還是驗證，選擇不同的圖片轉換流程
        if mode == 'train':
            self.transform = create_transforms(image_size)
        else:
            self.transform = create_validation_transforms(image_size)
        # print(f"--- DeepFakesDataset ({mode}) initialized with {len(self.data)} file paths. ---")

    def __len__(self):
        """ 返回資料集的總長度 """
        return len(self.data)

    def __getitem__(self, idx):
        """
        核心的即時載入邏輯。
        只有當 DataLoader 需要第 'idx' 個項目時，這個函式才會被呼叫。
        """
        # 1. 從列表中取得圖片的路徑和標籤
        image_path, label = self.data[idx]

        # 2. 直到此刻，才從硬碟讀取圖片
        image = cv2.imread(image_path)
        if image is None:
            print(f"Warning: Could not read image at {image_path}. Returning a black image.")
            image = np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)

        # 3. 執行圖片轉換
        image = self.transform(image=image)['image']

        # 4. 將標籤轉換為 PyTorch 需要的格式
        label = torch.tensor(label, dtype=torch.float32)

        return image, label