import os
from torch.utils.data import Dataset
from PIL import Image

class BurstDeflickerDataset(Dataset):
    def __init__(self, root_path, mode='train', transform=None):
        self.transform = transform
        self.data_pairs = []
        
        input_root = os.path.join(root_path, mode + '-resize', 'input')
        gt_root = os.path.join(root_path, mode + '-resize', 'gt')
        
        scenes = sorted(os.listdir(input_root))
        for scene in scenes:
            scene_input_path = os.path.join(input_root, scene)
            scene_gt_path = os.path.join(gt_root, scene)
            
            if not os.path.exists(scene_gt_path):
                continue

            input_files = sorted([f for f in os.listdir(scene_input_path) 
                                 if f.lower().endswith(('.jpg', '.png', '.jpeg'))])
            
            # --- 평가 속도 조절 로직 ---
            if mode == 'train':
                # 학습 시에는 씬당 3장 추출
                num_to_select = min(3, len(input_files))
                selected_indices = [i for i in range(0, len(input_files), len(input_files)//num_to_select)][:num_to_select]
            else:
                # 테스트(평가) 시에는 씬당 딱 1장만 추출 (평가 시간 대폭 단축)
                selected_indices = [0] 

            for idx in selected_indices:
                f = input_files[idx]
                self.data_pairs.append({
                    'input': os.path.join(scene_input_path, f),
                    'gt': scene_gt_path  # 경로만 전달
                })

    def __len__(self):
        return len(self.data_pairs)

    def __getitem__(self, idx):
        pair = self.data_pairs[idx]
        input_img = Image.open(pair['input']).convert('RGB')
        
        # GT 폴더에서 첫 번째 이미지 매칭
        gt_dir = pair['gt']
        gt_files = sorted([f for f in os.listdir(gt_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))])
        actual_gt_path = os.path.join(gt_dir, gt_files[0])
        gt_img = Image.open(actual_gt_path).convert('RGB')
        
        if self.transform:
            input_img = self.transform(input_img)
            gt_img = self.transform(gt_img)
            
        return input_img, gt_img