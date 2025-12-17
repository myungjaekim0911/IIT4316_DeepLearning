import os
import torch
import random
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms

def visualize_burst_scene(root_dir, scene_num, mode='train'):
    """
    scene_num: '0001', '0025' 등 4자리 문자열 입력
    """
    folder_path = f"{mode}-resize"
    input_dir = os.path.join(root_dir, folder_path, 'input', scene_num)
    gt_dir = os.path.join(root_dir, folder_path, 'gt', scene_num)

    # 1. 경로 존재 여부 확인
    if not os.path.exists(input_dir) or not os.path.exists(gt_dir):
        print(f"Error: 씬 번호 {scene_num}를 {folder_path}에서 찾을 수 없습니다.")
        return

    # 2. 파일 리스트 확보 및 사전순 정렬
    input_files = sorted([f for f in os.listdir(input_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))])
    gt_files = sorted([f for f in os.listdir(gt_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))])

    # 3. Train 로직: 10장 중 랜덤 3장 샘플링
    selected_indices = sorted(random.sample(range(len(input_files)), 3))
    
    # 4. 시각화 준비 (입력 3장 + GT 1장 = 총 4개 영역)
    plt.figure(figsize=(20, 5))
    plt.suptitle(f"Scene: {scene_num} ({mode} set)", fontsize=16)

    # 입력 이미지 3장 표시
    for i, idx in enumerate(selected_indices):
        img_path = os.path.join(input_dir, input_files[idx])
        img = Image.open(img_path).convert('RGB')
        
        plt.subplot(1, 4, i + 1)
        plt.imshow(img)
        plt.title(f"Input Index: {idx}\n({input_files[idx]})")
        plt.axis('off')

    # GT 이미지 표시 (첫 번째 GT 고정)
    gt_path = os.path.join(gt_dir, gt_files[0])
    gt_img = Image.open(gt_path).convert('RGB')
    
    plt.subplot(1, 4, 4)
    plt.imshow(gt_img)
    plt.title(f"Ground Truth\n({gt_files[0]})")
    plt.axis('off')

    plt.tight_layout()
    plt.show()

root_path = 'BurstFlicker-S' 
visualize_burst_scene(root_path, '0200', mode='train')