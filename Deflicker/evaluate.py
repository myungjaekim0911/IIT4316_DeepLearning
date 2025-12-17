import torch
import os
from torchvision.utils import save_image
from loss import calculate_psnr

def evaluate_model(model, test_loader, device, epoch=None, save_path='eval_results'):
    model.eval()
    total_psnr = 0.0
    image_count = 0
    
    # 결과 저장을 위한 폴더 생성
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    with torch.no_grad():
        for i, (inputs, gt) in enumerate(test_loader):
            inputs = inputs.to(device)
            gt = gt.to(device)
            
            # 모델 추론
            output = model(inputs)
            
            # PSNR 계산
            psnr = calculate_psnr(output, gt)
            total_psnr += psnr
            image_count += 1
            
            # 매 에폭의 첫 번째 배치 이미지만 샘플로 저장 (시각화)
            if i == 0 and epoch is not None:
                # 입력, 결과, 정답을 가로로 이어붙임 (Input | Output | GT)
                # output은 clamp를 통해 [0, 1] 범위를 유지하도록 함
                comparison = torch.cat([inputs, output.clamp(0, 1), gt], dim=3)
                save_image(comparison, os.path.join(save_path, f'epoch_{epoch:03d}_sample.png'))
                
    avg_psnr = total_psnr / image_count if image_count > 0 else 0
    return avg_psnr