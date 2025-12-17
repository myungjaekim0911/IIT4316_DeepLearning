import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import os
import csv

# 커스텀 파일들 임포트
from unet import DeflickerUNet
from dataset import BurstDeflickerDataset
from loss import CharbonnierLoss, SSIMLoss, calculate_psnr

# --- [1] 하이퍼파라미터 및 설정 ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
num_epochs = 50
batch_size = 4  # VRAM 최적화를 위해 1 유지
learning_rate = 1e-4
root_path = './BurstFlicker-S' # 데이터셋 경로
save_dir = './checkpoints'
os.makedirs(save_dir, exist_ok=True)
os.makedirs('./eval_results', exist_ok=True)

# --- [2] 데이터 로더 (Resize 및 3장 추출 반영) ---
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

train_dataset = BurstDeflickerDataset(root_path, mode='train', transform=transform)
test_dataset = BurstDeflickerDataset(root_path, mode='test', transform=transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# --- [3] 모델, 손실함수, 최적화 도구 ---
model = DeflickerUNet(in_channels=3).to(device) # U-Net 모델
criterion_pixel = CharbonnierLoss().to(device)
criterion_ssim = SSIMLoss().to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 로그 기록용 CSV 설정
log_file = 'training_log.csv'
with open(log_file, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['epoch', 'train_loss', 'test_psnr'])

# --- [4] 학습 루프 ---
print(f"Starting Training on {device}...")
print(f"Total training samples: {len(train_dataset)}")

best_psnr = 0.0

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    
    for i, (inputs, gts) in enumerate(train_loader):
        inputs, gts = inputs.to(device), gts.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        
        # 하이브리드 손실 함수 (Pixel + SSIM)
        loss_pixel = criterion_pixel(outputs, gts)
        loss_ssim = criterion_ssim(outputs, gts)
        total_loss = loss_pixel + (0.2 * loss_ssim)
        
        total_loss.backward()
        optimizer.step()
        
        running_loss += total_loss.item()
        
    avg_train_loss = running_loss / len(train_loader)
    
    # --- [5] 에폭별 평가 (Evaluation) ---
    model.eval()
    total_psnr = 0.0
    with torch.no_grad():
        for inputs, gts in test_loader:
            inputs, gts = inputs.to(device), gts.to(device)
            outputs = model(inputs)
            total_psnr += calculate_psnr(outputs, gts)
            
            # 마지막 에폭 샘플 이미지 저장 (비교용)
            if i == 0: 
                combined = torch.cat([inputs[0], outputs[0], gts[0]], dim=2)
                transforms.ToPILImage()(combined.cpu()).save(f'./eval_results/epoch_{epoch+1:03d}.png')

    avg_psnr = total_psnr / len(test_loader)
    print(f"Epoch [{epoch+1}/{num_epochs}] Loss: {avg_train_loss:.6f} | Test PSNR: {avg_psnr:.2f} dB")
    
    # 로그 저장
    with open(log_file, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([epoch+1, avg_train_loss, avg_psnr])
        
    # Best 모델 저장
    if avg_psnr > best_psnr:
        best_psnr = avg_psnr
        torch.save(model.state_dict(), os.path.join(save_dir, 'best_model.pth'))
        print(f"--> Best Model Saved! (PSNR: {avg_psnr:.2f})")

print("Training Complete.")