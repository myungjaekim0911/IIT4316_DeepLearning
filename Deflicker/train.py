import torch
import csv
import os
from torch.utils.data import DataLoader
from torchvision import transforms
from dataset import BurstDeflickerDataset
from resnet import DeflickerResNet
from unet import DeflickerUNet
from loss import CharbonnierLoss
from evaluate import evaluate_model

# 0. 메모리 초기화 및 환경 설정
torch.cuda.empty_cache()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 하이퍼파라미터
root_path = 'BurstFlicker-S'
epochs = 50   # 원하는 에폭 수로 조절하세요
lr = 1e-4
batch_size = 4     # 단일 이미지 방식이므로 4~8 정도가 적당합니다

# 1. 데이터셋 및 로더 준비 (Single-Image 방식의 Dataset 호출)
transform = transforms.Compose([
    transforms.Resize((256, 256)), # 비율 상관없이 256x256으로 고정
    transforms.ToTensor()
])
train_dataset = BurstDeflickerDataset(root_path, mode='train', transform=transform)
test_dataset = BurstDeflickerDataset(root_path, mode='test', transform=transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
# 평가는 시각화를 위해 batch_size=1로 유지하는 것이 좋습니다
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# 2. 모델 선언 (in_channels=3 확인)
# model = DeflickerUNet(in_channels=3).to(device)
model = DeflickerResNet(in_channels=3, num_blocks=34).to(device)
criterion = CharbonnierLoss().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

# 3. 로그 저장을 위한 CSV 설정
csv_file = 'training_log.csv'
if not os.path.exists(csv_file):
    with open(csv_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'train_loss', 'test_psnr'])

best_psnr = 0.0

print(f"Starting Training on {device}...")
print(f"Total training samples: {len(train_dataset)}")

# 4. 학습 루프
for epoch in range(1, epochs + 1):
    model.train()
    running_loss = 0.0
    
    for inputs, gts in train_loader:
        inputs, gts = inputs.to(device), gts.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, gts)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    
    avg_loss = running_loss / len(train_loader)
    
    # 5. 에폭 끝날 때마다 평가 및 샘플 이미지 저장
    # evaluate_model에 epoch 인자를 전달하여 시각화 기능을 활성화합니다.
    avg_psnr = evaluate_model(model, test_loader, device, epoch=epoch, save_path='eval_results')
    
    print(f"Epoch [{epoch}/{epochs}] Loss: {avg_loss:.6f} | Test PSNR: {avg_psnr:.2f} dB")
    
    # CSV에 기록
    with open(csv_file, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([epoch, avg_loss, avg_psnr])
        
    # Best 모델 저장 (PSNR 기준)
    if avg_psnr > best_psnr:
        best_psnr = avg_psnr
        torch.save(model.state_dict(), 'best_model.pth')
        print(f"--> Best Model Saved! (PSNR: {best_psnr:.2f})")

# 최종 모델 저장
torch.save(model.state_dict(), 'final_model.pth')
print("Training Complete.")