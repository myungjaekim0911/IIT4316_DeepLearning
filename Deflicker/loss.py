import torch
import torch.nn as nn
import torch.nn.functional as F

# 1. Charbonnier Loss
class CharbonnierLoss(nn.Module):
    def __init__(self, eps=1e-3):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, pred, gt):
        return torch.mean(torch.sqrt((pred - gt)**2 + self.eps**2))

# 2. PSNR 계산 함수
def calculate_psnr(pred, gt):
    mse = F.mse_loss(pred, gt)
    if mse == 0:
        return float('inf')
    # 데이터가 [0, 1] 범위라고 가정할 때 PIXEL_MAX는 1.0
    psnr = 20 * torch.log10(1.0 / torch.sqrt(mse))
    return psnr.item()