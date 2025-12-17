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
    
class SSIMLoss(nn.Module):
    def __init__(self, window_size=11):
        super(SSIMLoss, self).__init__()
        self.window_size = window_size
        self.register_buffer('window', self._create_window(window_size))

    def _create_window(self, window_size):
        # 가우시안 커널 생성
        def gaussian(window_size, sigma):
            gauss = torch.exp(torch.Tensor([-(x - window_size//2)**2 / float(2*sigma**2) for x in range(window_size)]))
            return gauss/gauss.sum()
        _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        return _2D_window.expand(3, 1, window_size, window_size).contiguous()

    def forward(self, img1, img2):
        # SSIM 공식 계산
        mu1 = F.conv2d(img1, self.window, padding=self.window_size//2, groups=3)
        mu2 = F.conv2d(img2, self.window, padding=self.window_size//2, groups=3)
        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2
        sigma1_sq = F.conv2d(img1 * img1, self.window, padding=self.window_size//2, groups=3) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, self.window, padding=self.window_size//2, groups=3) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, self.window, padding=self.window_size//2, groups=3) - mu1_mu2
        C1 = 0.01**2
        C2 = 0.03**2
        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        return 1 - ssim_map.mean()

# 2. PSNR 계산 함수
def calculate_psnr(pred, gt):
    mse = F.mse_loss(pred, gt)
    if mse == 0:
        return float('inf')
    # 데이터가 [0, 1] 범위라고 가정할 때 PIXEL_MAX는 1.0
    psnr = 20 * torch.log10(1.0 / torch.sqrt(mse))
    return psnr.item()