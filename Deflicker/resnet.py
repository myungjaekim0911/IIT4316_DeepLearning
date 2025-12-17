import torch
import torch.nn as nn

class ResBlock(nn.Module):
    def __init__(self, channels):
        super(ResBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        )

    def forward(self, x):
        return x + self.conv(x)

class DeflickerResNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, num_blocks=18):
        super(DeflickerResNet, self).__init__()
        
        # 입력을 받아 64채널로 확장
        self.initial = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        
        # 깊은 특징 추출을 위한 잔차 블록
        self.res_blocks = nn.Sequential(
            *[ResBlock(64) for _ in range(num_blocks)]
        )
        
        # 최종 RGB 이미지 복원
        self.final = nn.Conv2d(64, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        identity = x # 입력 자체를 기억
        out = self.initial(x)
        out = self.res_blocks(out)
        out = self.final(out)
        return identity + out # Global Residual: 입력에 변화량(잔차)만 더함