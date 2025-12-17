# ResNet

Residual Blocks: 18  
Kernel Size: 3 for all layers  
Padding: 1  
Activation Function: ReLU  

same above, 35 blocks  

# U-Net

Depth: 3  
Channel Width: 64>128>256>512  
Upsampling: ConvTranspose2d  

# U-Net w/ SSIM

Depth: 3  
Channel Width: 64>128>256>512  
Upsampling: ConvTranspose2d  

Charbonnier(0.8), SSIM(0.2)  