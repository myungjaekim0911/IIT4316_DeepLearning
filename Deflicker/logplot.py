import pandas as pd
import matplotlib.pyplot as plt

# CSV 파일 읽기
data = pd.read_csv('training_log.csv')

plt.figure(figsize=(12, 5))

# Loss 그래프
plt.subplot(1, 2, 1)
plt.plot(data['epoch'], data['train_loss'], label='Train Loss', color='red')
plt.title('Training Loss Landscape')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# PSNR 그래프
plt.subplot(1, 2, 2)
plt.plot(data['epoch'], data['test_psnr'], label='Test PSNR', color='blue')
plt.title('Test PSNR Landscape')
plt.xlabel('Epoch')
plt.ylabel('PSNR (dB)')
plt.legend()

plt.tight_layout()
plt.savefig('learning_curve.png')
plt.show()