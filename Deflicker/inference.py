import torch
import os
from PIL import Image
from torchvision import transforms
from model import DeflickerResNet  # 기존에 만든 모델 클래스

def run_inference(model_path, input_image_path, output_path):
    # 1. 디바이스 설정 (VRAM 부족 시 'cpu'로 변경하세요)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 2. 모델 로드
    # 학습할 때 사용한 num_blocks와 동일해야 합니다 (기본 12)
    model = DeflickerResNet(in_channels=3, num_blocks=6).to(device)
    
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Successfully loaded model from {model_path}")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    model.eval()

    # 3. 이미지 로드 및 전처리 (Resize 제외, 원본 해상도 유지)
    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    try:
        img = Image.open(input_image_path).convert('RGB')
        input_tensor = transform(img).unsqueeze(0).to(device) # (1, 3, H, W)
    except Exception as e:
        print(f"Error loading image: {e}")
        return

    # 4. 모델 추론
    print(f"Processing image: {input_image_path}...")
    with torch.no_grad():
        try:
            output = model(input_tensor)
        except torch.cuda.OutOfMemoryError:
            print("CUDA Out of Memory! 원본 해상도가 너무 커서 GPU 메모리가 부족합니다.")
            print("CPU 모드로 전환하여 재시도합니다...")
            model.to('cpu')
            input_tensor = input_tensor.to('cpu')
            output = model(input_tensor)

    # 5. 결과 저장
    output_img = output.squeeze(0).cpu().clamp(0, 1) # [0, 1] 범위 제한
    output_img = transforms.ToPILImage()(output_img)
    
    # 결과 폴더 생성
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    output_img.save(output_path)
    print(f"Done! Result saved to: {output_path}")

# --- 사용 예시 ---
if __name__ == "__main__":
    # 실제 환경에 맞게 경로를 수정하세요
    MODEL_FILE = 'best_model.pth' 
    INPUT_FILE = 'BurstFlicker-S/test-resize/input/0008/277A2629.JPG' # 변환하고 싶은 원본 이미지 경로
    OUTPUT_FILE = 'inference_results/result_0008.png' # 저장될 경로
    
    run_inference(MODEL_FILE, INPUT_FILE, OUTPUT_FILE)