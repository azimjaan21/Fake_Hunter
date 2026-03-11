import gradio as gr
import torch
from torchvision import transforms
from PIL import Image
from models.hybrid_detector import get_hybrid_model

# 전역 변수
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = None

def load_model():
    """모델 로드"""
    global model
    print("모델 로딩 중...")
    model = get_hybrid_model(device)
    model.load_state_dict(torch.load('hybrid_vae_detector.pth'))
    model.eval()
    print("✅ 모델 로드 완료!")

def predict_image(image):
    """이미지 예측"""
    if image is None:
        return "이미지를 업로드해주세요.", None
    
    # 이미지 전처리
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    # PIL Image로 변환
    if not isinstance(image, Image.Image):
        image = Image.fromarray(image)
    
    img_tensor = transform(image).unsqueeze(0).to(device)
    
    # 예측
    with torch.no_grad():
        output = model(img_tensor)
        probabilities = torch.softmax(output, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0][predicted_class].item() * 100
        real_prob = probabilities[0][0].item() * 100
        vae_prob = probabilities[0][1].item() * 100
    
    # 결과 텍스트
    if predicted_class == 0:
        result = f"🟢 **Real 이미지** (신뢰도: {confidence:.2f}%)"
        color = "green"
    else:
        result = f"🔴 **VAE 생성 이미지** (신뢰도: {confidence:.2f}%)"
        color = "red"
    
    # 상세 정보
    details = f"""
### 예측 결과
- **판정**: {result}

### 확률 분포
- Real 확률: {real_prob:.2f}%
- VAE 확률: {vae_prob:.2f}%

### 해석
{'이 이미지는 실제 카메라로 촬영된 진짜 이미지로 판단됩니다.' if predicted_class == 0 else '이 이미지는 VAE 생성 모델로 만들어진 합성 이미지로 판단됩니다.'}
"""
    
    # 확률 시각화용 딕셔너리
    prob_dict = {
        "Real": real_prob / 100,
        "VAE Generated": vae_prob / 100
    }
    
    return details, prob_dict

# Gradio 인터페이스
def create_interface():
    """웹 인터페이스 생성"""
    
    with gr.Blocks(title="VAE 이미지 탐지기") as demo:
        gr.Markdown("""
        # 🔍 VAE 생성 이미지 탐지 시스템
        
        이미지를 업로드하면 **Real(진짜)** 인지 **VAE 생성(가짜)** 인지 판별합니다.
        
        **정확도: 98%** | **CNN + 주파수 분석 하이브리드 모델**
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                input_image = gr.Image(
                    label="이미지 업로드",
                    type="pil",
                    height=400
                )
                
                predict_btn = gr.Button("🔍 분석하기", variant="primary", size="lg")
                
                gr.Markdown("""
                ### 📌 사용 방법
                1. 이미지를 드래그 앤 드롭하거나 클릭해서 업로드
                2. "분석하기" 버튼 클릭
                3. 결과 확인
                
                ### ✅ 지원 형식
                JPG, PNG, JPEG
                """)
            
            with gr.Column(scale=1):
                output_text = gr.Markdown(label="분석 결과")
                output_plot = gr.Label(label="확률 분포", num_top_classes=2)
        
        # 예시 이미지
        gr.Markdown("### 📸 예시 이미지로 테스트")
        gr.Examples(
            examples=[
                ["dataset/test/real/real_0000.jpg"],
                ["dataset/test/vae/vae_real_0000.jpg"],
            ],
            inputs=input_image,
            label="예시 이미지 (클릭하면 자동 로드)"
        )
        
        # 이벤트 연결
        predict_btn.click(
            fn=predict_image,
            inputs=input_image,
            outputs=[output_text, output_plot]
        )
        
        gr.Markdown("""
        ---
        ### 📊 모델 정보
        - **아키텍처**: ResNet-18 + 주파수 분석 (FFT)
        - **학습 데이터**: 400장 (Real 200 + VAE 200)
        - **테스트 정확도**: 98%
        - **Real 탐지율**: 98% (49/50)
        - **VAE 탐지율**: 98% (49/50)
        """)
    
    return demo

if __name__ == '__main__':
    # 모델 로드
    load_model()
    
    # 웹 인터페이스 실행
    demo = create_interface()
    demo.launch(
        share=False,  # True로 하면 공개 링크 생성
        server_name="127.0.0.1",
        server_port=7860
    )