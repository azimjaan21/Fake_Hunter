# VAE 생성 이미지 탐지 시스템

VAE(Variational Autoencoder)로 생성된 이미지를 탐지하는 딥러닝 시스템입니다.

## 📊 성능

- **정확도**: 98% (테스트 100장 기준)
- **Real 탐지**: 98% (49/50)
- **VAE 탐지**: 98% (49/50)

## 🎯 특징

- **하이브리드 아키텍처**: CNN + 주파수 분석 (FFT)
- **웹 인터페이스**: Gradio 기반 실시간 탐지
- **GPU 가속**: CUDA 지원

---

## 🚀 빠른 시작

### 1. 환경 설정
```bash
# 가상환경 생성 (선택사항)
python -m venv venv

# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate

# 라이브러리 설치
pip install -r requirements.txt
```

**⚠️ 주의**: PyTorch는 GPU에 맞게 별도 설치 필요

- **CUDA 12.x**: 
```bash
pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu128
```

- **CUDA 11.x** 또는 **CPU만**:
```bash
pip install torch torchvision
```

---

### 2. 웹 인터페이스 실행
```bash
python web_demo.py
```

브라우저에서 자동으로 열립니다: `http://127.0.0.1:7860`

---

## 📁 파일 구조
```
vae_detector/
├── models/
│   ├── detector.py              # 기본 CNN 모델
│   └── hybrid_detector.py       # 하이브리드 모델
├── utils/
│   └── dataloader.py            # 데이터 로더
├── hybrid_vae_detector.pth      # 학습된 모델 (필수!)
├── web_demo.py                  # 웹 인터페이스
├── requirements.txt             # 필요 라이브러리
└── README.md                    # 이 파일
```

---

## 💻 사용 방법

### 웹 인터페이스

1. `python web_demo.py` 실행
2. 이미지 업로드 (드래그 앤 드롭)
3. "분석하기" 버튼 클릭
4. 결과 확인 (Real / VAE + 신뢰도)

---

## 🔬 기술 상세

### 아키텍처
```
입력 이미지
    ↓
┌─────────────┬──────────────┐
│ CNN 브랜치   │ 주파수 분석   │
│ (ResNet-18) │ (FFT)        │
└─────────────┴──────────────┘
    ↓
특징 융합
    ↓
Real / VAE 판별
```

### CNN 브랜치
- ResNet-18 백본 (ImageNet 사전학습)
- 텍스처, 구조적 패턴 학습

### 주파수 분석 브랜치
- FFT(Fast Fourier Transform)
- VAE의 고주파 손실 탐지

---

## ⚙️ 시스템 요구사항

### 최소 사양
- Python 3.8+
- RAM: 8GB
- 저장공간: 2GB

### 권장 사양
- Python 3.11
- NVIDIA GPU (CUDA 지원)
- RAM: 16GB+

---

## 📝 학습 데이터

- **Train**: 400장 (Real 200 + VAE 200)
- **Test**: 100장 (Real 50 + VAE 50)
- **출처**: CelebA (Real) + Stable Diffusion VAE (생성)

---

## ⚠️ 제한사항

1. **학습 모델**: Stable Diffusion VAE 위주
2. **일반화**: 다른 VAE/생성 모델은 성능 미검증
3. **도메인**: 얼굴 이미지 중심 (풍경, 사물 등은 재학습 필요)
4. **데이터 크기**: 소규모 (과적합 가능성)

---

## 🔧 문제 해결

### Q: "No module named 'torch'" 오류
A: PyTorch 설치 필요
```bash
pip install torch torchvision
```

### Q: "Can't open file 'hybrid_vae_detector.pth'"
A: 모델 파일이 같은 폴더에 있는지 확인

### Q: GPU를 못 찾음
A: CUDA 설치 확인 또는 CPU 모드로 실행됨 (느림)

### Q: 웹페이지가 안 열림
A: 
```bash
# 포트 변경
python web_demo.py
# 코드에서 server_port=7860 → 8080으로 변경
```

---

## 📄 라이선스

이 프로젝트는 교육/연구 목적으로 제작되었습니다.

---

## 📧 문의

문제가 발생하면 이슈를 남겨주세요.
```

### 24-5. 저장

`Ctrl + S`

---

## 📦 전달 파일 체크리스트

이제 다음 파일들을 압축해서 전달하면 됩니다:
```
✅ models/ (폴더 전체)
✅ utils/ (폴더 전체)
✅ hybrid_vae_detector.pth
✅ web_demo.py
✅ requirements.txt
✅ README.md