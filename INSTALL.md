# 설치 가이드

본 프로젝트는 **Mac(Ollama)** 과 **GPU(vLLM)** 환경을 모두 지원하며, 자동으로 환경을 감지합니다.

## 멀티턴 최적화

- **Mac (Ollama)**: KV Cache 자동 관리로 대화 맥락 유지
- **GPU (vLLM)**: Prefix Caching으로 2~3배 빠른 멀티턴 응답

---

## 1️⃣ Mac 환경 (Ollama)

### 필수 요구사항
- macOS (Apple Silicon 또는 Intel)
- Python 3.11+
- Ollama 설치 필요

### 설치 단계

**1. Ollama 설치**
```bash
# Homebrew로 설치
brew install ollama

# 또는 공식 사이트에서 다운로드
# https://ollama.ai/download
```

**2. Ollama 모델 다운로드**
```bash
# Llama 3.1 8B 모델 설치 (4비트 양자화)
ollama pull llama3.1:8b-instruct-q4_K_M
```

**3. Python 패키지 설치**
```bash
pip install -r requirements-mac.txt
```

**4. 환경 변수 설정**
`.env` 파일에서 Ollama 모델명 확인:
```bash
OLLAMA_MODEL_NAME="llama3.1:8b-instruct-q4_K_M"
```

**5. 실행**
```bash
python main.py
```

### 파인튜닝 모델 사용 (선택사항)

Ollama에서 파인튜닝 모델을 사용하려면:

**1. Modelfile 생성**
```bash
cat > Modelfile << EOF
FROM ./Finetuning_Models/llama_3.1
TEMPLATE """{{ .System }}

{{ .Prompt }}"""
PARAMETER temperature 0.3
PARAMETER top_p 0.9
EOF
```

**2. 모델 생성**
```bash
ollama create counselor-finetuned -f Modelfile
```

**3. `.env` 파일 수정**
```bash
OLLAMA_MODEL_NAME="counselor-finetuned"
```

---

## 2️⃣ GPU 환경 (vLLM)

### 필수 요구사항
- Linux 또는 Windows (WSL2)
- NVIDIA GPU (CUDA 12.0+)
- Python 3.11+
- 최소 16GB GPU 메모리 (Llama 3.1 8B 기준)

### 설치 단계

**1. CUDA 설치 확인**
```bash
nvidia-smi
```

**2. Python 패키지 설치**
```bash
pip install -r requirements-gpu.txt
```

**3. HuggingFace 로그인 (Llama 모델 사용 시 필요)**
```bash
huggingface-cli login
```

**4. 환경 변수 설정**
`.env` 파일에서 LoRA 어댑터 경로 확인:
```bash
FINETUNED_MODEL_PATH="./Finetuning_Models/llama_3.1"
```

**5. 실행**
```bash
python main.py
```

### GPU 메모리 부족 시

`models/vllm_wrapper.py:48` 에서 메모리 사용률 조정:
```python
gpu_memory_utilization=0.7  # 기본값 0.9에서 낮춤
```

---

## 3️⃣ 환경 강제 지정 (선택사항)

자동 감지를 무시하고 특정 환경을 강제하려면:

`.env` 파일에 추가:
```bash
FORCE_ENVIRONMENT="mac"   # 또는 "gpu"
```

---

## 4️⃣ 확인 사항

### Mac 환경
```bash
# Ollama 실행 확인
ollama list

# 모델이 있는지 확인
ollama list | grep llama3.1
```

### GPU 환경
```bash
# CUDA 확인
nvidia-smi

# PyTorch CUDA 확인
python -c "import torch; print(torch.cuda.is_available())"
```

---

## 📝 패키지 구조

```
requirements-base.txt     # 공통 패키지
requirements-mac.txt      # Mac용 (Ollama + Transformers)
requirements-gpu.txt      # GPU용 (vLLM)
requirements.txt          # 기존 파일 (더 이상 사용 안 함)
```

---

## 🔧 트러블슈팅

### Ollama 연결 오류
```bash
# Ollama 서버 재시작
brew services restart ollama
```

### vLLM 설치 오류
```bash
# CUDA 버전 확인 후 호환되는 PyTorch 설치
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

### 모델 로드 오류
```bash
# 환경 감지 확인
python -c "from models.env_detector import detect_environment; print(detect_environment())"
```

---

## 🚀 성능 비교

| 환경 | 초기 응답 | 멀티턴 응답 | 메모리 |
|------|-----------|-------------|--------|
| **Mac (Ollama)** | ~2초 | ~1초 (KV cache) | ~4GB |
| **GPU (vLLM)** | ~1초 | ~0.3초 (Prefix cache) | ~14GB |

---

## 문의

문제가 발생하면 이슈를 등록해주세요!
