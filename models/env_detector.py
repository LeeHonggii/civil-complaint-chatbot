"""
환경 자동 감지 유틸리티
- GPU 환경 (CUDA) → vLLM 사용
- Mac 환경 (Apple Silicon) → Ollama 사용
"""

import platform
import os


def detect_environment():
    """
    실행 환경을 자동 감지

    Returns:
        str: 'gpu' 또는 'mac'
    """
    # 환경 변수로 강제 지정 가능
    force_env = os.getenv("FORCE_ENVIRONMENT")
    if force_env in ["gpu", "mac"]:
        print(f"[환경 감지] 강제 지정: {force_env.upper()}")
        return force_env

    system = platform.system()
    machine = platform.machine()

    # Mac 감지 (Apple Silicon 또는 Intel)
    if system == "Darwin":
        print(f"[환경 감지] Mac 감지 ({machine})")
        print(f"[환경 감지] → Ollama 사용")
        return "mac"

    # GPU(CUDA) 가능 여부 확인
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_count = torch.cuda.device_count()
            memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)

            print(f"[환경 감지] GPU 감지!")
            print(f"  - GPU 이름: {gpu_name}")
            print(f"  - GPU 개수: {gpu_count}")
            print(f"  - VRAM: {memory_gb:.1f}GB")

            # vLLM 설치 여부 확인
            try:
                import vllm
                print(f"  - vLLM 설치됨: v{vllm.__version__}")
                print(f"[환경 감지] → vLLM 사용")
                return "gpu"
            except ImportError:
                print(f"  - ⚠️ vLLM 미설치 (pip install vllm)")
                print(f"[환경 감지] → Ollama로 fallback")
                return "mac"
        else:
            print(f"[환경 감지] CUDA 사용 불가 (GPU 없거나 드라이버 문제)")
    except ImportError:
        print(f"[환경 감지] PyTorch 미설치")

    # 기본값: Mac (안전한 선택)
    print(f"[환경 감지] 기본값: Mac 모드 (Ollama) 사용")
    return "mac"


def is_gpu_available():
    """GPU(CUDA) 사용 가능 여부 확인"""
    return detect_environment() == "gpu"


def is_mac_environment():
    """Mac 환경 여부 확인"""
    return detect_environment() == "mac"
