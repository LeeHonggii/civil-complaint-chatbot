"""
LoRA 어댑터를 베이스 모델에 병합

Usage:
    python merge_lora_adapter.py
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import os

# 경로 설정
ADAPTER_PATH = "./Finetuning_Models/llama_3.1"
BASE_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
OUTPUT_PATH = "./Finetuning_Models/llama_3.1_merged"

def merge_adapter():
    """LoRA 어댑터를 베이스 모델에 병합"""

    print("=" * 60)
    print("LoRA 어댑터 병합 시작")
    print("=" * 60)

    # 1. 베이스 모델 로드
    print(f"\n[1/4] 베이스 모델 로드 중: {BASE_MODEL}")
    print("⚠️  Llama 3.1 8B 모델 다운로드 중... (시간이 걸릴 수 있습니다)")

    try:
        base_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            low_cpu_mem_usage=True
        )
        print("✅ 베이스 모델 로드 완료")
    except Exception as e:
        print(f"❌ 베이스 모델 로드 실패: {e}")
        print("\n해결 방법:")
        print("1. Hugging Face 토큰이 필요할 수 있습니다:")
        print("   huggingface-cli login")
        print("2. meta-llama 모델 접근 권한이 필요합니다:")
        print("   https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct")
        return False

    # 2. LoRA 어댑터 로드
    print(f"\n[2/4] LoRA 어댑터 로드 중: {ADAPTER_PATH}")
    try:
        model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
        print("✅ LoRA 어댑터 로드 완료")
    except Exception as e:
        print(f"❌ LoRA 어댑터 로드 실패: {e}")
        return False

    # 3. 병합
    print(f"\n[3/4] LoRA 어댑터를 베이스 모델에 병합 중...")
    try:
        model = model.merge_and_unload()
        print("✅ 병합 완료")
    except Exception as e:
        print(f"❌ 병합 실패: {e}")
        return False

    # 4. 저장
    print(f"\n[4/4] 병합된 모델 저장 중: {OUTPUT_PATH}")
    try:
        os.makedirs(OUTPUT_PATH, exist_ok=True)

        # 모델 저장
        model.save_pretrained(
            OUTPUT_PATH,
            safe_serialization=True,
            max_shard_size="2GB"
        )

        # 토크나이저 저장
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
        tokenizer.save_pretrained(OUTPUT_PATH)

        print("✅ 병합된 모델 저장 완료")
        print(f"   저장 위치: {OUTPUT_PATH}")

        # 저장된 파일 확인
        print("\n📁 저장된 파일:")
        for file in os.listdir(OUTPUT_PATH):
            file_path = os.path.join(OUTPUT_PATH, file)
            if os.path.isfile(file_path):
                size_mb = os.path.getsize(file_path) / (1024 * 1024)
                print(f"   - {file} ({size_mb:.1f} MB)")

    except Exception as e:
        print(f"❌ 저장 실패: {e}")
        return False

    print("\n" + "=" * 60)
    print("✅ 모든 작업 완료!")
    print("=" * 60)
    print("\n다음 단계:")
    print("1. Ollama Modelfile 생성:")
    print("   python create_ollama_modelfile.py")
    print("2. Ollama에 모델 등록:")
    print("   ollama create llama3.1-counselor -f Modelfile")

    return True


if __name__ == "__main__":
    # 의존성 체크
    try:
        import transformers
        import peft
        print(f"✅ transformers 버전: {transformers.__version__}")
        print(f"✅ peft 버전: {peft.__version__}")
    except ImportError as e:
        print(f"❌ 필수 라이브러리 누락: {e}")
        print("\n설치 명령어:")
        print("  pip install transformers peft accelerate bitsandbytes")
        exit(1)

    # GPU 확인
    if torch.cuda.is_available():
        print(f"✅ GPU 사용 가능: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        print("✅ Apple Silicon (MPS) 사용 가능")
    else:
        print("⚠️  CPU 모드 (느릴 수 있습니다)")

    print()

    # 병합 실행
    success = merge_adapter()

    if not success:
        exit(1)
