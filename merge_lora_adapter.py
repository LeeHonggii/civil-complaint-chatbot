"""
LoRA 어댑터를 베이스 모델에 병합

Usage:
    python merge_lora_adapter.py --model llama    # Llama 3.1 8B (기본값)
    python merge_lora_adapter.py --model mistral  # Mistral 7B v0.2
    python merge_lora_adapter.py --model gemma    # Gemma 2 9B
    python merge_lora_adapter.py --model bccard   # BCCard Llama 3 8B
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import os
import argparse

# 모델별 설정
MODEL_CONFIGS = {
    "llama": {
        "adapter_path": "./Finetuning_Models/llama_3.1",
        "base_model": "meta-llama/Llama-3.1-8B-Instruct",
        "output_path": "./Finetuning_Models/llama_3.1_merged",
        "ollama_name": "llama3.1-counselor",
        "display_name": "Llama 3.1 8B Instruct"
    },
    "mistral": {
        "adapter_path": "./Finetuning_Models/mistral-7b-instruct-v0.2",
        "base_model": "mistralai/Mistral-7B-Instruct-v0.2",
        "output_path": "./Finetuning_Models/mistral_7b_merged",
        "ollama_name": "mistral7b-counselor",
        "display_name": "Mistral 7B Instruct v0.2"
    },
    "gemma": {
        "adapter_path": "./Finetuning_Models/gemma-2-9b-it-sft-lora",
        "base_model": "google/gemma-2-9b-it",
        "output_path": "./Finetuning_Models/gemma_2_9b_merged",
        "ollama_name": "gemma2-counselor",
        "display_name": "Gemma 2 9B Instruct"
    },
    "bccard": {
        "adapter_path": "./Finetuning_Models/llama-3-kor-bccard-finance",
        "base_model": "BCCard/Llama-3-Kor-BCCard-Finance-8B",
        "output_path": "./Finetuning_Models/bccard_llama3_merged",
        "ollama_name": "bccard-llama3-counselor",
        "display_name": "BCCard Llama 3 8B"
    }
}

def merge_adapter(config):
    """LoRA 어댑터를 베이스 모델에 병합"""

    adapter_path = config["adapter_path"]
    base_model = config["base_model"]
    output_path = config["output_path"]
    display_name = config["display_name"]
    ollama_name = config["ollama_name"]

    print("=" * 60)
    print(f"LoRA 어댑터 병합 시작: {display_name}")
    print("=" * 60)

    # 1. 베이스 모델 로드
    print(f"\n[1/4] 베이스 모델 로드 중: {base_model}")
    print(f"⚠️  {display_name} 모델 다운로드 중... (시간이 걸릴 수 있습니다)")

    try:
        base_model_obj = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            low_cpu_mem_usage=True,
            trust_remote_code=True
        )
        print("✅ 베이스 모델 로드 완료")
    except Exception as e:
        print(f"❌ 베이스 모델 로드 실패: {e}")
        print("\n해결 방법:")
        print("1. Hugging Face 토큰이 필요할 수 있습니다:")
        print("   huggingface-cli login")
        print(f"2. {display_name} 모델 접근 권한이 필요합니다:")
        print(f"   https://huggingface.co/{base_model}")
        return False

    # 2. LoRA 어댑터 로드
    print(f"\n[2/4] LoRA 어댑터 로드 중: {adapter_path}")
    try:
        model = PeftModel.from_pretrained(
            base_model_obj,
            adapter_path,
            is_trainable=False
        )
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
    print(f"\n[4/4] 병합된 모델 저장 중: {output_path}")
    try:
        os.makedirs(output_path, exist_ok=True)

        # 모델 저장
        model.save_pretrained(
            output_path,
            safe_serialization=True,
            max_shard_size="2GB"
        )

        # 토크나이저 저장
        tokenizer = AutoTokenizer.from_pretrained(base_model)
        tokenizer.save_pretrained(output_path)

        print("✅ 병합된 모델 저장 완료")
        print(f"   저장 위치: {output_path}")

        # 저장된 파일 확인
        print("\n📁 저장된 파일:")
        for file in os.listdir(output_path):
            file_path = os.path.join(output_path, file)
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
    print(f"1. Ollama에 모델 등록:")
    print(f"   ollama create {ollama_name} -f Modelfile.{ollama_name.replace('-counselor', '')}")
    print(f"2. 환경 변수 설정 (.env):")
    print(f"   OLLAMA_MODEL_NAME={ollama_name}")
    print(f"   FINETUNED_MODEL_PATH={output_path}")

    return True


if __name__ == "__main__":
    # 커맨드 라인 인자 파싱
    parser = argparse.ArgumentParser(description="LoRA 어댑터를 베이스 모델에 병합")
    parser.add_argument(
        "--model",
        type=str,
        choices=["llama", "mistral", "gemma", "bccard"],
        default="llama",
        help="병합할 모델 선택 (기본값: llama)"
    )
    args = parser.parse_args()

    # 선택된 모델 설정 가져오기
    config = MODEL_CONFIGS[args.model]

    print("\n" + "=" * 60)
    print(f"선택된 모델: {config['display_name']}")
    print("=" * 60)

    # 의존성 체크
    try:
        import transformers
        import peft
        print(f"\n✅ transformers 버전: {transformers.__version__}")
        print(f"✅ peft 버전: {peft.__version__}")
    except ImportError as e:
        print(f"\n❌ 필수 라이브러리 누락: {e}")
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
    success = merge_adapter(config)

    if not success:
        exit(1)
