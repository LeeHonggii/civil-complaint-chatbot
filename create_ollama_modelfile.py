"""
Ollama Modelfile 생성

병합된 모델을 Ollama로 가져오기 위한 Modelfile 생성
"""

import os

# 경로 설정
MERGED_MODEL_PATH = "./Finetuning_Models/llama_3.1_merged"
MODELFILE_PATH = "./Modelfile"

def create_modelfile():
    """Ollama Modelfile 생성"""

    print("=" * 60)
    print("Ollama Modelfile 생성")
    print("=" * 60)

    # 병합된 모델 존재 확인
    if not os.path.exists(MERGED_MODEL_PATH):
        print(f"❌ 병합된 모델을 찾을 수 없습니다: {MERGED_MODEL_PATH}")
        print("\n먼저 LoRA 어댑터를 병합하세요:")
        print("  python merge_lora_adapter.py")
        return False

    # Modelfile 내용
    modelfile_content = f"""# Llama 3.1 8B Counselor (Fine-tuned)
# 상담 데이터로 파인튜닝된 모델

FROM {os.path.abspath(MERGED_MODEL_PATH)}

# 파라미터 설정
PARAMETER temperature 0.3
PARAMETER top_p 0.9
PARAMETER top_k 40
PARAMETER num_ctx 4096
PARAMETER stop "<|eot_id|>"
PARAMETER stop "<|end_of_text|>"

# 시스템 프롬프트 (상담 전문가)
SYSTEM \"\"\"당신은 전문 상담사입니다.

## 답변 원칙
1. 구체적이고 명확한 답변 제공
2. 이전 대화 맥락을 고려하여 자연스럽게 답변
3. 정보가 불확실하면 "확인이 필요합니다" 안내
4. 친근하고 자연스러운 말투
5. 간결하게 답변 (불필요한 설명 자제)
\"\"\"

# 템플릿 설정 (Llama 3.1 형식)
TEMPLATE \"\"\"{{- if .System }}
<|start_header_id|>system<|end_header_id|>

{{ .System }}<|eot_id|>
{{- end }}
<|start_header_id|>user<|end_header_id|>

{{ .Prompt }}<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>

\"\"\"
"""

    # Modelfile 저장
    try:
        with open(MODELFILE_PATH, "w", encoding="utf-8") as f:
            f.write(modelfile_content)

        print(f"✅ Modelfile 생성 완료: {MODELFILE_PATH}")
        print("\n📄 내용:")
        print("-" * 60)
        print(modelfile_content)
        print("-" * 60)

    except Exception as e:
        print(f"❌ Modelfile 생성 실패: {e}")
        return False

    print("\n" + "=" * 60)
    print("✅ Modelfile 생성 완료!")
    print("=" * 60)
    print("\n다음 단계:")
    print("1. Ollama에 모델 등록:")
    print("   ollama create llama3.1-counselor -f Modelfile")
    print("\n2. 모델 테스트:")
    print("   ollama run llama3.1-counselor \"안녕하세요\"")
    print("\n3. 환경 변수 설정 (.env):")
    print("   OLLAMA_MODEL_NAME=llama3.1-counselor")

    return True


if __name__ == "__main__":
    success = create_modelfile()

    if not success:
        exit(1)
