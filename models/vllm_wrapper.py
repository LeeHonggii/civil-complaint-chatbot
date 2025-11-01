"""
vLLM Wrapper (GPU 환경용)
- LoRA 어댑터 지원
- Prefix Caching으로 멀티턴 최적화
"""

import os
from typing import List, Dict, Optional


class VLLMWrapper:
    """vLLM 기반 LLM Wrapper (LoRA + Prefix Caching)"""

    def __init__(self):
        from vllm import LLM, SamplingParams

        self.LLM = LLM
        self.SamplingParams = SamplingParams
        self.model = None

        # 모델 선택
        model_name = os.getenv("MODEL_NAME", "llama").lower()
        use_finetuned = os.getenv("USE_FINETUNED_MODEL", "false").lower() == "true"

        if model_name == "mistral":
            if use_finetuned:
                self.merged_model_path = os.getenv("MISTRAL_MODEL_PATH", "./Finetuning_Models/mistral_7b_merged")
                self.model_display_name = "Mistral 7B Instruct v0.2 (파인튜닝)"
            else:
                self.merged_model_path = "mistralai/Mistral-7B-Instruct-v0.2"
                self.model_display_name = "Mistral 7B Instruct v0.2 (베이스)"
        elif model_name == "gemma":
            if use_finetuned:
                self.merged_model_path = os.getenv("GEMMA_MODEL_PATH", "./Finetuning_Models/gemma_2_9b_merged")
                self.model_display_name = "Gemma 2 9B Instruct (파인튜닝)"
            else:
                self.merged_model_path = "google/gemma-2-9b-it"
                self.model_display_name = "Gemma 2 9B Instruct (베이스)"
        elif model_name == "bccard":
            if use_finetuned:
                self.merged_model_path = os.getenv("BCCARD_MODEL_PATH", "./Finetuning_Models/bccard_llama3_merged")
                self.model_display_name = "BCCard Llama 3 8B (파인튜닝)"
            else:
                self.merged_model_path = "BCCard/Llama-3-Kor-BCCard-Finance-8B"
                self.model_display_name = "BCCard Llama 3 8B (베이스)"
        else:  # llama (기본값)
            if use_finetuned:
                self.merged_model_path = os.getenv("LLAMA_MODEL_PATH", "./Finetuning_Models/llama_3.1_merged")
                self.model_display_name = "Llama 3.1 8B Instruct (파인튜닝)"
            else:
                self.merged_model_path = "meta-llama/Meta-Llama-3.1-8B-Instruct"
                self.model_display_name = "Llama 3.1 8B Instruct (베이스)"

    def load_model(self):
        """모델 로드 (한 번만 실행)"""
        if self.model is not None:
            return

        print(f"[vLLM] 파인튜닝된 모델 로드 중: {self.model_display_name}")
        print(f"[vLLM] 경로: {self.merged_model_path}")

        self.model = self.LLM(
            model=self.merged_model_path,
            enable_prefix_caching=True,  # ✨ Prefix Caching 활성화 (멀티턴 최적화)
            tensor_parallel_size=1,
            gpu_memory_utilization=0.9,
            trust_remote_code=True
        )
        print(f"[vLLM] {self.model_display_name} 로드 완료! (Prefix Caching ON)")

    def generate(
        self,
        prompt: str,
        temperature: float = 0.3,
        top_p: float = 0.9,
        max_tokens: int = 512,
        stop: Optional[List[str]] = None
    ) -> str:
        """
        텍스트 생성

        Args:
            prompt: 프롬프트
            temperature: 생성 온도
            top_p: Top-p 샘플링
            max_tokens: 최대 토큰 수
            stop: 중단 토큰 리스트

        Returns:
            생성된 텍스트
        """
        self.load_model()

        # 샘플링 파라미터
        sampling_params = self.SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            stop=stop or ["고객:", "\n\n\n"]
        )

        # 생성 (병합된 모델 사용, prefix caching 자동 적용됨)
        outputs = self.model.generate(
            [prompt],
            sampling_params
        )

        return outputs[0].outputs[0].text.strip()


# 싱글톤 인스턴스
_vllm_instance = None


def reset_vllm_wrapper():
    """vLLM Wrapper 싱글톤 초기화 (모델 변경 시 사용)"""
    global _vllm_instance
    _vllm_instance = None
    print("[vLLM] 싱글톤 인스턴스 초기화됨")


def get_vllm_wrapper() -> VLLMWrapper:
    """vLLM Wrapper 싱글톤 인스턴스 반환"""
    global _vllm_instance
    if _vllm_instance is None:
        _vllm_instance = VLLMWrapper()
    return _vllm_instance
