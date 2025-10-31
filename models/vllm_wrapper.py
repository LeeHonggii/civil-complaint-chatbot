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
        # 병합된 모델 경로 사용
        self.merged_model_path = os.getenv(
            "FINETUNED_MODEL_PATH",
            "./Finetuning_Models/llama_3.1_merged"
        )

    def load_model(self):
        """모델 로드 (한 번만 실행)"""
        if self.model is not None:
            return

        print(f"[vLLM] 파인튜닝된 모델 로드 중: {self.merged_model_path}")

        self.model = self.LLM(
            model=self.merged_model_path,
            enable_prefix_caching=True,  # ✨ Prefix Caching 활성화 (멀티턴 최적화)
            tensor_parallel_size=1,
            gpu_memory_utilization=0.9,
            trust_remote_code=True
        )
        print(f"[vLLM] 파인튜닝 모델 로드 완료! (Prefix Caching ON)")

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


def get_vllm_wrapper() -> VLLMWrapper:
    """vLLM Wrapper 싱글톤 인스턴스 반환"""
    global _vllm_instance
    if _vllm_instance is None:
        _vllm_instance = VLLMWrapper()
    return _vllm_instance
