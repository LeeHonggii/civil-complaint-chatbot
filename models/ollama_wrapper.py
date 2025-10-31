"""
Ollama Wrapper (Mac 환경용)
- KV Cache 자동 관리
- 세션 기반 멀티턴 대화
"""

import os
from typing import List, Optional


class OllamaWrapper:
    """Ollama 기반 LLM Wrapper (KV Cache 자동 관리)"""

    def __init__(self):
        import ollama
        self.client = ollama.Client()
        # 파인튜닝된 상담 모델 사용
        self.model_name = os.getenv("OLLAMA_MODEL_NAME", "llama3.1-counselor")

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
            stop: 중단 토큰 리스트 (Ollama에서는 사용 제한적)

        Returns:
            생성된 텍스트
        """
        try:
            response = self.client.generate(
                model=self.model_name,
                prompt=prompt,
                options={
                    "temperature": temperature,
                    "top_p": top_p,
                    "num_predict": max_tokens,
                },
                # KV cache는 Ollama가 자동으로 관리
                keep_alive="5m"  # 모델을 메모리에 5분간 유지 (KV cache 유지)
            )
            return response['response'].strip()

        except Exception as e:
            print(f"[Ollama] 생성 오류: {e}")
            raise

    def check_model_exists(self) -> bool:
        """모델이 설치되어 있는지 확인"""
        try:
            import ollama
            result = ollama.list()
            # ollama.list()는 Pydantic 모델을 반환 (models 속성이 Model 객체 리스트)
            model_list = getattr(result, 'models', [])
            model_names = [model.model for model in model_list]

            # :latest 태그 포함/미포함 모두 체크
            return (self.model_name in model_names or
                    f"{self.model_name}:latest" in model_names)
        except Exception as e:
            print(f"[Ollama] 모델 확인 중 오류: {e}")
            return False


# 싱글톤 인스턴스
_ollama_instance = None


def get_ollama_wrapper() -> OllamaWrapper:
    """Ollama Wrapper 싱글톤 인스턴스 반환"""
    global _ollama_instance
    if _ollama_instance is None:
        _ollama_instance = OllamaWrapper()

        # 모델 존재 여부 확인 및 안내
        if not _ollama_instance.check_model_exists():
            print(f"[Ollama] 경고: 모델 '{_ollama_instance.model_name}'이 설치되지 않았습니다.")
            print(f"[Ollama] 다음 명령어로 모델을 설치하세요:")
            print(f"  ollama pull {_ollama_instance.model_name}")
        else:
            print(f"[Ollama] 모델 '{_ollama_instance.model_name}' 사용 준비 완료")

    return _ollama_instance
