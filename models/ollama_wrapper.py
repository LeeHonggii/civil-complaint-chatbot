"""
Ollama Wrapper (Mac 환경용)
- KV Cache 자동 관리
- 세션 기반 멀티턴 대화
- 스트리밍 응답 지원
"""

import os
import logging
from typing import List, Optional, Iterator

# 로거 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class OllamaWrapper:
    """Ollama 기반 LLM Wrapper (KV Cache 자동 관리)"""

    def __init__(self):
        import ollama
        self.client = ollama.Client()

        # 모델 선택
        model_name = os.getenv("MODEL_NAME", "llama").lower()
        use_finetuned = os.getenv("USE_FINETUNED_MODEL", "false").lower() == "true"

        if model_name == "mistral":
            if use_finetuned:
                self.model_name = os.getenv("MISTRAL_OLLAMA_NAME", "mistral7b-counselor")
                self.model_display_name = "Mistral 7B Instruct v0.2 (파인튜닝)"
            else:
                self.model_name = os.getenv("MISTRAL_BASE_OLLAMA_NAME", "mistral:latest")
                self.model_display_name = "Mistral 7B Instruct v0.2 (베이스)"
        elif model_name == "gemma":
            if use_finetuned:
                self.model_name = os.getenv("GEMMA_OLLAMA_NAME", "gemma2-counselor")
                self.model_display_name = "Gemma 2 9B Instruct (파인튜닝)"
            else:
                self.model_name = os.getenv("GEMMA_BASE_OLLAMA_NAME", "gemma2:9b-instruct-q4_K_M")
                self.model_display_name = "Gemma 2 9B Instruct (베이스)"
        elif model_name == "bccard":
            if use_finetuned:
                self.model_name = os.getenv("BCCARD_OLLAMA_NAME", "bccard-llama3-counselor")
                self.model_display_name = "BCCard Llama 3 8B (파인튜닝)"
            else:
                self.model_name = os.getenv("BCCARD_BASE_OLLAMA_NAME", "hf.co/BCCard/Llama-3-Kor-BCCard-Finance-8B:latest")
                self.model_display_name = "BCCard Llama 3 8B (베이스)"
        else:  # llama (기본값)
            if use_finetuned:
                self.model_name = os.getenv("LLAMA_OLLAMA_NAME", "llama3.1-counselor")
                self.model_display_name = "Llama 3.1 8B Instruct (파인튜닝)"
            else:
                self.model_name = os.getenv("LLAMA_BASE_OLLAMA_NAME", "llama3.1:8b-instruct-q4_K_M")
                self.model_display_name = "Llama 3.1 8B Instruct (베이스)"

        logger.info(f"[Ollama] 초기화: {self.model_display_name} (모델명: {self.model_name})")

    def generate(
        self,
        prompt: str,
        temperature: float = 0.3,
        top_p: float = 0.9,
        max_tokens: int = 512,
        stop: Optional[List[str]] = None
    ) -> str:
        """
        텍스트 생성 (비스트리밍)

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
            logger.info(f"[Ollama] 생성 시작 - 모델: {self.model_display_name}")
            response = self.client.generate(
                model=self.model_name,
                prompt=prompt,
                options={
                    "temperature": temperature,
                    "top_p": top_p,
                    "num_predict": max_tokens,
                },
                stream=False,
                # KV cache는 Ollama가 자동으로 관리
                keep_alive="5m"  # 모델을 메모리에 5분간 유지 (KV cache 유지)
            )
            result = response['response'].strip()
            logger.info(f"[Ollama] 생성 완료 - 길이: {len(result)} 문자")
            return result

        except Exception as e:
            logger.error(f"[Ollama] 생성 오류: {e}")
            raise

    def generate_stream(
        self,
        prompt: str,
        temperature: float = 0.3,
        top_p: float = 0.9,
        max_tokens: int = 512,
        stop: Optional[List[str]] = None
    ) -> Iterator[str]:
        """
        텍스트 생성 (스트리밍)

        Args:
            prompt: 프롬프트
            temperature: 생성 온도
            top_p: Top-p 샘플링
            max_tokens: 최대 토큰 수
            stop: 중단 토큰 리스트

        Yields:
            생성된 텍스트 청크
        """
        try:
            logger.info(f"[Ollama] 스트리밍 생성 시작 - 모델: {self.model_display_name}")

            stream = self.client.generate(
                model=self.model_name,
                prompt=prompt,
                options={
                    "temperature": temperature,
                    "top_p": top_p,
                    "num_predict": max_tokens,
                },
                stream=True,
                keep_alive="5m"
            )

            total_chars = 0
            for chunk in stream:
                if 'response' in chunk:
                    text = chunk['response']
                    total_chars += len(text)
                    yield text

            logger.info(f"[Ollama] 스트리밍 생성 완료 - 총 길이: {total_chars} 문자")

        except Exception as e:
            logger.error(f"[Ollama] 스트리밍 생성 오류: {e}")
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


def reset_ollama_wrapper():
    """Ollama Wrapper 싱글톤 초기화 (모델 변경 시 사용)"""
    global _ollama_instance
    _ollama_instance = None
    logger.info("[Ollama] 싱글톤 인스턴스 초기화됨")


def get_ollama_wrapper() -> OllamaWrapper:
    """Ollama Wrapper 싱글톤 인스턴스 반환"""
    global _ollama_instance
    if _ollama_instance is None:
        _ollama_instance = OllamaWrapper()

        # 모델 존재 여부 확인 및 안내
        if not _ollama_instance.check_model_exists():
            print(f"[Ollama] 경고: 모델 '{_ollama_instance.model_display_name}'이 설치되지 않았습니다.")
            print(f"[Ollama] Ollama 모델 이름: {_ollama_instance.model_name}")
            print(f"[Ollama] 다음 명령어로 모델을 설치하세요:")
            print(f"  ollama create {_ollama_instance.model_name} -f Modelfile.{_ollama_instance.model_name.replace('-counselor', '')}")
        else:
            print(f"[Ollama] 모델 '{_ollama_instance.model_display_name}' 사용 준비 완료")

    return _ollama_instance
