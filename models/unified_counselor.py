"""
Unified Counselor Node (멀티 환경 지원)
- Mac (Ollama) / GPU (vLLM) 자동 감지
- KV Cache / Prefix Caching으로 멀티턴 최적화
- Few-shot 예시 활용
- 대화 맥락 기반 답변

Input: GraphState
  - user_query: str
  - recent_context: List[Message] (최근 8턴)
  - retrieved_examples: List[Dict] (1-2개)

Process:
  - 환경 자동 감지
  - Few-shot 예시 포맷팅
  - 대화 맥락 포함 (KV cache 최적화)
  - LLM 호출

Output: GraphState
  - model_response: str
"""

from typing import TYPE_CHECKING, List, Dict, Iterator
from .env_detector import detect_environment
import logging

# 로거 설정
logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from graph import GraphState, Message


# 전역 모델 인스턴스 (싱글톤)
_model_wrapper = None


def reset_model_wrapper():
    """모델 wrapper 싱글톤 초기화 (모델 변경 시 사용)"""
    global _model_wrapper
    _model_wrapper = None
    logger.info("[Unified Counselor] 모델 wrapper 초기화됨")


def get_model_wrapper():
    """환경에 맞는 모델 wrapper 반환 (싱글톤)"""
    global _model_wrapper

    if _model_wrapper is None:
        env = detect_environment()

        if env == "gpu":
            from .vllm_wrapper import get_vllm_wrapper
            _model_wrapper = get_vllm_wrapper()
            print("[Unified Counselor] vLLM 모드 (Prefix Caching 활성화)")

        else:  # mac
            from .ollama_wrapper import get_ollama_wrapper
            _model_wrapper = get_ollama_wrapper()
            print("[Unified Counselor] Ollama 모드 (KV Cache 자동 관리)")

    return _model_wrapper


def unified_counselor(state: "GraphState") -> "GraphState":
    """
    질의응답 모델 (멀티 환경 지원)
    - Mac → Ollama (KV cache)
    - GPU → vLLM (Prefix caching)
    - Few-shot 예시 활용
    - 대화 맥락 기반 답변
    """

    try:
        # 모델 wrapper 가져오기
        model = get_model_wrapper()

        user_query = state["user_query"]
        recent_context = state.get("recent_context", [])
        retrieved_examples = state.get("retrieved_examples", [])

        # 프롬프트 생성 (KV cache 최적화를 위해 구조화)
        prompt_text = build_qa_prompt(
            user_query=user_query,
            recent_context=recent_context,
            retrieved_examples=retrieved_examples
        )

        # 모델 호출 (KV cache / Prefix caching 자동 적용)
        response_text = model.generate(
            prompt=prompt_text,
            temperature=0.3,
            top_p=0.9,
            max_tokens=512,
            stop=["고객:", "\n\n\n"]
        )

        # 상태 업데이트
        state["model_response"] = response_text

        print(f"[Unified Counselor] 답변 생성 완료 ({len(response_text)}자)")

    except Exception as e:
        print(f"[Unified Counselor] Error: {e}")
        import traceback
        traceback.print_exc()
        state["model_response"] = "죄송합니다. 답변 생성 중 오류가 발생했습니다."
        state["error"] = f"통합 모델 오류: {str(e)}"

    return state


def build_qa_prompt(
    user_query: str,
    recent_context: List["Message"],
    retrieved_examples: List[Dict]
) -> str:
    """
    QA 프롬프트 생성 (KV Cache 최적화 구조)

    구조:
    1. 시스템 프롬프트 (고정) ← KV cache에서 재사용
    2. Few-shot 예시 (고정) ← KV cache에서 재사용
    3. 대화 맥락 (변동) ← 새로 계산
    4. 현재 질문 (변동) ← 새로 계산
    """

    # 1. 시스템 프롬프트 (고정 - KV cache 재사용됨)
    system_prompt = """당신은 전문 상담사입니다.

## 답변 원칙
1. 구체적이고 명확한 답변 제공
2. 이전 대화 맥락을 고려하여 자연스럽게 답변
3. 유사 예시를 참고하되, 현재 상황에 맞게 답변
4. 정보가 불확실하면 "확인이 필요합니다" 안내
5. 친근하고 자연스러운 말투
6. 간결하게 답변 (불필요한 설명 자제)
"""

    # 2. Few-shot 예시 포맷팅 (고정 - KV cache 재사용됨)
    examples_text = ""
    if retrieved_examples:
        examples_text = "\n# 📚 유사한 상담 예시:\n"
        for i, ex in enumerate(retrieved_examples, 1):
            examples_text += f"\n[예시 {i}]\n"

            # 메타데이터 정보 추가
            if ex.get('domain') or ex.get('task_category') or ex.get('source'):
                examples_text += "메타데이터:\n"
                if ex.get('domain'):
                    examples_text += f"  - 도메인: {ex['domain']}\n"
                if ex.get('task_category'):
                    examples_text += f"  - 질문 유형: {ex['task_category']}\n"
                if ex.get('source'):
                    examples_text += f"  - 출처: {ex['source']}\n"
                examples_text += "\n"

            # 상담 대화 포함
            if ex.get('conversation'):
                conv_preview = ex['conversation'][:300]
                examples_text += f"상담 대화:\n{conv_preview}...\n\n"

            examples_text += f"질문: {ex['instruction']}\n"
            examples_text += f"답변: {ex['output']}\n"

    # 3. 대화 맥락 포맷팅 (변동 - 매번 새로 계산)
    context_text = "\n## 💬 이전 대화 맥락:\n"

    if recent_context:
        for msg in recent_context:
            role = "고객" if msg["role"] == "user" else "상담사"
            context_text += f"{role}: {msg['content']}\n"
    else:
        context_text += "(첫 대화입니다)\n"

    # 4. 현재 질문 (변동 - 매번 새로 계산)
    query_text = f"\n## ❓ 현재 질문:\n고객: {user_query}\n\n상담사: "

    return system_prompt + examples_text + context_text + query_text


def unified_counselor_stream(
    user_query: str,
    recent_context: List["Message"],
    retrieved_examples: List[Dict]
) -> Iterator[str]:
    """
    질의응답 모델 (스트리밍 버전)
    - Mac → Ollama (KV cache)
    - GPU → vLLM (Prefix caching)
    - Few-shot 예시 활용
    - 대화 맥락 기반 답변

    Yields:
        생성된 텍스트 청크
    """

    try:
        # 모델 wrapper 가져오기
        model = get_model_wrapper()

        # 프롬프트 생성
        prompt_text = build_qa_prompt(
            user_query=user_query,
            recent_context=recent_context,
            retrieved_examples=retrieved_examples
        )

        logger.info(f"[Unified Counselor Stream] 스트리밍 답변 생성 시작")

        # 스트리밍이 지원되는지 확인
        if hasattr(model, 'generate_stream'):
            # 스트리밍 모드로 생성
            for chunk in model.generate_stream(
                prompt=prompt_text,
                temperature=0.3,
                top_p=0.9,
                max_tokens=512,
                stop=["고객:", "\n\n\n"]
            ):
                yield chunk
        else:
            # 스트리밍 미지원 시 일반 생성 후 한 번에 반환
            logger.warning(f"[Unified Counselor Stream] 스트리밍 미지원, 일반 생성 모드 사용")
            response = model.generate(
                prompt=prompt_text,
                temperature=0.3,
                top_p=0.9,
                max_tokens=512,
                stop=["고객:", "\n\n\n"]
            )
            yield response

        logger.info(f"[Unified Counselor Stream] 스트리밍 답변 생성 완료")

    except Exception as e:
        logger.error(f"[Unified Counselor Stream] Error: {e}")
        import traceback
        traceback.print_exc()
        yield "죄송합니다. 답변 생성 중 오류가 발생했습니다."
