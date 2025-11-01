from typing import TypedDict, List, Optional, Literal, Dict, Any
from datetime import datetime
from langgraph.graph import StateGraph, END
import uuid

# langgraph 0.0.20에서는 START가 없으므로 None을 시작 노드로 사용
START = None


# ============================================================
# 1. 상태 정의
# ============================================================

class Message(TypedDict):
    """개별 메시지 구조"""
    role: Literal["user", "assistant"]
    content: str
    timestamp: str  # ISO format string


class GraphState(TypedDict):
    """LangGraph 전체 상태"""
    
    # 입력
    user_query: str
    conversation_history: List[Message]  # 전체 대화 저장
    
    # ✨ 메타데이터 (새로 추가)
    metadata: Optional[Dict[str, Any]]
    
    # 검색 (Few-shot)
    retrieved_examples: Optional[List[Dict[str, Any]]]
    
    # 컨텍스트 (슬라이딩 윈도우)
    recent_context: List[Message]  # 최근 8-10턴만
    
    # 출력
    model_response: Optional[str]
    
    # 메타
    session_id: str
    error: Optional[str]


# ============================================================
# 2. 세션 저장소 (메모리 기반)
# ============================================================

class SessionStore:
    """세션별 대화 히스토리 저장"""
    
    def __init__(self):
        self.sessions: Dict[str, Dict[str, Any]] = {}
    
    def create_session(self) -> str:
        """새 세션 생성"""
        session_id = str(uuid.uuid4())
        self.sessions[session_id] = {
            "conversation_history": [],
            "created_at": datetime.now().isoformat(),
            "last_updated": datetime.now().isoformat()
        }
        return session_id
    
    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """세션 조회"""
        return self.sessions.get(session_id)
    
    def update_session(self, session_id: str, conversation_history: List[Message]):
        """세션 업데이트"""
        if session_id in self.sessions:
            self.sessions[session_id]["conversation_history"] = conversation_history
            self.sessions[session_id]["last_updated"] = datetime.now().isoformat()
    
    def add_message(self, session_id: str, role: str, content: str):
        """메시지 추가"""
        if session_id in self.sessions:
            message: Message = {
                "role": role,
                "content": content,
                "timestamp": datetime.now().isoformat()
            }
            self.sessions[session_id]["conversation_history"].append(message)
            self.sessions[session_id]["last_updated"] = datetime.now().isoformat()
    
    def get_conversation_history(self, session_id: str) -> List[Message]:
        """대화 히스토리 조회"""
        session = self.get_session(session_id)
        return session["conversation_history"] if session else []
    
    def clear_session(self, session_id: str):
        """세션 삭제"""
        if session_id in self.sessions:
            del self.sessions[session_id]
    
    def get_recent_context(self, session_id: str, window_size: int = 8) -> List[Message]:
        """슬라이딩 윈도우 - 최근 N턴만 반환"""
        history = self.get_conversation_history(session_id)
        return history[-window_size:] if len(history) > window_size else history


# 전역 세션 저장소
session_store = SessionStore()


# ============================================================
# 3. 노드 함수들
# ============================================================

def extract_metadata_node(state: GraphState) -> GraphState:
    """메타데이터 추출 노드"""
    from models.metadata_extractor import extract_metadata
    return extract_metadata(state)


def retrieve_examples_node(state: GraphState) -> GraphState:
    """Vector DB 검색 노드"""
    from models.vector_store import retrieve_examples
    return retrieve_examples(state)


def route_by_environment(state: GraphState) -> str:
    """환경에 따라 라우팅"""
    from models.env_detector import detect_environment
    env = detect_environment()

    if env == "gpu":
        print("[라우터] GPU 환경 감지 → vLLM 노드로 라우팅")
        return "vllm"
    else:
        print("[라우터] Mac/CPU 환경 감지 → Ollama 노드로 라우팅")
        return "ollama"


def vllm_counselor_node(state: GraphState) -> GraphState:
    """vLLM 상담 모델 노드 (GPU)"""
    from models.vllm_wrapper import get_vllm_wrapper

    print("[vLLM Counselor] 답변 생성 중 (Prefix Caching)...")

    try:
        model = get_vllm_wrapper()

        user_query = state["user_query"]
        recent_context = state.get("recent_context", [])
        retrieved_examples = state.get("retrieved_examples", [])

        # 프롬프트 생성
        from models.unified_counselor import build_qa_prompt
        prompt_text = build_qa_prompt(
            user_query=user_query,
            recent_context=recent_context,
            retrieved_examples=retrieved_examples
        )

        # vLLM 생성
        response_text = model.generate(
            prompt=prompt_text,
            temperature=0.3,
            top_p=0.9,
            max_tokens=512,
            stop=["고객:", "\n\n\n"]
        )

        state["model_response"] = response_text
        print(f"[vLLM Counselor] 답변 생성 완료 ({len(response_text)}자)")

    except Exception as e:
        print(f"[vLLM Counselor] Error: {e}")
        state["model_response"] = "죄송합니다. 답변 생성 중 오류가 발생했습니다."
        state["error"] = f"vLLM 오류: {str(e)}"

    return state


def ollama_counselor_node(state: GraphState) -> GraphState:
    """Ollama 상담 모델 노드 (Mac/CPU)"""
    from models.ollama_wrapper import get_ollama_wrapper

    print("[Ollama Counselor] 답변 생성 중 (KV Cache)...")

    try:
        model = get_ollama_wrapper()

        user_query = state["user_query"]
        recent_context = state.get("recent_context", [])
        retrieved_examples = state.get("retrieved_examples", [])

        # 프롬프트 생성
        from models.unified_counselor import build_qa_prompt
        prompt_text = build_qa_prompt(
            user_query=user_query,
            recent_context=recent_context,
            retrieved_examples=retrieved_examples
        )

        # Ollama 생성
        response_text = model.generate(
            prompt=prompt_text,
            temperature=0.3,
            top_p=0.9,
            max_tokens=512,
            stop=["고객:", "\n\n\n"]
        )

        state["model_response"] = response_text
        print(f"[Ollama Counselor] 답변 생성 완료 ({len(response_text)}자)")

    except Exception as e:
        print(f"[Ollama Counselor] Error: {e}")
        state["model_response"] = "죄송합니다. 답변 생성 중 오류가 발생했습니다."
        state["error"] = f"Ollama 오류: {str(e)}"

    return state


# ============================================================
# 4. 그래프 구조 정의 (단순화)
# ============================================================

def create_graph() -> StateGraph:
    """LangGraph 생성 및 구조 정의 (환경별 라우팅)"""

    # 그래프 초기화
    workflow = StateGraph(GraphState)

    # 노드 추가 (메타데이터 추출, 예시 검색, 환경별 상담사)
    workflow.add_node("extract_metadata", extract_metadata_node)
    workflow.add_node("retrieve_examples", retrieve_examples_node)
    workflow.add_node("vllm_counselor", vllm_counselor_node)
    workflow.add_node("ollama_counselor", ollama_counselor_node)

    # 엣지 설정
    workflow.set_entry_point("extract_metadata")
    workflow.add_edge("extract_metadata", "retrieve_examples")

    # ✨ 조건부 라우팅: 환경에 따라 vLLM 또는 Ollama로 분기
    workflow.add_conditional_edges(
        "retrieve_examples",
        route_by_environment,
        {
            "vllm": "vllm_counselor",
            "ollama": "ollama_counselor"
        }
    )

    # 양쪽 경로 모두 END로
    workflow.add_edge("vllm_counselor", END)
    workflow.add_edge("ollama_counselor", END)

    return workflow.compile()


# ============================================================
# 5. 헬퍼 함수
# ============================================================

def initialize_state(
    user_query: str,
    session_id: Optional[str] = None,
    window_size: int = 8
) -> GraphState:
    """초기 상태 생성"""
    
    # 세션 ID 처리
    if session_id is None:
        session_id = session_store.create_session()
    
    # 대화 히스토리 로드
    conversation_history = session_store.get_conversation_history(session_id)
    recent_context = session_store.get_recent_context(session_id, window_size)
    
    # 현재 사용자 메시지 추가
    session_store.add_message(session_id, "user", user_query)
    
    # 상태 생성
    state: GraphState = {
        "user_query": user_query,
        "conversation_history": conversation_history,
        "metadata": None,  # ✨ 추가
        "retrieved_examples": None,
        "recent_context": recent_context,
        "model_response": None,
        "session_id": session_id,
        "error": None
    }
    
    return state


def finalize_state(state: GraphState):
    """최종 상태 저장"""
    
    # 어시스턴트 응답 저장
    if state["model_response"]:
        session_store.add_message(
            state["session_id"],
            "assistant",
            state["model_response"]
        )
    
    # 전체 대화 히스토리 업데이트
    conversation_history = session_store.get_conversation_history(state["session_id"])
    state["conversation_history"] = conversation_history
    
    return state


# ============================================================
# 6. 실행 함수
# ============================================================

def run_graph(
    user_query: str,
    session_id: Optional[str] = None,
    window_size: int = 8,
    stream: bool = False
) -> Dict[str, Any]:
    """
    그래프 실행 메인 함수

    Args:
        user_query: 사용자 질문
        session_id: 세션 ID (None이면 새로 생성)
        window_size: 슬라이딩 윈도우 크기
        stream: 스트리밍 모드 여부

    Returns:
        {
            "response": 모델 응답 (stream=False인 경우),
            "response_stream": 스트리밍 iterator (stream=True인 경우),
            "session_id": 세션 ID,
            "metadata": 메타데이터,
            "retrieved_examples": 검색된 예시들,
            "error": 에러 메시지 (있을 경우)
        }
    """

    try:
        # 1. 그래프 생성
        graph = create_graph()

        # 2. 초기 상태 생성
        state = initialize_state(user_query, session_id, window_size)

        # 스트리밍 모드인 경우
        if stream:
            # 메타데이터 추출 및 검색만 수행 (통합 모델 실행 전까지)
            from models.metadata_extractor import extract_metadata
            from models.vector_store import retrieve_examples

            # 메타데이터 추출
            state = extract_metadata(state)

            # Vector Store 검색
            state = retrieve_examples(state)

            # 스트리밍 응답 생성
            from models.unified_counselor import unified_counselor_stream

            response_stream = unified_counselor_stream(
                user_query=user_query,
                recent_context=state.get("recent_context", []),
                retrieved_examples=state.get("retrieved_examples", [])
            )

            # 결과 반환 (스트리밍 iterator 포함)
            return {
                "response_stream": response_stream,
                "session_id": state["session_id"],
                "metadata": state.get("metadata"),
                "retrieved_examples": state.get("retrieved_examples"),
                "error": state.get("error")
            }

        # 일반 모드
        else:
            # 3. 그래프 실행
            result = graph.invoke(state)

            # 4. 최종 상태 저장
            final_state = finalize_state(result)

            # 5. 결과 반환
            return {
                "response": final_state.get("model_response", "응답 생성 실패"),
                "session_id": final_state["session_id"],
                "metadata": final_state.get("metadata"),
                "retrieved_examples": final_state.get("retrieved_examples"),
                "error": final_state.get("error")
            }

    except Exception as e:
        return {
            "response": f"오류 발생: {str(e)}",
            "session_id": session_id,
            "metadata": None,
            "retrieved_examples": None,
            "error": str(e)
        }


# ============================================================
# 7. 유틸리티 함수
# ============================================================

def format_conversation_history(history: List[Message]) -> str:
    """대화 히스토리를 텍스트로 포맷팅"""
    formatted = []
    for msg in history:
        role = "고객" if msg["role"] == "user" else "상담사"
        formatted.append(f"{role}: {msg['content']}")
    return "\n".join(formatted)


def get_session_info(session_id: str) -> Optional[Dict[str, Any]]:
    """세션 정보 조회"""
    return session_store.get_session(session_id)


def clear_all_sessions():
    """모든 세션 초기화"""
    session_store.sessions.clear()


# ============================================================
# 테스트 코드
# ============================================================

if __name__ == "__main__":
    print("=== LangGraph 구조 테스트 ===\n")
    
    # 1. 새 세션 시작
    print("1️⃣ 첫 번째 질문")
    result1 = run_graph("카드 분실 신고하려고요")
    print(f"응답: {result1['response']}")
    print(f"세션 ID: {result1['session_id']}")
    print(f"메타데이터: {result1['metadata']}\n")
    
    # 2. 같은 세션에서 후속 질문
    print("2️⃣ 후속 질문 (같은 세션)")
    result2 = run_graph(
        "어제 분실했어요",
        session_id=result1['session_id']
    )
    print(f"응답: {result2['response']}")
    print(f"세션 ID: {result2['session_id']}\n")
    
    # 3. 대화 히스토리 확인
    print("3️⃣ 대화 히스토리")
    history = session_store.get_conversation_history(result1['session_id'])
    print(format_conversation_history(history))
    print(f"\n총 {len(history)}턴의 대화")