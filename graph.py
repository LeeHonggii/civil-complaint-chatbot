
from typing import TypedDict, List, Optional, Literal, Dict, Any
from datetime import datetime
from langgraph.graph import StateGraph, END
import uuid


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
    
    # 분류
    query_type: Optional[Literal["qa", "summary", "classification"]]
    
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
# 3. 노드 함수들 (실제 로직은 models/에서 import)
# ============================================================

def classify_query_node(state: GraphState) -> GraphState:
    """쿼리 분류 노드 (placeholder)"""
    from models.query_classifier import classify_query
    return classify_query(state)


def retrieve_examples_node(state: GraphState) -> GraphState:
    """Vector DB 검색 노드 (placeholder)"""
    from models.vector_store import retrieve_examples
    return retrieve_examples(state)


def qa_model_node(state: GraphState) -> GraphState:
    """QA 모델 노드 (placeholder)"""
    from models.qa_model import qa_model
    return qa_model(state)


def summary_model_node(state: GraphState) -> GraphState:
    """요약 모델 노드 (placeholder)"""
    from models.summary_model import summary_model
    return summary_model(state)


def classification_model_node(state: GraphState) -> GraphState:
    """분류 모델 노드 (placeholder)"""
    from models.classification_model import classification_model
    return classification_model(state)


# ============================================================
# 4. 라우팅 함수
# ============================================================

def route_by_query_type(state: GraphState) -> str:
    """query_type에 따라 라우팅"""
    query_type = state.get("query_type", "qa")
    
    if query_type == "summary":
        return "summary_model"
    elif query_type == "classification":
        return "classification_model"
    else:  # qa (기본값)
        return "retrieve_examples"


# ============================================================
# 5. 그래프 구조 정의
# ============================================================

def create_graph() -> StateGraph:
    """LangGraph 생성 및 구조 정의"""
    
    # 그래프 초기화
    workflow = StateGraph(GraphState)
    
    # 노드 추가
    workflow.add_node("classify_query", classify_query_node)
    workflow.add_node("retrieve_examples", retrieve_examples_node)
    workflow.add_node("qa_model", qa_model_node)
    workflow.add_node("summary_model", summary_model_node)
    workflow.add_node("classification_model", classification_model_node)
    
    # 시작점
    workflow.set_entry_point("classify_query")
    
    # 조건부 라우팅
    workflow.add_conditional_edges(
        "classify_query",
        route_by_query_type,
        {
            "retrieve_examples": "retrieve_examples",  # qa 경로
            "summary_model": "summary_model",
            "classification_model": "classification_model"
        }
    )
    
    # QA 경로: 검색 → 모델
    workflow.add_edge("retrieve_examples", "qa_model")
    
    # 종료 엣지
    workflow.add_edge("qa_model", END)
    workflow.add_edge("summary_model", END)
    workflow.add_edge("classification_model", END)
    
    return workflow.compile()


# ============================================================
# 6. 헬퍼 함수
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
        "query_type": None,
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
# 7. 실행 함수
# ============================================================

def run_graph(
    user_query: str,
    session_id: Optional[str] = None,
    window_size: int = 8
) -> Dict[str, Any]:
    """
    그래프 실행 메인 함수
    
    Args:
        user_query: 사용자 질문
        session_id: 세션 ID (None이면 새로 생성)
        window_size: 슬라이딩 윈도우 크기
    
    Returns:
        {
            "response": 모델 응답,
            "session_id": 세션 ID,
            "query_type": 분류 타입,
            "error": 에러 메시지 (있을 경우)
        }
    """
    
    try:
        # 1. 그래프 생성
        graph = create_graph()
        
        # 2. 초기 상태 생성
        state = initialize_state(user_query, session_id, window_size)
        
        # 3. 그래프 실행
        result = graph.invoke(state)
        
        # 4. 최종 상태 저장
        final_state = finalize_state(result)
        
        # 5. 결과 반환
        return {
            "response": final_state.get("model_response", "응답 생성 실패"),
            "session_id": final_state["session_id"],
            "query_type": final_state.get("query_type", "unknown"),
            "error": final_state.get("error")
        }
    
    except Exception as e:
        return {
            "response": f"오류 발생: {str(e)}",
            "session_id": session_id,
            "query_type": None,
            "error": str(e)
        }


# ============================================================
# 8. 유틸리티 함수
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
    print(f"쿼리 타입: {result1['query_type']}\n")
    
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
    
    # 4. 최근 컨텍스트만 확인
    print("\n4️⃣ 최근 컨텍스트 (윈도우 크기: 4)")
    recent = session_store.get_recent_context(result1['session_id'], window_size=4)
    print(format_conversation_history(recent))
