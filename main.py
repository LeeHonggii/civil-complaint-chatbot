"""
상담 챗봇 - Streamlit UI
- 멀티턴 대화 지원
- 세션 관리
- LangGraph 파이프라인 통합
"""

import streamlit as st
import os
from dotenv import load_dotenv
from graph import run_graph, session_store, format_conversation_history
from models import initialize_vector_store

# 환경 변수 로드
load_dotenv()

# 페이지 설정
st.set_page_config(
    page_title="상담 챗봇",
    page_icon="🤖",
    layout="centered"
)

# 타이틀
st.title("🤖 상담 챗봇")
st.caption("고객 상담을 도와드립니다")

# Vector Store 초기화 (캐싱)
@st.cache_resource
def get_vector_store():
    """Vector Store 초기화 (Streamlit 캐싱)"""
    initialize_vector_store()
    return True

# Vector Store 자동 초기화
with st.spinner("🔧 Vector Store 초기화 중..."):
    try:
        get_vector_store()
    except Exception as e:
        st.error(f"❌ Vector Store 초기화 실패: 데이터베이스를 불러올 수 없습니다.")
        st.stop()

# 사이드바
with st.sidebar:
    st.header("⚙️ 설정")

    # 모델 선택
    st.subheader("🤖 모델 선택")

    # 모델 이름 매핑
    model_display_names = {
        "llama": "Llama 3.1 8B Instruct",
        "mistral": "Mistral 7B Instruct v0.2",
        "gemma": "Gemma 2 9B Instruct",
        "bccard": "BCCard Llama 3 8B"
    }

    # 현재 환경에서 사용 가능한 모델 확인
    model_choice = st.selectbox(
        "추론 모델",
        options=["llama", "mistral", "gemma", "bccard"],
        format_func=lambda x: model_display_names.get(x, x),
        index=["llama", "mistral", "gemma", "bccard"].index(os.getenv("MODEL_NAME", "llama")),
        help="대화 생성에 사용할 모델을 선택하세요"
    )

    # 파인튜닝 모델 사용 체크박스
    use_finetuned = st.checkbox(
        "파인튜닝 모델 사용",
        value=os.getenv("USE_FINETUNED_MODEL", "false").lower() == "true",
        help="체크하면 파인튜닝된 모델을 사용하고, 체크 해제하면 베이스 모델을 사용합니다"
    )

    # 모델 변경 시 환경 변수 업데이트
    if "current_model" not in st.session_state:
        st.session_state.current_model = os.getenv("MODEL_NAME", "llama")

    if "use_finetuned" not in st.session_state:
        st.session_state.use_finetuned = os.getenv("USE_FINETUNED_MODEL", "false").lower() == "true"

    # 모델 또는 파인튜닝 설정 변경 감지
    model_changed = model_choice != st.session_state.current_model
    finetuned_changed = use_finetuned != st.session_state.use_finetuned

    if model_changed or finetuned_changed:
        os.environ["MODEL_NAME"] = model_choice
        os.environ["USE_FINETUNED_MODEL"] = "true" if use_finetuned else "false"
        st.session_state.current_model = model_choice
        st.session_state.use_finetuned = use_finetuned

        # 싱글톤 모델 인스턴스 초기화
        from models.unified_counselor import reset_model_wrapper
        from models.ollama_wrapper import reset_ollama_wrapper
        try:
            from models.vllm_wrapper import reset_vllm_wrapper
            reset_vllm_wrapper()
        except:
            pass  # GPU 환경이 아니면 vllm 모듈이 없을 수 있음
        reset_model_wrapper()
        reset_ollama_wrapper()

        model_type = "파인튜닝" if use_finetuned else "베이스"
        display_name = model_display_names.get(model_choice, model_choice)
        st.info(f"✨ {display_name} {model_type} 모델로 변경되었습니다. 다음 대화부터 적용됩니다.")

    # 현재 선택된 모델 표시
    model_type = "파인튜닝" if use_finetuned else "베이스"
    display_name = model_display_names.get(model_choice, model_choice)
    st.caption(f"🎯 현재 모델: {display_name} ({model_type})")

    st.divider()

    # 초기화 상태 표시
    st.success("✅ Vector Store 준비됨")

    st.divider()

    # 새 대화 시작 버튼
    if st.button("🔄 새 대화 시작", use_container_width=True):
        if "session_id" in st.session_state:
            # 기존 세션 클리어
            session_store.clear_session(st.session_state.session_id)
        # 세션 상태 초기화
        st.session_state.messages = []
        st.session_state.session_id = None
        st.rerun()

    st.divider()

    # 세션 정보
    if "session_id" in st.session_state and st.session_state.session_id:
        st.caption(f"📝 세션 ID: {st.session_state.session_id[:8]}...")

        # 대화 턴 수
        history = session_store.get_conversation_history(st.session_state.session_id)
        st.caption(f"💬 대화 턴: {len(history)//2}턴")

# 메인 화면
# 세션 상태 초기화
if "messages" not in st.session_state:
    st.session_state.messages = []

if "session_id" not in st.session_state:
    st.session_state.session_id = None

# 대화 히스토리 표시
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 사용자 입력
if prompt := st.chat_input("메시지를 입력하세요..."):
    
    # 사용자 메시지 추가
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # 사용자 메시지 표시
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # 어시스턴트 응답 생성
    with st.chat_message("assistant"):
        response = ""  # 응답 변수 초기화
        response_placeholder = st.empty()

        try:
            # 로딩 메시지 표시
            response_placeholder.markdown("🤖 생성중...")

            # LangGraph 실행 (스트리밍 모드)
            result = run_graph(
                user_query=prompt,
                session_id=st.session_state.session_id,
                window_size=8,
                stream=True
            )

            # 세션 ID 저장 (첫 응답시)
            if st.session_state.session_id is None:
                st.session_state.session_id = result["session_id"]

            # 스트리밍 응답 표시
            full_response = ""

            # 스트리밍으로 응답 받기
            if "response_stream" in result and result["response_stream"]:
                for chunk in result["response_stream"]:
                    full_response += chunk
                    response_placeholder.markdown(full_response + "▌")
                response_placeholder.markdown(full_response)
                response = full_response

                # 스트리밍 완료 후 대화 히스토리에 저장
                from graph import session_store
                session_store.add_message(result["session_id"], "user", prompt)
                session_store.add_message(result["session_id"], "assistant", response)
            else:
                # 스트리밍 실패 시 일반 응답 사용
                response = result.get("response", "응답 생성 실패")
                response_placeholder.markdown(response)

                # 에러 표시 (있을 경우)
                if result.get("error"):
                    st.error(f"⚠️ {result['error']}")

        except Exception as e:
            response = f"죄송합니다. 오류가 발생했습니다: {str(e)}"
            response_placeholder.markdown("")
            st.error(response)

        # 어시스턴트 메시지 추가
        st.session_state.messages.append({"role": "assistant", "content": response})

# 하단 정보
st.divider()

# 푸터
model_name = os.getenv("MODEL_NAME", "llama")
use_finetuned = os.getenv("USE_FINETUNED_MODEL", "false").lower() == "true"
model_display_names = {
    "llama": "Llama 3.1 8B",
    "mistral": "Mistral 7B",
    "gemma": "Gemma 2 9B",
    "bccard": "BCCard Llama 3 8B"
}
model_display = model_display_names.get(model_name, "Unknown")
model_type = " (파인튜닝)" if use_finetuned else " (베이스)"
st.caption(f"Powered by LangGraph + {model_display}{model_type} + Pinecone/ChromaDB")