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

# Vector Store 자동 초기화
if "vector_store_initialized" not in st.session_state:
    st.session_state.vector_store_initialized = False

if not st.session_state.vector_store_initialized:
    with st.spinner("🔧 Vector Store 초기화 중..."):
        try:
            initialize_vector_store()
            st.session_state.vector_store_initialized = True
            st.success("✅ Vector Store 초기화 완료!")
        except Exception as e:
            st.error(f"❌ Vector Store 초기화 실패: {e}")
            st.stop()

# 사이드바
with st.sidebar:
    st.header("⚙️ 설정")
    
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
        with st.spinner("생각 중..."):
            try:
                # LangGraph 실행
                result = run_graph(
                    user_query=prompt,
                    session_id=st.session_state.session_id,
                    window_size=8
                )
                
                # 세션 ID 저장 (첫 응답시)
                if st.session_state.session_id is None:
                    st.session_state.session_id = result["session_id"]
                
                # 응답 표시
                response = result["response"]
                st.markdown(response)
                
                # 에러 표시 (있을 경우)
                if result.get("error"):
                    st.error(f"⚠️ {result['error']}")
                
                # 쿼리 타입 표시 (디버그용)
                with st.expander("🔍 상세 정보"):
                    st.caption(f"쿼리 타입: {result.get('query_type', 'unknown')}")
                
            except Exception as e:
                response = f"죄송합니다. 오류가 발생했습니다: {str(e)}"
                st.error(response)
        
        # 어시스턴트 메시지 추가
        st.session_state.messages.append({"role": "assistant", "content": response})

# 하단 정보
st.divider()
with st.expander("ℹ️ 사용 방법"):
    st.markdown("""
    ### 💡 팁
    
    **질의응답 (QA)**
    - "카드 분실 시점은 언제인가요?"
    - "수수료는 얼마인가요?"
    
    **요약**
    - "지금까지 뭘 얘기했어?"
    - "대화 내용 정리해줘"
    
    **분류**
    - "이건 무슨 민원이야?"
    - "어떤 유형이야?"
    
    ### 🔧 기능
    - ✅ 멀티턴 대화 지원 (최근 8턴 맥락 유지)
    - ✅ Few-shot 예시 활용
    - ✅ 세션별 대화 저장
    """)

# 푸터
st.caption("Powered by LangGraph + OpenAI + ChromaDB")