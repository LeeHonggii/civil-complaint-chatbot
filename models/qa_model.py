"""
QA Model Node
- Few-shot 예시와 대화 맥락을 활용한 질의응답
- OpenAI API 사용

Input: GraphState
  - user_query: str
  - recent_context: List[Message] (최근 8턴)
  - retrieved_examples: List[Dict] (Top-3 예시)

Process:
  - Few-shot 프롬프트 생성
  - 맥락 정보 포함
  - OpenAI API 호출

Output: GraphState
  - model_response: str
"""

import os
from typing import TYPE_CHECKING
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

if TYPE_CHECKING:
    from graph import GraphState


def qa_model(state: "GraphState") -> "GraphState":
    """
    질의응답 모델
    
    Input: GraphState
      - user_query: 사용자 질문
      - recent_context: 최근 대화 맥락
      - retrieved_examples: Few-shot 예시들
    
    Process:
      - Few-shot 프롬프트 생성
      - OpenAI API 호출
    
    Output: GraphState
      - model_response: 답변
    """
    
    try:
        # LLM 초기화
        llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.3,
            api_key=os.getenv("OPENAI_API_KEY")
        )
        
        user_query = state["user_query"]
        recent_context = state.get("recent_context", [])
        retrieved_examples = state.get("retrieved_examples", [])
        
        # Few-shot 예시 포맷팅 (1개만)
        examples_text = ""
        if retrieved_examples and len(retrieved_examples) > 0:
            ex = retrieved_examples[0]  # 첫 번째 예시만
            examples_text = "\n# 유사한 상담 예시:\n\n[예시]\n"
            
            # 상담 대화 포함
            if ex.get('conversation'):
                examples_text += f"상담 대화:\n{ex['conversation']}\n\n"
            
            examples_text += f"질문: {ex['instruction']}\n"
            examples_text += f"답변: {ex['output']}\n"
        
        # 대화 맥락 포맷팅
        context_text = ""
        if recent_context:
            context_text = "\n# 이전 대화 맥락:\n"
            for msg in recent_context:
                role = "고객" if msg["role"] == "user" else "상담사"
                context_text += f"{role}: {msg['content']}\n"
        
        # 프롬프트 템플릿
        prompt = ChatPromptTemplate.from_messages([
            ("system", """당신은 전문 상담사입니다.

고객의 질문에 정확하고 친절하게 답변하세요.

답변 원칙:
1. 간결하고 명확하게 답변 (불필요한 설명 자제)
2. 이전 대화 맥락을 고려하여 답변
3. 유사 예시를 참고하되, 현재 상황에 맞게 답변
4. 정보가 불확실하면 "확인이 필요합니다" 등으로 안내
5. 자연스럽고 친근한 말투 사용

{examples}
"""),
            ("user", "{context}\n# 현재 질문:\n고객: {query}\n\n상담사:")
        ])
        
        # 체인 생성 및 실행
        chain = prompt | llm
        response = chain.invoke({
            "examples": examples_text,
            "context": context_text,
            "query": user_query
        })
        
        # 상태 업데이트
        state["model_response"] = response.content.strip()
        
        print(f"[QA Model] 답변 생성 완료 ({len(response.content)}자)")
        
    except Exception as e:
        print(f"[QA Model] Error: {e}")
        state["model_response"] = "죄송합니다. 답변 생성 중 오류가 발생했습니다."
        state["error"] = f"QA 모델 오류: {str(e)}"
    
    return state