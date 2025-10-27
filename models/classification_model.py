"""
Classification Model Node
- 상담 내용의 유형/카테고리 분류
- OpenAI API 사용

Input: GraphState
  - user_query: str
  - recent_context: List[Message] (최근 대화)
  - retrieved_examples: List[Dict] (분류 예시)

Process:
  - 대화 맥락 분석
  - 분류 프롬프트 생성
  - OpenAI API 호출

Output: GraphState
  - model_response: str (분류 결과)
    예: "단일 요건 민원", "카드 분실/도난" 등
"""

import os
from typing import TYPE_CHECKING
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

if TYPE_CHECKING:
    from graph import GraphState


def classification_model(state: "GraphState") -> "GraphState":
    """
    상담 유형 분류 모델
    
    Input: GraphState
      - user_query: 사용자 질문
      - recent_context: 최근 대화 맥락
      - retrieved_examples: 분류 예시들
    
    Process:
      - 대화 맥락 분석
      - 분류 프롬프트 생성
      - OpenAI API 호출
    
    Output: GraphState
      - model_response: 분류 결과
    """
    
    try:
        # LLM 초기화
        llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.1,  # 분류는 낮은 temperature
            api_key=os.getenv("OPENAI_API_KEY")
        )
        
        user_query = state["user_query"]
        recent_context = state.get("recent_context", [])
        retrieved_examples = state.get("retrieved_examples", [])
        
        # Few-shot 예시 포맷팅
        examples_text = ""
        if retrieved_examples:
            examples_text = "\n# 분류 예시:\n"
            for i, ex in enumerate(retrieved_examples, 1):
                examples_text += f"\n[예시 {i}]\n"
                examples_text += f"질문: {ex['instruction']}\n"
                examples_text += f"상담 내용: {ex['input'][:200]}...\n"
                examples_text += f"분류 결과: {ex['output']}\n"
        
        # 대화 맥락 포맷팅
        context_text = ""
        if recent_context:
            context_text = "\n# 대화 맥락:\n"
            for msg in recent_context:
                role = "고객" if msg["role"] == "user" else "상담사"
                context_text += f"{role}: {msg['content']}\n"
        else:
            # 맥락이 없으면 현재 질문만
            context_text = f"\n# 대화 내용:\n고객: {user_query}\n"
        
        # 프롬프트 템플릿
        prompt = ChatPromptTemplate.from_messages([
            ("system", """당신은 상담 내용을 분류하는 전문가입니다.

분류 원칙:
1. 대화 내용을 분석하여 적절한 카테고리로 분류
2. 가능한 분류 유형:
   - 상담 요건: "단일 요건 민원", "다수 요건 민원"
   - 상담 주제: "카드 분실/도난", "요금 문의", "서비스 신청", "요금제 변경", "예약 및 일정" 등
   - 상담 내용: "일반 문의", "업무 처리", "고충 민원"
3. 명확하고 간결하게 분류 결과만 제시
4. 예시를 참고하되, 현재 상황에 맞게 판단

{examples}
"""),
            ("user", """{context}

위 상담 내용의 유형을 분류해주세요.

분류 결과:""")
        ])
        
        # 체인 생성 및 실행
        chain = prompt | llm
        response = chain.invoke({
            "examples": examples_text,
            "context": context_text
        })
        
        # 상태 업데이트
        state["model_response"] = response.content.strip()
        
        print(f"[Classification Model] 분류 완료: {state['model_response']}")
        
    except Exception as e:
        print(f"[Classification Model] Error: {e}")
        state["model_response"] = "죄송합니다. 분류 중 오류가 발생했습니다."
        state["error"] = f"분류 모델 오류: {str(e)}"
    
    return state
