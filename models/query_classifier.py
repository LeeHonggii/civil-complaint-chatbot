"""
Input: GraphState
  - user_query: str
  - recent_context: List[Message]

Process:
  - OpenAI API 호출 (few-shot)
  - 키워드: "요약", "정리" → summary
  - 키워드: "무슨", "유형", "카테고리" → classification
  - 기본값: qa

Output: GraphState
  - query_type: "qa" | "summary" | "classification"
  
"""

import os
from typing import TYPE_CHECKING, Literal
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from graph import GraphState


class QueryType(BaseModel):
    """쿼리 타입 분류 결과"""
    query_type: Literal["qa", "summary", "classification"] = Field(
        description="쿼리의 타입: qa(질의응답), summary(요약), classification(분류)"
    )


def classify_query(state: "GraphState") -> "GraphState":
    """
    사용자 쿼리를 분석하여 타입 분류
    
    Args:
        state: GraphState
            - user_query: 사용자 질문
            - recent_context: 최근 대화 맥락
    
    Returns:
        state: GraphState (query_type 업데이트)
    """
    
    try:
        # LLM 초기화 (structured output)
        llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0,
            api_key=os.getenv("OPENAI_API_KEY")
        )
        structured_llm = llm.with_structured_output(QueryType)
        
        user_query = state["user_query"]
        recent_context = state.get("recent_context", [])
        
        # 맥락 포맷팅
        context_text = ""
        if recent_context:
            context_text = "\n이전 대화:\n"
            for msg in recent_context[-4:]:  # 최근 4턴만
                role = "고객" if msg["role"] == "user" else "상담사"
                context_text += f"{role}: {msg['content']}\n"
        
        # 프롬프트 템플릿
        prompt = ChatPromptTemplate.from_messages([
            ("system", """당신은 상담 쿼리를 분류하는 AI입니다.

사용자의 질문을 분석하여 다음 3가지 중 하나로 분류하세요:

1. "qa" - 질의응답
   - 구체적인 정보를 묻는 질문
   - 예: "카드 분실 시점은?", "수수료는 얼마야?", "어떻게 신청해?"
   
2. "summary" - 요약
   - 대화 내용을 정리/요약 요청
   - 예: "지금까지 뭘 얘기했어?", "정리해줘", "요약해줘"
   
3. "classification" - 분류
   - 상담 유형/카테고리를 묻는 질문
   - 예: "이건 무슨 민원이야?", "어떤 유형이야?", "무슨 카테고리야?"
"""),
            ("user", "{context}\n현재 질문: {query}")
        ])
        
        # 체인 생성 및 실행
        chain = prompt | structured_llm
        result = chain.invoke({
            "context": context_text,
            "query": user_query
        })
        
        # 상태 업데이트
        state["query_type"] = result.query_type
        
        print(f"[Query Classifier] '{user_query}' → {result.query_type}")
        
    except Exception as e:
        print(f"[Query Classifier] Error: {e}")
        # 에러 시 기본값
        state["query_type"] = "qa"
        state["error"] = f"분류 중 오류: {str(e)}"
    
    return state