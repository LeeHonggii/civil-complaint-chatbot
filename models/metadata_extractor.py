"""
Metadata Extractor Node
- 사용자 질문 분석하여 검색 도메인 결정
- 대화 맥락 정보 추출

Input: GraphState
  - user_query: str
  - recent_context: List[Message]
  - conversation_history: List[Message]

Process:
  - conversation_turns 계산
  - domain LLM 추론 (금융/통신/여행 중 선택)

Output: GraphState
  - metadata: Dict {
      "domain": str,
      "conversation_turns": int
    }
"""

import os
from typing import TYPE_CHECKING, Literal
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from graph import GraphState


# 회사명 → 도메인 매핑 (vector_store에서 사용)
DOMAIN_MAPPING = {
    "하나카드": "금융",
    "엘지유플러스": "통신",
    "액티벤처": "여행"
}


class DomainClassification(BaseModel):
    """도메인 분류 결과"""
    domain: Literal["금융", "통신", "여행", "기타"] = Field(
        description="질문이 속한 도메인: 금융(카드/계좌/결제), 통신(요금제/데이터/통화), 여행(예약/숙소/일정), 기타"
    )


def extract_metadata(state: "GraphState") -> "GraphState":
    """
    메타데이터 자동 추출
    
    Args:
        state: GraphState
            - user_query: 사용자 질문
            - recent_context: 최근 대화
            - conversation_history: 전체 대화
    
    Returns:
        state: GraphState (metadata 업데이트)
    """
    
    try:
        # 1. conversation_turns 계산
        conversation_history = state.get("conversation_history", [])
        conversation_turns = len(conversation_history) // 2
        
        # 2. domain LLM 추론
        llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0,
            api_key=os.getenv("OPENAI_API_KEY")
        )
        structured_llm = llm.with_structured_output(DomainClassification)
        
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
            ("system", """당신은 상담 질문을 도메인별로 분류하는 AI입니다.

사용자의 질문을 분석하여 다음 4가지 도메인 중 하나로 분류하세요:

## 1. 금융
- 카드 관련: 분실, 발급, 정지, 해제, 재발급
- 결제/포인트: 결제 수단, 포인트, 마일리지
- 계좌: 계좌 개설, 이체, 조회
- 예시: "카드 분실했어요", "포인트 조회", "결제 취소"

## 2. 통신
- 요금제: 요금제 변경, 데이터 요금
- 서비스: 통화, 문자, 데이터, 로밍
- 단말기: 기기 변경, AS
- 예시: "요금제 변경하고 싶어요", "데이터 추가", "통화 품질"

## 3. 여행
- 예약: 숙소, 항공편, 렌터카
- 일정: 여행 계획, 일정 변경
- 서비스: 픽업, 체크아웃, 조식
- 예시: "호텔 예약", "공항 픽업", "체크아웃 시간"

## 4. 기타
- 위 3가지에 속하지 않는 경우
- 불명확하거나 일반적인 질문

대화 맥락을 고려하여 가장 적절한 도메인을 선택하세요.
"""),
            ("user", "{context}\n현재 질문: {query}")
        ])
        
        # 체인 생성 및 실행
        chain = prompt | structured_llm
        result = chain.invoke({
            "context": context_text,
            "query": user_query
        })
        
        domain = result.domain
        
        # 3. 메타데이터 구성
        state["metadata"] = {
            "domain": domain,
            "conversation_turns": conversation_turns
        }
        
        print(f"[Metadata Extractor] 추출 완료:")
        print(f"  - domain: {domain}")
        print(f"  - conversation_turns: {conversation_turns}")
        
    except Exception as e:
        print(f"[Metadata Extractor] Error: {e}")
        # 에러 시 기본값 (모든 도메인 검색)
        state["metadata"] = {
            "domain": "기타",
            "conversation_turns": 0
        }
        state["error"] = f"메타데이터 추출 오류: {str(e)}"
    
    return state
