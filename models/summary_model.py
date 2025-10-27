"""
Summary Model Node
- 전체 대화 내용을 요약
- OpenAI API 사용

Input: GraphState
  - conversation_history: List[Message] (전체 대화)
  - retrieved_examples: List[Dict] (요약 예시, 옵션)

Process:
  - 전체 대화를 텍스트로 변환
  - 요약 프롬프트 생성
  - OpenAI API 호출

Output: GraphState
  - model_response: str (요약 결과)
"""

import os
from typing import TYPE_CHECKING
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

if TYPE_CHECKING:
    from graph import GraphState


def summary_model(state: "GraphState") -> "GraphState":
    """
    대화 요약 모델
    
    Input: GraphState
      - conversation_history: 전체 대화 내역
      - retrieved_examples: 요약 예시들 (옵션)
    
    Process:
      - 전체 대화를 텍스트로 포맷팅
      - 요약 프롬프트 생성
      - OpenAI API 호출
    
    Output: GraphState
      - model_response: 요약된 내용
    """
    
    try:
        # LLM 초기화
        llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.3,
            api_key=os.getenv("OPENAI_API_KEY")
        )
        
        conversation_history = state.get("conversation_history", [])
        retrieved_examples = state.get("retrieved_examples", [])
        
        # 대화 히스토리가 없으면
        if not conversation_history:
            state["model_response"] = "아직 진행된 대화가 없습니다."
            return state
        
        # 전체 대화 포맷팅
        conversation_text = ""
        for msg in conversation_history:
            role = "고객" if msg["role"] == "user" else "상담사"
            conversation_text += f"{role}: {msg['content']}\n"
        
        # Few-shot 예시 포맷팅 (있으면)
        examples_text = ""
        if retrieved_examples:
            examples_text = "\n# 요약 예시:\n"
            for i, ex in enumerate(retrieved_examples[:2], 1):  # 2개만
                examples_text += f"\n[예시 {i}]\n"
                examples_text += f"원본 대화:\n{ex['input'][:300]}...\n"
                examples_text += f"요약:\n{ex['output']}\n"
        
        # 프롬프트 템플릿
        prompt = ChatPromptTemplate.from_messages([
            ("system", """당신은 상담 내용을 요약하는 전문가입니다.

요약 원칙:
1. 핵심 내용만 간결하게 정리 (3-5 문장)
2. 고객의 주요 요청사항과 상담사의 안내 내용 포함
3. 시간 순서대로 정리
4. 중요한 정보(날짜, 금액, 절차 등)는 반드시 포함
5. 불필요한 인사말이나 반복 내용은 제외

{examples}
"""),
            ("user", """다음 상담 내용을 요약해주세요:

{conversation}

요약:""")
        ])
        
        # 체인 생성 및 실행
        chain = prompt | llm
        response = chain.invoke({
            "examples": examples_text,
            "conversation": conversation_text
        })
        
        # 상태 업데이트
        state["model_response"] = response.content.strip()
        
        print(f"[Summary Model] 요약 완료 ({len(conversation_history)}턴 → {len(response.content)}자)")
        
    except Exception as e:
        print(f"[Summary Model] Error: {e}")
        state["model_response"] = "죄송합니다. 요약 생성 중 오류가 발생했습니다."
        state["error"] = f"요약 모델 오류: {str(e)}"
    
    return state
