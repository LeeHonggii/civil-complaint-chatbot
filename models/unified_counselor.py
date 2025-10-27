"""
Unified Counselor Node
- 질의응답(QA) 처리
- Few-shot 예시 활용 (1-2개)
- 대화 맥락 기반 답변

Input: GraphState
  - user_query: str
  - recent_context: List[Message] (최근 8턴)
  - retrieved_examples: List[Dict] (1-2개)

Process:
  - Few-shot 예시 포맷팅
  - 대화 맥락 포함
  - LLM 호출

Output: GraphState
  - model_response: str
  
"""

import os
from typing import TYPE_CHECKING, List, Dict
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

if TYPE_CHECKING:
    from graph import GraphState, Message


def unified_counselor(state: "GraphState") -> "GraphState":
    """
    질의응답 모델 (단순화)
    - Few-shot 예시 활용
    - 대화 맥락 기반 답변
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
        
        # 프롬프트 생성
        prompt_text = build_qa_prompt(
            user_query=user_query,
            recent_context=recent_context,
            retrieved_examples=retrieved_examples
        )
        
        # LLM 호출
        response = llm.invoke(prompt_text)
        
        # 상태 업데이트
        state["model_response"] = response.content.strip()
        
        print(f"[Unified Counselor] 답변 생성 완료 ({len(response.content)}자)")
        
    except Exception as e:
        print(f"[Unified Counselor] Error: {e}")
        state["model_response"] = "죄송합니다. 답변 생성 중 오류가 발생했습니다."
        state["error"] = f"통합 모델 오류: {str(e)}"
    
    return state


def build_qa_prompt(
    user_query: str,
    recent_context: List["Message"],
    retrieved_examples: List[Dict]
) -> str:
    """
    QA 프롬프트 생성
    - Few-shot 예시 활용
    - 최근 대화 맥락 포함
    - 메타데이터 정보 포함
    """
    
    # 1. Few-shot 예시 포맷팅 (메타데이터 포함)
    examples_text = ""
    if retrieved_examples:
        examples_text = "\n# 📚 유사한 상담 예시:\n"
        for i, ex in enumerate(retrieved_examples, 1):
            examples_text += f"\n[예시 {i}]\n"
            
            # ✨ 메타데이터 정보 추가
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
                conv_preview = ex['conversation'][:300]  # 300자로 제한
                examples_text += f"상담 대화:\n{conv_preview}...\n\n"
            
            examples_text += f"질문: {ex['instruction']}\n"
            examples_text += f"답변: {ex['output']}\n"
    
    # 2. 시스템 프롬프트
    system_prompt = """당신은 전문 상담사입니다.

## 답변 원칙
1. 구체적이고 명확한 답변 제공
2. 이전 대화 맥락을 고려하여 자연스럽게 답변
3. 유사 예시를 참고하되, 현재 상황에 맞게 답변
4. 정보가 불확실하면 "확인이 필요합니다" 안내
5. 친근하고 자연스러운 말투
6. 간결하게 답변 (불필요한 설명 자제)
"""
    
    # 3. 유사 예시 추가
    if examples_text:
        system_prompt += examples_text
    
    # 4. 대화 맥락 포맷팅
    context_text = "\n## 💬 이전 대화 맥락:\n"
    
    if recent_context:
        for msg in recent_context:
            role = "고객" if msg["role"] == "user" else "상담사"
            context_text += f"{role}: {msg['content']}\n"
    else:
        context_text += "(첫 대화입니다)\n"
    
    # 5. 현재 질문
    query_text = f"\n## ❓ 현재 질문:\n고객: {user_query}\n\n상담사: "
    
    return system_prompt + context_text + query_text