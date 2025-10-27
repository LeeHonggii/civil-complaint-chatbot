"""
Vector Store Node
- ChromaDB를 사용하여 Few-shot 예시 검색
- 각 태스크별 50개 샘플 데이터 저장
- Top-3 유사 예시 반환

Input: GraphState
  - user_query: str
  - query_type: "qa" | "summary" | "classification"

Process:
  - query_type에 따라 해당 컬렉션 선택
  - user_query로 유사도 검색
  - Top-3 예시 반환

Output: GraphState
  - retrieved_examples: List[Dict]
    [{"instruction": "...", "input": "...", "output": "..."}]
"""

import os
import pandas as pd
from typing import TYPE_CHECKING, List, Dict, Any
import chromadb
from chromadb.config import Settings

if TYPE_CHECKING:
    from graph import GraphState


# ChromaDB 클라이언트 (전역)
CHROMA_CLIENT = None
COLLECTIONS = {}

# ✨ 회사명 → 도메인 매핑
DOMAIN_MAPPING = {
    "하나카드": "금융",
    "엘지유플러스": "통신",
    "액티벤처": "여행"
}


def initialize_vector_store(
    qa_data_path: str = "./data/outputs/see_out - see_out.csv",
    raw_data_path: str = "./data/raw_data_messages.jsonl"
):
    """
    Vector Store 초기화 - 50개 샘플 데이터 로드
    상담 대화 내용을 포함하여 Few-shot 예시로 활용
    
    Args:
        qa_data_path: Q&A CSV 파일 경로
        raw_data_path: 원본 상담 대화 JSONL 파일 경로
    """
    global CHROMA_CLIENT, COLLECTIONS
    
    print("[Vector Store] 초기화 중...")
    
    # ChromaDB 클라이언트 생성
    CHROMA_CLIENT = chromadb.Client(Settings(
        anonymized_telemetry=False,
        is_persistent=False  # 메모리 기반
    ))
    
    # 1. Q&A 데이터 로드
    if not os.path.exists(qa_data_path):
        print(f"[Vector Store] Error: {qa_data_path} 파일을 찾을 수 없습니다.")
        return
    
    print(f"[Vector Store] Q&A 데이터 로드 중: {qa_data_path}")
    df_qa = pd.read_csv(qa_data_path)
    
    # ✨ short_answer == 1인 것 제외 (짧은 답변 제외)
    df_filtered = df_qa[df_qa['short answer'] != 1].copy()
    print(f"[Vector Store] 필터링 완료: {len(df_qa)} → {len(df_filtered)}개 (short_answer != 1)")
    
    # ✨ 전체 데이터 사용 (50개 제한 제거)
    df_sample = df_filtered
    print(f"[Vector Store] 전체 데이터 사용: {len(df_sample)}개")
    
    # 2. 원본 상담 대화 로드
    raw_data_dict = {}
    if os.path.exists(raw_data_path):
        print(f"[Vector Store] 원본 대화 데이터 로드 중: {raw_data_path}")
        import json
        with open(raw_data_path, 'r', encoding='utf-8-sig') as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    raw_data_dict[item['source_id']] = item
        print(f"[Vector Store] 원본 대화 {len(raw_data_dict)}개 로드 완료")
    else:
        print(f"[Vector Store] Warning: {raw_data_path} 파일을 찾을 수 없습니다. 원본 대화 없이 진행합니다.")
    
    # 3. QA 컬렉션 생성
    collection_name = "qa_examples"
    collection = CHROMA_CLIENT.create_collection(
        name=collection_name,
        metadata={"description": "QA task examples with conversation context"}
    )
    COLLECTIONS[collection_name] = collection
    
    # 4. 데이터 추가
    documents = []
    metadatas = []
    ids = []
    
    for idx, row in df_sample.iterrows():
        # 검색용 텍스트 (instruction)
        doc_text = str(row['instruction'])
        documents.append(doc_text)
        
        # source_id로 원본 대화 조회
        source_id = str(row['source_id'])
        conversation_text = ""
        source = ""
        domain = "기타"
        
        if source_id in raw_data_dict:
            raw_item = raw_data_dict[source_id]
            
            # ✨ source 추출
            source = raw_item.get('source', '')
            
            # ✨ domain 매핑
            domain = DOMAIN_MAPPING.get(source, "기타")
            
            messages = raw_item.get('messages', [])
            
            # 최근 3턴만 추출 (토큰 절약)
            recent_messages = messages[-6:] if len(messages) > 6 else messages
            
            # 대화 텍스트 생성
            conversation_lines = []
            for msg in recent_messages:
                if 'user' in msg:
                    conversation_lines.append(f"고객: {msg['user']}")
                elif 'assistance' in msg:
                    conversation_lines.append(f"상담사: {msg['assistance']}")
            
            conversation_text = "\n".join(conversation_lines)
        
        # ✨ 메타데이터 (source, domain 추가)
        metadatas.append({
            "instruction": str(row['instruction']),
            "output": str(row['output']),
            "conversation": conversation_text,
            "task_category": str(row['task_category']),
            "source": source,              # 회사명
            "domain": domain,              # 도메인 (금융/통신/여행)
            "source_id": source_id
        })
        
        ids.append(f"qa_{idx}")
    
    # ChromaDB에 추가
    if documents:
        collection.add(
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )
        print(f"[Vector Store] QA: {len(documents)}개 예시 추가 (대화 맥락 포함)")
    
    print(f"[Vector Store] 초기화 완료! 총 컬렉션: {list(COLLECTIONS.keys())}")


def retrieve_examples(state: "GraphState") -> "GraphState":
    """
    메타데이터 기반 Few-shot 예시 검색
    
    Input: GraphState
      - user_query: str
      - metadata: Dict {
          "domain": str,
          "conversation_turns": int
        }
    
    Process:
      - domain으로 필터링
      - conversation_turns에 따라 n_results 결정 (≤20: 2개, >20: 1개)
      - user_query로 유사도 검색
    
    Output: GraphState
      - retrieved_examples: List[Dict]
    """
    
    global CHROMA_CLIENT, COLLECTIONS
    
    try:
        # Vector Store 초기화 확인
        if CHROMA_CLIENT is None or not COLLECTIONS:
            print("[Vector Store] 초기화되지 않음. 초기화 중...")
            initialize_vector_store()
        
        user_query = state["user_query"]
        metadata = state.get("metadata", {})
        
        # 메타데이터 추출
        domain = metadata.get("domain", "기타")
        conversation_turns = metadata.get("conversation_turns", 0)
        
        # ✨ conversation_turns에 따라 검색 개수 결정
        if conversation_turns <= 20:
            n_results = 2  # 짧은 대화 → 예시 2개
        else:
            n_results = 1  # 긴 대화 → 예시 1개 (토큰 절약)
        
        collection = COLLECTIONS.get("qa_examples")
        
        if collection is None:
            print(f"[Vector Store] Warning: qa_examples 컬렉션이 없습니다.")
            state["retrieved_examples"] = []
            return state
        
        # ✨ domain 필터 적용
        if domain != "기타":
            # domain이 명확한 경우 필터링
            results = collection.query(
                query_texts=[user_query],
                n_results=n_results,
                where={"domain": domain}  # domain 필터
            )
            print(f"[Vector Store] domain='{domain}' 필터 적용")
        else:
            # domain이 기타인 경우 전체 검색
            results = collection.query(
                query_texts=[user_query],
                n_results=n_results
            )
            print(f"[Vector Store] domain 필터 없음 (전체 검색)")
        
        # 결과 포맷팅
        examples = []
        if results["metadatas"] and results["metadatas"][0]:
            for metadata_item in results["metadatas"][0]:
                examples.append({
                    "instruction": metadata_item["instruction"],
                    "conversation": metadata_item.get("conversation", ""),
                    "output": metadata_item["output"],
                    "task_category": metadata_item.get("task_category", ""),
                    "source": metadata_item.get("source", ""),
                    "domain": metadata_item.get("domain", "")
                })
        
        state["retrieved_examples"] = examples
        
        print(f"[Vector Store] {len(examples)}개 예시 검색 완료 (turns={conversation_turns}, n_results={n_results})")
        
    except Exception as e:
        print(f"[Vector Store] Error: {e}")
        state["retrieved_examples"] = []
        state["error"] = f"검색 중 오류: {str(e)}"
    
    return state


def get_examples_by_type(query_type: str, n: int = 3) -> List[Dict[str, Any]]:
    """
    특정 타입의 예시를 직접 가져오기 (유틸리티)
    
    Args:
        query_type: "qa" | "summary" | "classification"
        n: 가져올 예시 수
    
    Returns:
        List[Dict]: 예시 리스트
    """
    global COLLECTIONS
    
    collection_name = f"{query_type}_examples"
    
    if COLLECTIONS.get(collection_name) is None:
        initialize_vector_store()
    
    collection = COLLECTIONS.get(collection_name)
    if collection is None:
        return []
    
    # 랜덤 샘플링
    results = collection.get(limit=n)
    
    examples = []
    if results["metadatas"]:
        for metadata in results["metadatas"]:
            examples.append({
                "instruction": metadata["instruction"],
                "output": metadata["output"],
                "task_category": metadata.get("task_category", "")
            })
    
    return examples


def clear_vector_store():
    """Vector Store 초기화 (메모리 해제)"""
    global CHROMA_CLIENT, COLLECTIONS
    
    if CHROMA_CLIENT:
        # 컬렉션 삭제
        for collection_name in list(COLLECTIONS.keys()):
            try:
                CHROMA_CLIENT.delete_collection(collection_name)
            except:
                pass
        
        COLLECTIONS.clear()
        CHROMA_CLIENT = None
        print("[Vector Store] 초기화됨")