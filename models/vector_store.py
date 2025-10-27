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
    
    # short_answer != 1.0인 것만 필터링 (긴 답변만)
    df_filtered = df_qa[df_qa['short answer'] != 1.0].copy()
    print(f"[Vector Store] 필터링 완료: {len(df_qa)} → {len(df_filtered)}개 (short_answer != 1.0)")
    
    # 50개 샘플만 선택
    df_sample = df_filtered.head(50)
    print(f"[Vector Store] 샘플 선택: {len(df_sample)}개")
    
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
        
        if source_id in raw_data_dict:
            raw_item = raw_data_dict[source_id]
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
        
        # 메타데이터
        metadatas.append({
            "instruction": str(row['instruction']),
            "output": str(row['output']),
            "conversation": conversation_text,  # 원본 대화 추가
            "task_category": str(row['task_category']),
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
    유사한 Few-shot 예시 검색 (1개만)
    
    Input: GraphState
      - user_query: str
      - query_type: "qa" | "summary" | "classification"
    
    Process:
      - query_type에 따라 컬렉션 선택
      - user_query로 유사도 검색
      - Top-1 예시 반환
    
    Output: GraphState
      - retrieved_examples: List[Dict]
        [{"instruction": "...", "conversation": "...", "output": "..."}]
    """
    
    global CHROMA_CLIENT, COLLECTIONS
    
    try:
        # Vector Store 초기화 확인
        if CHROMA_CLIENT is None or not COLLECTIONS:
            print("[Vector Store] 초기화되지 않음. 초기화 중...")
            initialize_vector_store()
        
        user_query = state["user_query"]
        query_type = state.get("query_type", "qa")
        
        # 해당 태스크 컬렉션 선택
        collection_name = f"{query_type}_examples"
        collection = COLLECTIONS.get(collection_name)
        
        if collection is None:
            print(f"[Vector Store] Warning: {collection_name} 컬렉션이 없습니다.")
            state["retrieved_examples"] = []
            return state
        
        # 유사도 검색 (Top-1만)
        results = collection.query(
            query_texts=[user_query],
            n_results=1
        )
        
        # 결과 포맷팅
        examples = []
        if results["metadatas"] and results["metadatas"][0]:
            for metadata in results["metadatas"][0]:
                examples.append({
                    "instruction": metadata["instruction"],
                    "conversation": metadata.get("conversation", ""),
                    "output": metadata["output"],
                    "task_category": metadata.get("task_category", "")
                })
        
        state["retrieved_examples"] = examples
        
        print(f"[Vector Store] {query_type}에서 {len(examples)}개 예시 검색 완료")
        
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