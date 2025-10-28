"""
Vector Store Node
- Pinecone 또는 ChromaDB를 사용하여 Few-shot 예시 검색
- PINECONE_API_KEY가 있으면 Pinecone, 없으면 ChromaDB 사용
- 저장된 DB가 있으면 로드, 없으면 새로 생성
- 도메인 기반 필터링 및 동적 예시 개수 조정

Input: GraphState
  - user_query: str
  - metadata: Dict {"domain": str, "conversation_turns": int}

Process:
  - domain으로 필터링
  - conversation_turns에 따라 n_results 결정
  - 유사도 검색

Output: GraphState
  - retrieved_examples: List[Dict]
"""

import os
import pandas as pd
from typing import TYPE_CHECKING, List, Dict, Any, Optional
import chromadb
from chromadb.config import Settings

if TYPE_CHECKING:
    from graph import GraphState


# 전역 변수
VECTOR_STORE_TYPE = None  # "pinecone" or "chromadb"
CHROMA_CLIENT = None
CHROMA_COLLECTION = None
PINECONE_INDEX = None

# 도메인 매핑
DOMAIN_MAPPING = {
    "하나카드": "금융",
    "엘지유플러스": "통신",
    "액티벤처": "여행"
}


def _detect_vector_store_type() -> str:
    """사용할 Vector Store 타입 자동 감지"""
    if os.getenv("PINECONE_API_KEY"):
        return "pinecone"
    return "chromadb"


def initialize_vector_store(
    qa_data_path: str = "./data/outputs/see_out - see_out.csv",
    raw_data_path: str = "./data/raw_data_messages.jsonl",
    force_reload: bool = False
):
    """
    Vector Store 초기화
    - PINECONE_API_KEY가 있으면 Pinecone 사용 (실패 시 ChromaDB로 fallback)
    - 없으면 ChromaDB 사용 (디스크 저장)
    - 저장된 DB가 있으면 로드, 없으면 새로 생성
    
    Args:
        qa_data_path: Q&A CSV 파일 경로
        raw_data_path: 원본 상담 대화 JSONL 파일 경로
        force_reload: True면 기존 DB 무시하고 새로 생성
    """
    global VECTOR_STORE_TYPE, CHROMA_CLIENT, CHROMA_COLLECTION, PINECONE_INDEX
    
    # Vector Store 타입 결정
    VECTOR_STORE_TYPE = _detect_vector_store_type()
    print(f"[Vector Store] {VECTOR_STORE_TYPE.upper()} 모드로 초기화 중...")
    
    # Vector Store별 초기화
    if VECTOR_STORE_TYPE == "pinecone":
        try:
            # Pinecone: 저장된 인덱스 확인
            if _check_pinecone_exists() and not force_reload:
                print(f"[Vector Store] ✅ 저장된 Pinecone 인덱스 발견! 로딩 스킵")
                _load_pinecone()
                print(f"[Vector Store] 초기화 완료!")
                return
            else:
                if force_reload:
                    print(f"[Vector Store] force_reload=True, 데이터 재생성 중...")
        except Exception as e:
            print(f"[Vector Store] Pinecone 확인 중 오류 발생")
            # ChromaDB가 이미 있으면 바로 로드하고 종료
            if _check_chromadb_exists():
                print(f"[Vector Store] ✅ 저장된 ChromaDB 발견! 바로 로딩합니다...")
                VECTOR_STORE_TYPE = "chromadb"
                _load_chromadb()
                print(f"[Vector Store] 초기화 완료!")
                return
            else:
                print(f"[Vector Store] ChromaDB로 전환합니다...")
                VECTOR_STORE_TYPE = "chromadb"
    
    if VECTOR_STORE_TYPE == "chromadb":
        # ChromaDB: 저장된 DB 확인
        if _check_chromadb_exists() and not force_reload:
            print(f"[Vector Store] ✅ 저장된 ChromaDB 발견! 로딩 중...")
            _load_chromadb()
            print(f"[Vector Store] 초기화 완료!")
            return
        else:
            if force_reload:
                print(f"[Vector Store] force_reload=True, 데이터 재생성 중...")
    
    # 저장된 DB가 없거나 force_reload=True인 경우 새로 생성
    print(f"[Vector Store] 저장된 DB가 없습니다. 새로 생성 중...")
    
    # 1. Q&A 데이터 로드
    if not os.path.exists(qa_data_path):
        print(f"[Vector Store] Error: {qa_data_path} 파일을 찾을 수 없습니다.")
        return
    
    print(f"[Vector Store] Q&A 데이터 로드 중: {qa_data_path}")
    df_qa = pd.read_csv(qa_data_path)
    
    # short_answer == 1인 것 제외 (짧은 답변 제외)
    df_filtered = df_qa[df_qa['short answer'] != 1].copy()
    print(f"[Vector Store] 필터링 완료: {len(df_qa)} → {len(df_filtered)}개 (short_answer != 1)")
    
    # 전체 데이터 사용
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
        print(f"[Vector Store] Warning: {raw_data_path} 파일을 찾을 수 없습니다.")
    
    # 3. 데이터 준비
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
            
            # source 추출
            source = raw_item.get('source', '')
            
            # domain 매핑
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
        
        # 메타데이터 (source, domain 추가)
        metadatas.append({
            "instruction": str(row['instruction']),
            "output": str(row['output']),
            "conversation": conversation_text,
            "task_category": str(row['task_category']),
            "source": source,
            "domain": domain,
            "source_id": source_id
        })
        
        ids.append(f"qa_{idx}")
    
    # 4. Vector Store별 초기화 (Pinecone 실패 시 ChromaDB로 재시도)
    try:
        if VECTOR_STORE_TYPE == "pinecone":
            _initialize_pinecone(documents, metadatas, ids)
        else:
            _initialize_chromadb(documents, metadatas, ids, force_reload)
    except Exception as e:
        if VECTOR_STORE_TYPE == "pinecone":
            print(f"[Vector Store] Pinecone 초기화 실패, ChromaDB로 전환합니다...")
            VECTOR_STORE_TYPE = "chromadb"
            _initialize_chromadb(documents, metadatas, ids, force_reload=False)
        else:
            raise e
    
    print(f"[Vector Store] 초기화 완료!")


def _check_chromadb_exists() -> bool:
    """ChromaDB가 디스크에 저장되어 있는지 확인"""
    chroma_dir = "./chroma_db"
    return os.path.exists(chroma_dir) and os.path.isdir(chroma_dir)


def _load_chromadb():
    """저장된 ChromaDB 로드"""
    global CHROMA_CLIENT, CHROMA_COLLECTION
    
    CHROMA_CLIENT = chromadb.Client(Settings(
        anonymized_telemetry=False,
        persist_directory="./chroma_db",
        is_persistent=True
    ))
    
    try:
        CHROMA_COLLECTION = CHROMA_CLIENT.get_collection(name="qa_examples")
        count = CHROMA_COLLECTION.count()
        print(f"[Vector Store] ChromaDB 로드 완료: {count}개 벡터")
    except Exception as e:
        print(f"[Vector Store] ChromaDB 로드 실패: {e}")
        CHROMA_COLLECTION = None


def _initialize_chromadb(documents: List[str], metadatas: List[Dict], ids: List[str], force_reload: bool = False):
    """ChromaDB 초기화 (배치 처리, 디스크 저장)"""
    global CHROMA_CLIENT, CHROMA_COLLECTION
    
    # ChromaDB 클라이언트 생성 (디스크 저장)
    CHROMA_CLIENT = chromadb.Client(Settings(
        anonymized_telemetry=False,
        persist_directory="./chroma_db",  # ✅ 디스크에 저장
        is_persistent=True
    ))
    
    # force_reload=True일 때만 기존 컬렉션 삭제
    if force_reload:
        try:
            CHROMA_CLIENT.delete_collection(name="qa_examples")
            print(f"[Vector Store] force_reload=True, 기존 ChromaDB 컬렉션 삭제")
        except:
            pass
    
    # 컬렉션 생성
    CHROMA_COLLECTION = CHROMA_CLIENT.create_collection(
        name="qa_examples",
        metadata={"description": "QA task examples with conversation context"}
    )
    
    # 배치로 추가 (배치 크기 제한 대응)
    if documents:
        batch_size = 5000  # ChromaDB 안전 배치 크기
        total = len(documents)
        
        for i in range(0, total, batch_size):
            end_idx = min(i + batch_size, total)
            batch_docs = documents[i:end_idx]
            batch_metas = metadatas[i:end_idx]
            batch_ids = ids[i:end_idx]
            
            CHROMA_COLLECTION.add(
                documents=batch_docs,
                metadatas=batch_metas,
                ids=batch_ids
            )
            print(f"[Vector Store] ChromaDB 배치 {i//batch_size + 1}: {len(batch_docs)}개 추가 ({end_idx}/{total})")
        
        print(f"[Vector Store] ChromaDB: 총 {total}개 예시 추가 완료 (./chroma_db에 저장됨)")


def _check_pinecone_exists() -> bool:
    """Pinecone 인덱스가 존재하고 데이터가 있는지 확인"""
    try:
        from pinecone import Pinecone
    except ImportError:
        return False
    
    if not os.getenv("PINECONE_API_KEY"):
        return False
    
    try:
        pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        index_name = os.getenv("PINECONE_INDEX_NAME", "counselor-qa")
        
        # 인덱스 존재 확인
        existing_indexes = [idx.name for idx in pc.list_indexes()]
        if index_name not in existing_indexes:
            return False
        
        # 데이터 개수 확인
        index = pc.Index(index_name)
        stats = index.describe_index_stats()
        total_count = stats.get('total_vector_count', 0)
        
        if total_count > 0:
            print(f"[Vector Store] 기존 Pinecone 인덱스 발견: {total_count}개 벡터")
            return True
        
        return False
        
    except Exception as e:
        print(f"[Vector Store] Pinecone 확인 중 오류: {e}")
        return False


def _load_pinecone():
    """저장된 Pinecone 인덱스 로드"""
    global PINECONE_INDEX
    
    try:
        from pinecone import Pinecone
        
        pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        index_name = os.getenv("PINECONE_INDEX_NAME", "counselor-qa")
        
        PINECONE_INDEX = pc.Index(index_name)
        stats = PINECONE_INDEX.describe_index_stats()
        total_count = stats.get('total_vector_count', 0)
        
        print(f"[Vector Store] Pinecone 로드 완료: {total_count}개 벡터")
        
    except Exception as e:
        print(f"[Vector Store] Pinecone 로드 실패: {e}")
        PINECONE_INDEX = None


def _initialize_pinecone(documents: List[str], metadatas: List[Dict], ids: List[str]):
    """Pinecone 초기화"""
    global PINECONE_INDEX
    
    try:
        from pinecone import Pinecone, ServerlessSpec
        from langchain_openai import OpenAIEmbeddings
    except ImportError:
        print("[Vector Store] Error: Pinecone 라이브러리가 설치되지 않았습니다.")
        print("설치 명령어: pip install pinecone-client langchain-openai")
        return
    
    # Pinecone 초기화
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    
    # 인덱스 이름
    index_name = os.getenv("PINECONE_INDEX_NAME", "counselor-qa")
    
    # 인덱스 존재 확인
    existing_indexes = [idx.name for idx in pc.list_indexes()]
    
    # 기존 인덱스 삭제 (force_reload 대응)
    if index_name in existing_indexes:
        print(f"[Vector Store] 기존 Pinecone 인덱스 '{index_name}' 삭제 중...")
        pc.delete_index(index_name)
        import time
        time.sleep(5)  # 삭제 대기
    
    # 인덱스 생성
    print(f"[Vector Store] Pinecone 인덱스 '{index_name}' 생성 중...")
    
    # Pinecone Serverless 설정
    cloud = os.getenv("PINECONE_CLOUD", "gcp")  # gcp 또는 aws
    region = os.getenv("PINECONE_REGION", "us-central1")  # gcp의 경우 us-central1
    
    pc.create_index(
        name=index_name,
        dimension=1536,  # OpenAI embedding 차원
        metric="cosine",
        spec=ServerlessSpec(
            cloud=cloud,
            region=region
        )
    )
    print(f"[Vector Store] Pinecone 인덱스 '{index_name}' 생성 완료 (cloud={cloud}, region={region})")
    
    PINECONE_INDEX = pc.Index(index_name)
    
    # Embedding 모델
    embeddings = OpenAIEmbeddings(
        model="text-embedding-ada-002",
        openai_api_key=os.getenv("OPENAI_API_KEY")
    )
    
    # 배치로 업로드
    batch_size = 100  # Pinecone 권장 배치 크기
    total = len(documents)
    
    print(f"[Vector Store] Pinecone에 {total}개 벡터 업로드 중...")
    
    for i in range(0, total, batch_size):
        end_idx = min(i + batch_size, total)
        batch_docs = documents[i:end_idx]
        batch_metas = metadatas[i:end_idx]
        batch_ids = ids[i:end_idx]
        
        # Embedding 생성
        batch_embeddings = embeddings.embed_documents(batch_docs)
        
        # Pinecone 업로드 형식
        vectors = []
        for j, (doc_id, embedding, metadata) in enumerate(zip(batch_ids, batch_embeddings, batch_metas)):
            vectors.append({
                "id": doc_id,
                "values": embedding,
                "metadata": metadata
            })
        
        # 업로드
        PINECONE_INDEX.upsert(vectors=vectors)
        print(f"[Vector Store] Pinecone 배치 {i//batch_size + 1}: {len(vectors)}개 업로드 ({end_idx}/{total})")
    
    print(f"[Vector Store] Pinecone: 총 {total}개 벡터 업로드 완료")


def retrieve_examples(state: "GraphState") -> "GraphState":
    """
    메타데이터 기반 Few-shot 예시 검색
    - Pinecone 또는 ChromaDB에서 검색
    
    Input: GraphState
      - user_query: str
      - metadata: Dict {"domain": str, "conversation_turns": int}
    
    Output: GraphState
      - retrieved_examples: List[Dict]
    """
    global VECTOR_STORE_TYPE, CHROMA_COLLECTION, PINECONE_INDEX
    
    try:
        # Vector Store 초기화 확인
        if VECTOR_STORE_TYPE is None:
            print("[Vector Store] 초기화되지 않음. 초기화 중...")
            initialize_vector_store()
        
        user_query = state["user_query"]
        metadata = state.get("metadata", {})
        
        # 메타데이터 추출
        domain = metadata.get("domain", "기타")
        conversation_turns = metadata.get("conversation_turns", 0)
        
        # conversation_turns에 따라 검색 개수 결정
        if conversation_turns <= 20:
            n_results = 2  # 짧은 대화 → 예시 2개
        else:
            n_results = 1  # 긴 대화 → 예시 1개 (토큰 절약)
        
        # Vector Store별 검색
        if VECTOR_STORE_TYPE == "pinecone":
            examples = _search_pinecone(user_query, domain, n_results)
        else:
            examples = _search_chromadb(user_query, domain, n_results)
        
        state["retrieved_examples"] = examples
        
        print(f"[Vector Store] {len(examples)}개 예시 검색 완료 (turns={conversation_turns}, n_results={n_results})")
        
    except Exception as e:
        print(f"[Vector Store] Error: {e}")
        state["retrieved_examples"] = []
        state["error"] = f"검색 중 오류: {str(e)}"
    
    return state


def _search_chromadb(user_query: str, domain: str, n_results: int) -> List[Dict]:
    """ChromaDB에서 검색"""
    global CHROMA_COLLECTION
    
    if CHROMA_COLLECTION is None:
        print(f"[Vector Store] Warning: ChromaDB 컬렉션이 없습니다.")
        return []
    
    # domain 필터 적용
    if domain != "기타":
        results = CHROMA_COLLECTION.query(
            query_texts=[user_query],
            n_results=n_results,
            where={"domain": domain}
        )
        print(f"[Vector Store] ChromaDB domain='{domain}' 필터 적용")
    else:
        results = CHROMA_COLLECTION.query(
            query_texts=[user_query],
            n_results=n_results
        )
        print(f"[Vector Store] ChromaDB domain 필터 없음 (전체 검색)")
    
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
    
    return examples


def _search_pinecone(user_query: str, domain: str, n_results: int) -> List[Dict]:
    """Pinecone에서 검색"""
    global PINECONE_INDEX
    
    if PINECONE_INDEX is None:
        print(f"[Vector Store] Warning: Pinecone 인덱스가 없습니다.")
        return []
    
    try:
        from langchain_openai import OpenAIEmbeddings
    except ImportError:
        print("[Vector Store] Error: langchain-openai가 설치되지 않았습니다.")
        return []
    
    # Query Embedding 생성
    embeddings = OpenAIEmbeddings(
        model="text-embedding-ada-002",
        openai_api_key=os.getenv("OPENAI_API_KEY")
    )
    query_embedding = embeddings.embed_query(user_query)
    
    # domain 필터 적용
    filter_dict = {"domain": domain} if domain != "기타" else {}
    
    # Pinecone 검색
    results = PINECONE_INDEX.query(
        vector=query_embedding,
        top_k=n_results,
        filter=filter_dict,
        include_metadata=True
    )
    
    if domain != "기타":
        print(f"[Vector Store] Pinecone domain='{domain}' 필터 적용")
    else:
        print(f"[Vector Store] Pinecone domain 필터 없음 (전체 검색)")
    
    # 결과 포맷팅
    examples = []
    for match in results.matches:
        metadata = match.metadata
        examples.append({
            "instruction": metadata.get("instruction", ""),
            "conversation": metadata.get("conversation", ""),
            "output": metadata.get("output", ""),
            "task_category": metadata.get("task_category", ""),
            "source": metadata.get("source", ""),
            "domain": metadata.get("domain", "")
        })
    
    return examples


def clear_vector_store():
    """Vector Store 초기화 (메모리 해제)"""
    global VECTOR_STORE_TYPE, CHROMA_CLIENT, CHROMA_COLLECTION, PINECONE_INDEX
    
    if VECTOR_STORE_TYPE == "chromadb" and CHROMA_CLIENT:
        try:
            CHROMA_CLIENT.delete_collection("qa_examples")
        except:
            pass
        CHROMA_CLIENT = None
        CHROMA_COLLECTION = None
    
    elif VECTOR_STORE_TYPE == "pinecone" and PINECONE_INDEX:
        # Pinecone은 클라우드 기반이므로 명시적 삭제 불필요
        PINECONE_INDEX = None
    
    VECTOR_STORE_TYPE = None
    print("[Vector Store] 초기화됨")