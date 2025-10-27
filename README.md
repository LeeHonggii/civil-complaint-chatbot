# civil-complaint-chatbot
민원상담-챗봇-프로젝트

> **팀명**: 상담랜드  
> **프로젝트**: 멀티턴 대화 맥락을 유지하는 고객 상담 챗봇  
> **기간**: 2025년 10월  

---

## 📋 목차

1. [프로젝트 개요](#-프로젝트-개요)
2. [주요 기능](#-주요-기능)
3. [시스템 아키텍처](#-시스템-아키텍처)
4. [파일 구조](#-파일-구조)
5. [데이터 구조](#-데이터-구조)
6. [태스크별 입출력](#-태스크별-입출력)
7. [설치 및 실행](#-설치-및-실행)
8. [사용 방법](#-사용-방법)
9. [향후 계획](#-향후-계획)

---

## 🎯 프로젝트 개요

### 목표
고객과의 **멀티턴 대화 맥락을 유지**하며 자연스럽게 응답하는 상담 챗봇 구축

### 핵심 기능
- ✅ **멀티턴 대화 지원**: 최근 8턴 맥락 유지 (슬라이딩 윈도우)
- ✅ **3가지 태스크 처리**: 질의응답(QA), 요약(Summary), 분류(Classification)
- ✅ **Few-shot Learning**: Vector DB를 활용한 유사 예시 검색
- ✅ **세션 관리**: 대화별 독립적인 세션 저장
- ✅ **실시간 응답**: Streamlit 기반 대화형 UI

### 기술 스택
- **Framework**: LangGraph (워크플로우), Streamlit (UI)
- **LLM**: OpenAI GPT-4o-mini (추후 파인튜닝 모델로 변경 예정)
- **Vector DB**: ChromaDB (메모리 기반)
- **데이터**: 상담 대화 데이터 (9,773건)

---

## ✨ 주요 기능

### 1. 자동 쿼리 분류
사용자 질문을 분석하여 자동으로 적절한 태스크로 라우팅
```
"카드 분실 시점은?" → QA
"지금까지 뭘 얘기했어?" → Summary
"이건 무슨 민원이야?" → Classification
```

### 2. 멀티턴 대화 맥락 유지
슬라이딩 윈도우 방식으로 최근 8턴의 대화를 기억
```
고객: 카드 분실했어요
상담사: 언제 분실하셨나요?
고객: 어제요  ← 맥락 이해!
상담사: 어제 분실하신 카드 재발급 도와드리겠습니다.
```

### 3. Few-shot 예시 활용
유사한 상담 사례를 검색하여 정확한 답변 생성
```
[유사 예시]
상담 대화: 고객이 해외에서 카드 사용...
질문: 고객은 무엇을 요청했나?
답변: 카드 정지 해제
```

---

## 🏗️ 시스템 아키텍처

### 전체 파이프라인
```
┌─────────────────────────────────────────────────────────┐
│                    사용자 입력                            │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│              Query Classifier                           │
│  - GPT-4o-mini로 쿼리 타입 분류                          │
│  - "qa" | "summary" | "classification"                  │
└─────────────────────────────────────────────────────────┘
                          ↓
            ┌─────────────┼─────────────┐
            ↓             ↓             ↓
    ┌──────────┐  ┌──────────┐  ┌──────────┐
    │    QA    │  │   요약    │  │   분류    │
    └──────────┘  └──────────┘  └──────────┘
         ↓             ↓             ↓
         │      ┌──────┴──────┐      │
         │      ↓             ↓      │
         │  Vector DB     Vector DB  │
         │  (예시 검색)   (예시 검색) │
         │      ↓             ↓      │
         │  Few-shot      Few-shot   │
         │  Prompt        Prompt     │
         └──────┼─────────┼──────────┘
                ↓
┌─────────────────────────────────────────────────────────┐
│           GPT-4o-mini (추후 파인튜닝 모델)                │
│           + 멀티턴 맥락 유지 (슬라이딩 윈도우)              │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│                    최종 답변 생성                          │
└─────────────────────────────────────────────────────────┘
```

### 세션 관리
```python
SessionStore (메모리 기반)
├── session_id_1
│   ├── conversation_history: [...]
│   ├── created_at: "2025-10-27T10:00:00"
│   └── last_updated: "2025-10-27T10:15:00"
├── session_id_2
│   └── ...
```

---

## 📁 파일 구조

```
project/
├── main.py                          # Streamlit UI 메인 앱
├── graph.py                         # LangGraph 구조 + 상태 정의 + 세션 관리
│
├── models/                          # 노드 모듈
│   ├── __init__.py                 # 패키지 초기화
│   ├── query_classifier.py         # 쿼리 분류 노드
│   ├── vector_store.py             # Vector DB 검색 노드
│   ├── qa_model.py                 # 질의응답 노드
│   ├── summary_model.py            # 요약 노드
│   └── classification_model.py     # 분류 노드
│
├── data/                            # 데이터
│   ├── outputs/
│   │   └── see_out - see_out.csv   # Q&A 데이터 (50개 샘플)
│   └── raw_data_messages.jsonl     # 원본 상담 대화 (9,773건)
│
├── .env                             # 환경 변수 (OPENAI_API_KEY)
├── requirements.txt                 # 의존성 패키지
└── README.md                        # 프로젝트 문서
```

---

## 📊 데이터 구조

### 1. see_out - see_out.csv (Q&A 데이터)
Few-shot 예시로 사용되는 질문-답변 쌍

```csv
source_id,file_id,task_category,instruction,object_key,object,short answer,output
200001,03_질의응답_200001_1,의문사형,고객은 무엇을 요구하고 있지?,무엇,what,NaN,정지 카드 해제
200001,03_질의응답_200001_2,의문사형,고객은 왜 핸드폰 문자 인증을 하기 어렵다고 했니?,,,NaN,지금은 한국 핸드폰 유심칩을 쓰기 어렵기 때문이야.
```

**주요 컬럼**:
- `source_id`: 원천 데이터 ID (상담 대화와 매칭)
- `task_category`: "의문사형" | "예/아니요형"
- `instruction`: 질문
- `output`: 답변
- `short answer`: 1.0이면 짧은 답변 (필터링 대상)

**사용 방식**:
- `short answer != 1.0`인 데이터만 사용 (긴 답변)
- 50개 샘플 선택
- Vector DB에 저장

---

### 2. raw_data_messages.jsonl (원본 상담 대화)
실제 고객-상담사 대화 내용

```json
{
  "source_id": "90001",
  "source": "액티벤처",
  "consulting_turns": "14",
  "consulting_length": 167,
  "messages": [
    {"user": "안녕하세요, 여행 관련해서 문의드릴게요. 5월 10일부터..."},
    {"assistance": "안녕하세요! 여행 계획 중이시군요. 어떤 숙소를..."},
    {"user": "파드마 우붓 리조트에서 4박을 하고 싶어요..."},
    {"assistance": "네, 파드마 우붓 리조트에서 왕복 픽업 가능합니다..."},
    ...
  ]
}
```

**주요 필드**:
- `source_id`: 고유 ID (see_out.csv와 매칭)
- `messages`: 대화 턴 배열
  - `user`: 고객 메시지
  - `assistance`: 상담사 메시지

**사용 방식**:
- `source_id`로 Q&A 데이터와 조인
- 최근 3턴(6개 메시지)만 추출
- Few-shot 예시의 맥락으로 활용

---

## 🔧 태스크별 입출력

### 1️⃣ Query Classifier (쿼리 분류)

**Input (GraphState)**:
```python
{
    "user_query": "카드 분실 시점은?",
    "recent_context": [
        {"role": "user", "content": "카드 분실했어요", "timestamp": "..."},
        {"role": "assistant", "content": "언제 분실하셨나요?", "timestamp": "..."}
    ]
}
```

**Process**:
- LangChain Structured Output 사용
- GPT-4o-mini 호출 (temperature=0)
- 3가지 타입 중 분류

**Output (GraphState)**:
```python
{
    "query_type": "qa"  # "qa" | "summary" | "classification"
}
```

---

### 2️⃣ Vector Store (예시 검색)

**Input (GraphState)**:
```python
{
    "user_query": "고객은 무엇을 요청했어?",
    "query_type": "qa"
}
```

**Process**:
- ChromaDB 유사도 검색
- Top-1 예시 반환
- source_id로 원본 대화 조인 (최근 3턴)

**Output (GraphState)**:
```python
{
    "retrieved_examples": [
        {
            "instruction": "고객은 무엇을 요구하고 있지?",
            "conversation": "고객: 카드 분실했어요\n상담사: 언제 분실하셨나요?...",
            "output": "정지 카드 해제",
            "task_category": "의문사형",
            "source_id": "200001"
        }
    ]
}
```

---

### 3️⃣ QA Model (질의응답)

**Input (GraphState)**:
```python
{
    "user_query": "카드 분실 시점은?",
    "recent_context": [최근 8턴 대화],
    "retrieved_examples": [Few-shot 예시 1개]
}
```

**Process**:
```
프롬프트 구성:
┌─────────────────────────────────┐
│ [예시]                           │
│ 상담 대화:                       │
│ 고객: ...                        │
│ 상담사: ...                      │
│                                 │
│ 질문: 고객은 무엇을 요청했어?     │
│ 답변: 정지 카드 해제             │
├─────────────────────────────────┤
│ # 이전 대화 맥락:                │
│ 고객: 카드 분실했어요             │
│ 상담사: 언제 분실하셨나요?        │
├─────────────────────────────────┤
│ # 현재 질문:                     │
│ 고객: 어제요                     │
│                                 │
│ 상담사:                          │
└─────────────────────────────────┘
```

**Output (GraphState)**:
```python
{
    "model_response": "어제 분실하신 카드 재발급을 도와드리겠습니다."
}
```

**API**:
- Model: `gpt-4o-mini` (추후 파인튜닝 모델로 변경 예정)
- Temperature: `0.3`

---

### 4️⃣ Summary Model (요약)

**Input (GraphState)**:
```python
{
    "conversation_history": [전체 대화 히스토리],
    "retrieved_examples": [요약 예시 2개 (옵션)]
}
```

**Process**:
- 전체 대화를 텍스트로 변환
- 3-5문장으로 요약

**Output (GraphState)**:
```python
{
    "model_response": "고객이 해외에서 카드 분실을 신고하고, 문자 인증 어려움으로 인해 대체 인증 방법을 안내받았습니다. 카드 재발급 절차를 안내받았습니다."
}
```

**API**:
- Model: `gpt-4o-mini` (추후 파인튜닝 모델로 변경 예정)
- Temperature: `0.3`

---

### 5️⃣ Classification Model (분류)

**Input (GraphState)**:
```python
{
    "user_query": "이건 무슨 민원이야?",
    "recent_context": [최근 대화],
    "retrieved_examples": [분류 예시 3개]
}
```

**Process**:
- 대화 내용을 분석하여 유형 분류
- 가능한 카테고리:
  - 상담 요건: "단일 요건 민원", "다수 요건 민원"
  - 상담 주제: "카드 분실/도난", "요금 문의", "서비스 신청" 등

**Output (GraphState)**:
```python
{
    "model_response": "단일 요건 민원 - 카드 분실/도난"
}
```

**API**:
- Model: `gpt-4o-mini` (추후 파인튜닝 모델로 변경 예정)
- Temperature: `0.1`

---

## 📦 GraphState 전체 구조

```python
class Message(TypedDict):
    """개별 메시지"""
    role: Literal["user", "assistant"]   # 발화자
    content: str                         # 메시지 내용
    timestamp: str                       # ISO 포맷 시간


class GraphState(TypedDict):
    """LangGraph 전체 상태"""
    
    # 입력
    user_query: str                      # 현재 사용자 질문
    conversation_history: List[Message]  # 전체 대화 히스토리
    
    # 분류
    query_type: Optional[Literal["qa", "summary", "classification"]]
    
    # 검색
    retrieved_examples: Optional[List[Dict[str, Any]]]
    
    # 컨텍스트
    recent_context: List[Message]        # 슬라이딩 윈도우 (최근 8턴)
    
    # 출력
    model_response: Optional[str]        # 최종 생성 답변
    
    # 메타
    session_id: str                      # 세션 ID (UUID)
    error: Optional[str]                 # 에러 메시지
```

---

## 🚀 설치 및 실행

### 1. 환경 설정

#### 필요 패키지 설치
```bash
pip install streamlit langchain langchain-openai chromadb pandas python-dotenv langgraph
```

#### 또는 requirements.txt 사용
```bash
pip install -r requirements.txt
```

**requirements.txt**:
```
streamlit==1.28.0
langchain==0.1.0
langchain-openai==0.0.2
chromadb==0.4.18
pandas==2.1.3
python-dotenv==1.0.0
langgraph==0.0.20
pydantic==2.5.0
```

---

### 2. 환경 변수 설정

프로젝트 루트에 `.env` 파일 생성:
```bash
OPENAI_API_KEY=your_openai_api_key_here
```

---

### 3. 데이터 준비

파일 위치 확인:
```
project/
└── data/
    ├── outputs/
    │   └── see_out - see_out.csv
    └── raw_data_messages.jsonl
```

---

### 4. 실행

```bash
streamlit run main.py
```

브라우저에서 자동으로 `http://localhost:8501` 열림

---

## 💡 사용 방법

### 1단계: 앱 시작
```bash
streamlit run main.py
```
- Vector Store가 자동으로 초기화됩니다 (첫 실행시 약 3-5초 소요)

---

### 2단계: 대화 시작

#### 질의응답 (QA)
```
사용자: 카드 분실했어요
챗봇: 언제 분실하셨나요?
사용자: 어제요
챗봇: 어제 분실하신 카드 재발급을 도와드리겠습니다...
```

#### 요약
```
사용자: 지금까지 뭘 얘기했어?
챗봇: 고객이 어제 카드를 분실하여 재발급을 요청하셨습니다...
```

#### 분류
```
사용자: 이건 무슨 민원이야?
챗봇: 단일 요건 민원 - 카드 분실/도난
```

---

### 3단계: 새 대화 시작

사이드바에서 **"🔄 새 대화 시작"** 버튼 클릭
- 이전 대화가 초기화됩니다
- 새로운 세션 ID가 생성됩니다

---

### UI 기능

#### 사이드바
- ✅ Vector Store 상태 표시
- 🔄 새 대화 시작 버튼
- 📝 현재 세션 ID
- 💬 대화 턴 수

#### 메인 화면
- 💬 대화 히스토리 표시
- 📝 메시지 입력창
- 🔍 상세 정보 (쿼리 타입 등)

---

## 🔮 향후 계획

### Phase 1: 모델 파인튜닝 (예정)
현재는 OpenAI GPT-4o-mini API를 사용하지만, 다음 단계로 **자체 파인튜닝 모델**로 교체 예정

#### 파인튜닝 데이터


#### 파인튜닝 방식


#### 교체 방법
각 노드의 LLM 호출 부분만 수정:
```python
# 현재 (API)
llm = ChatOpenAI(model="gpt-4o-mini")

# 파인튜닝 후
llm = load_finetuned_model("models/qa_finetuned.pth")
```

---

### Phase 2: 성능 개선
- [ ] Vector DB를 Persistent 모드로 변경 (디스크 저장)
- [ ] Embedding 모델 개선 (KoSimCSE → OpenAI Embedding)
- [ ] 더 많은 Few-shot 예시 활용 (1개 → 3개)
- [ ] 대화 요약 기능 추가 (긴 대화 압축)

---

### Phase 3: 배포 및 모니터링
- [ ] Docker 컨테이너화
- [ ] AWS/GCP 배포
- [ ] 로깅 및 모니터링 시스템
- [ ] A/B 테스트 (API vs 파인튜닝 모델)

---

## 📌 주의사항

### API 비용
- OpenAI API 사용으로 **비용 발생**
- 예상 비용 (GPT-4o-mini 기준):
  - 입력: $0.15 / 1M tokens
  - 출력: $0.60 / 1M tokens
  - 일일 예상: 약 100회 대화 시 $0.5-1 정도

### 데이터 개인정보
- 상담 데이터에서 개인정보는 `▲` 기호로 마스킹 처리됨
- 실제 운영 시 추가 보안 조치 필요

### 성능 제한
- 현재 메모리 기반 Vector DB (재시작 시 초기화)
- 동시 접속 사용자 제한 (Streamlit 특성상)
- 파인튜닝 모델 적용 전까지 API 의존성

---

## 👥 팀 정보

**팀명**: 상담랜드  
**프로젝트 기간**: 2025년 10월  
**기술 스택**: LangGraph, Streamlit, OpenAI, ChromaDB  


---

**Made with ❤️ by 상담랜드**