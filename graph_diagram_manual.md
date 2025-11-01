# LangGraph 구조 (환경별 라우팅)

```mermaid
%%{init: {'flowchart': {'curve': 'linear'}}}%%
graph TD
    Start([시작]):::startClass
    ExtractMeta[extract_metadata<br/>메타데이터 추출]:::nodeClass
    Retrieve[retrieve_examples<br/>Vector DB 검색]:::nodeClass
    Router{route_by_environment<br/>환경 감지}:::routerClass
    VllmNode[vllm_counselor<br/>vLLM 답변 생성<br/><i>GPU + Prefix Cache</i>]:::gpuClass
    OllamaNode[ollama_counselor<br/>Ollama 답변 생성<br/><i>Mac/CPU + KV Cache</i>]:::cpuClass
    End([종료]):::endClass

    Start --> ExtractMeta
    ExtractMeta --> Retrieve
    Retrieve --> Router
    Router -->|GPU 환경| VllmNode
    Router -->|Mac/CPU 환경| OllamaNode
    VllmNode --> End
    OllamaNode --> End

    classDef startClass fill:#e1f5e1,stroke:#4caf50,stroke-width:2px
    classDef nodeClass fill:#e3f2fd,stroke:#2196f3,stroke-width:2px
    classDef routerClass fill:#fff9c4,stroke:#fbc02d,stroke-width:3px
    classDef gpuClass fill:#f3e5f5,stroke:#9c27b0,stroke-width:2px
    classDef cpuClass fill:#ffe0b2,stroke:#ff9800,stroke-width:2px
    classDef endClass fill:#fce4ec,stroke:#e91e63,stroke-width:2px
```

## 노드 설명

1. **extract_metadata**: 사용자 쿼리에서 도메인과 대화 턴 수 추출
2. **retrieve_examples**: Pinecone/ChromaDB에서 유사한 상담 예시 검색 (Few-shot)
3. **route_by_environment**: 환경 감지 후 라우팅
   - GPU 감지 → vllm_counselor로 이동
   - Mac/CPU 감지 → ollama_counselor로 이동
4. **vllm_counselor**: vLLM으로 답변 생성 (Prefix Caching)
5. **ollama_counselor**: Ollama로 답변 생성 (KV Cache)

## 주요 특징

- **조건부 라우팅**: `add_conditional_edges()` 사용
- **환경별 최적화**:
  - GPU: vLLM + Prefix Caching
  - Mac/CPU: Ollama + KV Cache
- **통합 프롬프트**: 두 경로 모두 동일한 `build_qa_prompt()` 사용
