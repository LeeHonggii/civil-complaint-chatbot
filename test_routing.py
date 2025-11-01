"""
그래프 라우팅 테스트
"""
from dotenv import load_dotenv
load_dotenv()

print("=" * 60)
print("그래프 라우팅 테스트")
print("=" * 60)

# 1. 그래프 생성
print("\n1️⃣ 그래프 생성 중...")
from graph import create_graph

try:
    graph = create_graph()
    print("✅ 그래프 생성 완료")
except Exception as e:
    print(f"❌ 그래프 생성 실패: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# 2. 그래프 구조 확인
print("\n2️⃣ 그래프 구조 확인:")
print("-" * 60)
graph_info = graph.get_graph()
print(f"노드 개수: {len(graph_info.nodes)}")
print(f"노드 목록: {list(graph_info.nodes.keys())}")
print(f"\n엣지 개수: {len(graph_info.edges)}")
print("\n엣지 연결:")
for edge in graph_info.edges:
    print(f"  {edge.source} → {edge.target}")

# 3. 초기 상태로 테스트 (실제 실행은 하지 않음)
print("\n3️⃣ 라우팅 함수 테스트:")
print("-" * 60)

from graph import route_by_environment, GraphState

# 더미 상태 생성
dummy_state: GraphState = {
    "user_query": "테스트 쿼리",
    "conversation_history": [],
    "metadata": None,
    "retrieved_examples": None,
    "recent_context": [],
    "model_response": None,
    "session_id": "test",
    "error": None
}

print("\n환경 감지 및 라우팅 테스트:")
route_result = route_by_environment(dummy_state)
print(f"라우팅 결과: {route_result}")
print(f"  → {'vllm_counselor' if route_result == 'vllm' else 'ollama_counselor'} 노드로 이동")

print("\n" + "=" * 60)
print("테스트 완료!")
print("=" * 60)
