"""
LangGraph 구조 시각화
"""
from dotenv import load_dotenv
load_dotenv()

from graph import create_graph

print("=" * 60)
print("LangGraph 구조 시각화")
print("=" * 60)

# 그래프 생성
graph = create_graph()

# 1. Mermaid 다이어그램 생성 (텍스트) - xray=True로 내부 노드 표시
try:
    print("\n📊 Mermaid 다이어그램 (xray=True):")
    print("-" * 60)
    mermaid_code = graph.get_graph(xray=True).draw_mermaid()
    print(mermaid_code)
    print("-" * 60)

    # 파일로 저장
    with open("graph_diagram.md", "w") as f:
        f.write("```mermaid\n")
        f.write(mermaid_code)
        f.write("\n```")
    print("\n✅ graph_diagram.md 저장 완료")
    print("   (GitHub/Notion 등에서 Mermaid 렌더링 가능)")

except Exception as e:
    print(f"Mermaid 생성 실패: {e}")

# 2. PNG 이미지 생성 (playwright 사용) - xray=True로 내부 노드 표시
try:
    print("\n🖼️ PNG 이미지 생성 시도 (playwright 사용, xray=True)...")

    png_data = graph.get_graph(xray=True).draw_mermaid_png()

    # 파일로 저장
    with open("graph_diagram.png", "wb") as f:
        f.write(png_data)

    print("✅ graph_diagram.png 저장 완료")
    print("   파일 위치: ./graph_diagram.png")

except Exception as e:
    print(f"⚠️ PNG 생성 실패: {e}")
    print("   해결 방법:")
    print("   1. pip install playwright")
    print("   2. playwright install chromium")

# 3. 그래프 정보 출력 (xray=True로 내부 노드 포함)
print("\n📋 그래프 정보 (xray=True):")
print("-" * 60)
graph_info = graph.get_graph(xray=True)
print(f"노드 개수: {len(graph_info.nodes)}")
print(f"노드 목록: {list(graph_info.nodes.keys())}")
print(f"엣지 개수: {len(graph_info.edges)}")

print("\n🔗 노드 연결:")
for edge in graph_info.edges:
    source = edge.source
    target = edge.target
    print(f"  {source} → {target}")

print("\n" + "=" * 60)
print("시각화 완료!")
print("=" * 60)
