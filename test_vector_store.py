"""
Vector Store 재생성 테스트 스크립트
"""
from dotenv import load_dotenv
load_dotenv()

from models import initialize_vector_store

print("=" * 60)
print("Vector Store 재생성 중...")
print("=" * 60)

# force_reload=True로 기존 데이터 무시하고 재생성
initialize_vector_store(force_reload=True)

print("\n" + "=" * 60)
print("✅ Vector Store 재생성 완료!")
print("=" * 60)
