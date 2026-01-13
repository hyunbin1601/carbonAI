"""RAG 검색 디버깅 스크립트"""

import asyncio
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from react_agent.rag_tool import get_rag_tool


def test_rag_search():
    """RAG 검색 상세 디버깅"""

    rag_tool = get_rag_tool()

    print("=" * 80)
    print("1. RAG 도구 상태 확인")
    print("=" * 80)
    print(f"RAG Available: {rag_tool.available}")
    print(f"Knowledge Base Path: {rag_tool.knowledge_base_path}")
    print(f"Chroma DB Path: {rag_tool.chroma_db_path}")
    print(f"Chroma DB exists: {rag_tool.chroma_db_path.exists()}")

    if rag_tool.vectorstore is None:
        print("\n❌ Vectorstore가 None입니다!")
        return

    print(f"Vectorstore loaded: {rag_tool.vectorstore is not None}")

    # 벡터 DB에 저장된 문서 수 확인
    try:
        collection = rag_tool.vectorstore._collection
        count = collection.count()
        print(f"Total documents in Chroma DB: {count}")
    except Exception as e:
        print(f"Document count error: {e}")

    print("\n" + "=" * 80)
    print("2. 테스트 쿼리 검색")
    print("=" * 80)

    query = "제4차 온실가스 거래 제도에 대해서 간단히 알려줘"
    print(f"Query: {query}\n")

    # 키워드 추출 확인
    print("-" * 80)
    print("키워드 추출:")
    print("-" * 80)
    keyword_query = rag_tool._extract_keywords(query)
    print(f"원본: {query}")
    print(f"키워드: {keyword_query}\n")

    # 원시 검색 결과 확인 (임계값 없이)
    print("-" * 80)
    print("원시 검색 결과 (상위 10개, 임계값 없음):")
    print("-" * 80)

    # 키워드로 검색
    print("\n[키워드로 검색]")
    keyword_results = rag_tool.vectorstore.similarity_search_with_score(keyword_query, k=10)

    for i, (doc, distance) in enumerate(keyword_results[:10], 1):
        # distance를 similarity로 변환 (rag_tool.py의 로직)
        if distance < 0:
            similarity = 1.0
        elif distance > 2.0:
            similarity = max(0.0, 1.0 - (distance / 2.0))
        else:
            similarity = 1.0 - distance

        filename = doc.metadata.get('filename', 'unknown')
        chunk_idx = doc.metadata.get('chunk_index', 0)
        content_preview = doc.page_content[:100].replace('\n', ' ')

        print(f"\n{i}. {filename} (chunk {chunk_idx})")
        print(f"   Distance: {distance:.6f}")
        print(f"   Similarity: {similarity:.6f}")
        print(f"   Content: {content_preview}...")

    # 원본 쿼리로도 검색
    if keyword_query.lower() != query.lower():
        print("\n\n[원본 쿼리로 검색]")
        original_results = rag_tool.vectorstore.similarity_search_with_score(query, k=10)

        for i, (doc, distance) in enumerate(original_results[:10], 1):
            if distance < 0:
                similarity = 1.0
            elif distance > 2.0:
                similarity = max(0.0, 1.0 - (distance / 2.0))
            else:
                similarity = 1.0 - distance

            filename = doc.metadata.get('filename', 'unknown')
            chunk_idx = doc.metadata.get('chunk_index', 0)
            content_preview = doc.page_content[:100].replace('\n', ' ')

            print(f"\n{i}. {filename} (chunk {chunk_idx})")
            print(f"   Distance: {distance:.6f}")
            print(f"   Similarity: {similarity:.6f}")
            print(f"   Content: {content_preview}...")

    # 실제 search_documents 호출 결과
    print("\n\n" + "=" * 80)
    print("3. search_documents 실제 결과 (threshold=0.7)")
    print("=" * 80)

    results = rag_tool.search_documents(query, k=3, similarity_threshold=0.7)

    if results:
        print(f"\n✓ {len(results)}개 문서 찾음:")
        for i, result in enumerate(results, 1):
            print(f"\n{i}. {result['filename']} (chunk {result['chunk_index']})")
            print(f"   Similarity: {result['similarity']:.6f}")
            print(f"   Content: {result['content'][:150]}...")
    else:
        print("\n❌ 유사도 0.7 이상인 문서를 찾지 못했습니다.")
        print("\n원인 분석:")
        print("- 임계값 0.7이 너무 높을 수 있습니다.")
        print("- 키워드 추출이 잘못되었을 수 있습니다.")
        print("- 벡터 임베딩 모델이 적합하지 않을 수 있습니다.")

    # 임계값 0.5로 다시 시도
    print("\n\n" + "=" * 80)
    print("4. 임계값 0.5로 재시도")
    print("=" * 80)

    results_05 = rag_tool.search_documents(query, k=3, similarity_threshold=0.5)

    if results_05:
        print(f"\n✓ {len(results_05)}개 문서 찾음:")
        for i, result in enumerate(results_05, 1):
            print(f"\n{i}. {result['filename']} (chunk {result['chunk_index']})")
            print(f"   Similarity: {result['similarity']:.6f}")
            print(f"   Content: {result['content'][:150]}...")
    else:
        print("\n❌ 유사도 0.5 이상인 문서도 없습니다.")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    test_rag_search()
