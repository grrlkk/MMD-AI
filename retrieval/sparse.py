import os
import sys
import json
import argparse
import time

# 경로 설정 (필요시 환경에 맞게 조정)
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from db.opensearch import OpenSearchDB


def build_query(user_data: dict, top_k: int) -> dict:
    """dense_retriever와 동일한 user_data 형식을 받아 OpenSearch 쿼리를 생성합니다."""
    # 사용자 데이터 추출
    interest = user_data.get("conversation", "")
    major = user_data.get("education", {}).get("major", "")
    job_category = user_data.get("career", {}).get("job_category", "")
    tech_stack = " ".join(user_data.get("skills", {}).get("tech_stack", []))
    
    # 검색어 조합 (관심분야와 직무 카테고리를 합쳐서 사용)
    main_query = f"{interest} {job_category}".strip()

    search_query = {
        "query": {
            "bool": {
                "should": [
                    {
                        "multi_match": {
                            "query": main_query,
                            "fields": ["job_category^3", "title^2", "position_detail", "main_tasks"],
                            "boost": 5
                        }
                    },
                    {
                        "multi_match": {
                            "query": tech_stack,
                            "fields": ["tech_stack^2", "preferred_qualifications", "qualifications"],
                            "boost": 2
                        }
                    },
                    {
                        "multi_match": {
                            "query": major,
                            "fields": ["qualifications^2", "preferred_qualifications"],
                            "boost": 1.5
                        }
                    }
                ],
                "minimum_should_match": 1
            }
        },
        "size": top_k,
        "_source": False, # ID와 점수만 필요
        "sort": [
            {"_score": {"order": "desc"}},
            {"_id": {"order": "asc"}}
        ]
    }
    return search_query

def search(user_data: dict, top_k: int) -> list[tuple[str, float]]:
    """
    사용자 데이터를 기반으로 OpenSearch에서 BM25 검색을 수행하고 (ID, 점수) 리스트를 반환합니다.
    """
    opensearch = OpenSearchDB()
    search_query = build_query(user_data, top_k)
    response = opensearch.search(search_query)
    
    # (ID, 점수) 형태로 변환
    results = []
    for hit in response.get("hits", {}).get("hits", []):
        results.append((hit["_id"], hit["_score"]))
    
    return results

def main():
    """테스트를 위한 메인 함수"""
    parser = argparse.ArgumentParser(description="BM25 Retriever for job recommendations")
    parser.add_argument("user_json", help="Path to fake user JSON file")
    parser.add_argument("--top_k", type=int, default=10, help="Number of top results to return")
    args = parser.parse_args()

    with open(args.user_json, 'r', encoding='utf-8') as f:
        user = json.load(f)

    start = time.time()
    results = search(user, args.top_k)
    print(f"⏱ BM25 search time: {time.time() - start:.3f} sec")

    output = [{"id": doc_id, "score": score} for doc_id, score in results]
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()