import os
import time
from sentence_transformers import SentenceTransformer
from db.opensearch import OpenSearchDB

# ====== 전역 초기화 ======
print("Dense: 모델 로딩 중...")
model_load_start = time.time()
MODEL = SentenceTransformer("intfloat/multilingual-e-large-instruct")
model_load_duration = time.time() - model_load_start
print(f"Dense: 모델 로딩 완료. (소요 시간: {model_load_duration:.3f}초)")

def build_full_query(user_data: dict) -> str:
    """핵심 정보(희망 직무, 기술 스택) 중심으로 Dense 쿼리를 생성합니다."""
    desired_job = user_data.get("preferences", {}).get("desired_job") or user_data.get("career", {}).get("job_category", "")
    tech_stack = user_data.get("skills", {}).get("tech_stack", [])
    parts = []
    if desired_job: parts.append(f"희망 직무: {desired_job}")
    if tech_stack: parts.append(f"보유 기술: {', '.join(tech_stack)}")
    if not parts: return user_data.get("conversation", "").strip()
    return " | ".join(parts)

def build_knn_query(query_vector: list, top_k: int, filter_clauses: list) -> dict:
    """OpenSearch k-NN 쿼리를 생성합니다."""
    return {
        "size": top_k,
        "_source": False,
        "query": {
            "knn": {
                "embedding": {
                    "vector": query_vector,
                    "k": top_k,
                    "filter": { "bool": { "filter": filter_clauses } } if filter_clauses else None
                }
            }
        }
    }

def search(user_data: dict, top_k: int) -> list[tuple[str, float]]:
    """사용자 데이터를 기반으로 OpenSearch에서 필터가 적용된 k-NN 검색을 수행합니다."""
    full_query = build_full_query(user_data)
    if not full_query: return []
    prefix_query = f"query: {full_query}"
    query_vec = MODEL.encode(prefix_query, normalize_embeddings=True).tolist()

    filter_clauses = []
    locations = user_data.get("preferences", {}).get("desired_location", [])
    if locations:
        processed_locations = [loc.split()[0] for loc in locations if loc]
        if processed_locations:
            filter_clauses.append({"terms": {"location": processed_locations}})
    
    years = user_data.get("career", {}).get("years", 0)
    experience_filter = ["경력", "신입/경력"] if years > 0 else ["신입", "신입/경력"]
    filter_clauses.append({"terms": {"experience_level": experience_filter}})

    opensearch = OpenSearchDB()
    knn_query = build_knn_query(query_vec, top_k, filter_clauses)
    response = opensearch.search(knn_query)
    
    results = []
    for hit in response.get("hits", {}).get("hits", []):
        results.append((hit["_id"], hit["_score"]))
            
    return results