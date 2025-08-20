import os
import sys
import torch

# 리랭킹
from sentence_transformers import CrossEncoder

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from DB.opensearch import OpenSearchDB
from langchain_huggingface import HuggingFaceEmbeddings

# --- 전역 모델 변수 ---
# 기존 전역 변수
_embedding_model = None

# 리랭킹 모델을 위한 전역 변수
_reranker_model = None

# --- 모델 로딩 함수 ---
def get_embedding_model():
    """임베딩 모델을 초기화하고 반환합니다 (싱글톤 패턴)"""
    global _embedding_model
    if _embedding_model is None:
        print("Initializing embedding model for hybrid search...")
        _embedding_model = HuggingFaceEmbeddings(
            model_name="intfloat/multilingual-e5-large",
            model_kwargs={'device': 'cuda' if torch.cuda.is_available() else 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        print(f"✅ Embedding model initialized on {'cuda' if torch.cuda.is_available() else 'cpu'}")
    return _embedding_model

# ▼▼▼▼▼ 리랭킹 모델을 불러오는 함수 ▼▼▼▼▼
def get_reranker_model():
    """CrossEncoder 리랭커 모델을 초기화하고 반환합니다 (싱글톤 패턴)"""
    global _reranker_model
    if _reranker_model is None:
        print("Initializing reranker model...")
        _reranker_model = CrossEncoder(
            'BAAI/bge-reranker-base',
            max_length=512,
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )
        print(f"✅ Reranker model initialized on {'cuda' if torch.cuda.is_available() else 'cpu'}")
    return _reranker_model


# --- 쿼리 생성 및 포맷팅 함수 ---
def build_hybrid_query(user_input: dict, top_k: int = 5, exclude_ids: list = None) -> dict:
    if exclude_ids is None:
        exclude_ids = []

    major = user_input.get("candidate_major", "")
    interest = user_input.get("candidate_interest", "")
    career = user_input.get("candidate_career", "")
    tech_stack = " ".join(user_input.get("candidate_tech_stack", []))
    location = user_input.get("candidate_location", "")
    query = user_input.get("candidate_question", "")

    embedding_model = get_embedding_model()
    query_vector = embedding_model.embed_query(f"query: {query}")

    must_not_clauses = []
    if exclude_ids:
        must_not_clauses.append({"ids": {"values": exclude_ids}})

    search_query = {
        "query": { "bool": {
                "should": [
                    { "multi_match": { "query": interest, "fields": ["job_name^3", "title^2", "position_detail"], "boost": 3.0 }},
                    { "multi_match": { "query": tech_stack, "fields": ["position_detail", "preferred_qualifications", "qualifications"], "boost": 2.5 }},
                    { "multi_match": { "query": f"{major}", "fields": ["qualifications", "preferred_qualifications"], "boost": 1.5 }},
                    { "match": {"career": {"query": career, "boost": 1.5}}},
                    { "match": {"location": {"query": location, "boost": 1.2}}} if location else None,
                    { "knn": { "content_embedding": { "vector": query_vector, "k": top_k * 2, "boost": 2.0 }}}
                ],
                "must_not": must_not_clauses,
                "minimum_should_match": 1
            }},
        "size": top_k,
        "_source": { "excludes": ["content_embedding"] }
    }
    search_query["query"]["bool"]["should"] = [q for q in search_query["query"]["bool"]["should"] if q is not None]
    return search_query


def _format_hit_to_text(hit_source: dict) -> str:
    if not hit_source:
        return ""
    field_order_map = [
        ('title', '직무'), ('company_name', '회사'), ('job_category', '직무 카테고리'),
        ('location', '위치'), ('career', '경력'), ('dead_line', '마감일'),
        ('position_detail', '포지션 상세'), ('main_tasks', '주요 업무'),
        ('qualifications', '자격 요건'), ('preferred_qualifications', '우대 사항'),
        ('benefits', '혜택 및 복지'), ('hiring_process', '채용 과정'), ('url', '채용공고 URL')
    ]
    lines = ["[document]"]
    for field_key, display_name in field_order_map:
        value = hit_source.get(field_key)
        if value:
            if field_key in ['main_tasks', 'qualifications', 'preferred_qualifications', 'benefits'] and isinstance(value, list):
                formatted_value = '\n'.join([f"- {item}" for item in value])
                lines.append(f"{display_name}:\n{formatted_value}")
            elif isinstance(value, list):
                lines.append(f"{display_name}: {', '.join(value)}")
            else:
                lines.append(f"{display_name}: {value}")
    return "\n\n".join(lines)


# --- 메인 검색 함수 (use_reranker 플래그 추가) ---
def hybrid_search(user_profile: dict, top_k: int = 5, exclude_ids: list = None, use_reranker: bool = True) -> tuple[list[float], list[str], list[dict]]:
    """
    하이브리드 검색을 수행하고, use_reranker 플래그에 따라 선택적으로 리랭킹을 적용합니다.
    """
    opensearch = OpenSearchDB()

    # 리랭커를 사용할 경우 더 많은 후보군(retrieval_k)을 가져오고, 사용하지 않으면 top_k만 가져옴
    retrieval_k = top_k * 5 if use_reranker else top_k
    
    print(f"🔍 1단계 (Retrieval): OpenSearch에서 후보군 {retrieval_k}개를 검색합니다. (리랭킹: {'ON' if use_reranker else 'OFF'})")
    try:
        search_query = build_hybrid_query(user_profile, retrieval_k, exclude_ids)
        response = opensearch.search(search_query, size=retrieval_k)
    except Exception as e:
        print(f"❌ 하이브리드 검색 실패: {e}")
        return [], [], []

    initial_hits = response.get("hits", {}).get("hits", [])
    if not initial_hits:
        return [], [], []
    print(f"✅ 1단계 (Retrieval) 완료: {len(initial_hits)}개의 결과를 가져왔습니다.")

    # --- 2단계: 리랭킹 (use_reranker가 True일 때만 실행) ---
    if use_reranker:
        print("\n🔄 2단계 (Reranking): 가져온 결과의 순위를 재조정합니다.")
        reranker = get_reranker_model()
        rerank_query = f"{user_profile.get('candidate_interest', '')} {user_profile.get('candidate_question', '')}"
        sentence_pairs = [[rerank_query, _format_hit_to_text(hit.get('_source', {}))] for hit in initial_hits]
        rerank_scores = reranker.predict(sentence_pairs, show_progress_bar=False)
        reranked_results = sorted(zip(rerank_scores, initial_hits), key=lambda x: x[0], reverse=True)
        
        final_results = reranked_results[:top_k]
        
        scores = [float(score) for score, hit in final_results] # score 타입을 float로 통일
        documents = [hit.get("_source", {}) for score, hit in final_results]
        doc_ids = [hit.get("_id", "") for score, hit in final_results]
        
        print(f"✅ 2단계 (Reranking) 완료: 최종 {len(scores)}개의 결과가 선택되었습니다.")
        return scores, doc_ids, documents
    
    # --- 리랭킹을 사용하지 않을 경우 ---
    else:
        # OpenSearch의 점수와 결과를 그대로 반환
        scores = [hit.get("_score", 0.0) for hit in initial_hits]
        documents = [hit.get("_source", {}) for hit in initial_hits]
        doc_ids = [hit.get("_id", "") for hit in initial_hits]
        return scores, doc_ids, documents


# --- 실행 부분 ---
if __name__ == "__main__":
    base_user_info = {
        "user_id": 10,
        "candidate_major": "경영학",
        "candidate_interest": "서비스 기획자",
        "candidate_career": "5년",
        "candidate_tech_stack": [
            "UX/UI 설계", "데이터 분석", "A/B 테스트", "프로젝트 관리"
        ],
        "candidate_location": "서울 강남",
        "candidate_question": "안녕하세요, 데이터 기반의 의사결정을 중요하게 여기는 성장하는 스타트업에서 서비스 기획자 직무에 적합한 포지션이 있을까요?"
    }

    print("\n=== 하이브리드 검색 + 리랭킹 결과 ===")
    scores, doc_ids, documents = hybrid_search(base_user_info, top_k=5)

    if not scores:
        print("검색 결과가 없습니다.")
    else:
        for i, (score, doc_id, document) in enumerate(zip(scores, doc_ids, documents), 1):
            print(f"\n[최종 순위 {i}] Rerank Score: {score:.4f}, 문서 ID: {doc_id}")
            # print(document) # 전체 문서를 보려면 주석 해제
            print(f"  제목: {document.get('title', '정보 없음')}")
            print(f"  회사명: {document.get('company_name', '정보 없음')}")
            print(f"  지역: {document.get('location', '정보 없음')}")
            print(f"  주요 업무: {document.get('main_tasks', '정보 없음')}")
            print("-" * 50)