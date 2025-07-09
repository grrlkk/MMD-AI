import json
import argparse
import time
import sys
import os

# --- 설정값 ---
CANDIDATE_K = 100
ALPHA = 0.3
# -------------

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from . import dense
from . import sparse
from db.opensearch import OpenSearchDB


def normalize_scores(scores_dict: dict) -> dict:
    """Min-Max 정규화를 사용하여 점수를 0과 1 사이로 변환합니다."""
    if not scores_dict:
        return {}
    
    scores = list(scores_dict.values())
    min_score, max_score = min(scores), max(scores)

    if max_score == min_score:
        return {doc_id: 1.0 for doc_id in scores_dict} # 모든 점수가 같으면 1점으로 통일
        
    normalized = {
        doc_id: (score - min_score) / (max_score - min_score)
        for doc_id, score in scores_dict.items()
    }
    return normalized


def fetch_documents_by_ids(opensearch_client: OpenSearchDB, doc_ids: list[str]) -> dict[str, dict]:
    """주어진 문서 ID 리스트를 사용해 OpenSearch에서 전체 문서를 조회합니다."""
    if not doc_ids:
        return {}
    
    query = {"query": {"terms": {"_id": doc_ids}}, "size": len(doc_ids)}
    response = opensearch_client.search(query)
    return {hit['_id']: hit['_source'] for hit in response.get("hits", {}).get("hits", [])}


def print_reranked_results(final_results: list[dict], documents_map: dict):
    """최종 재정렬된 결과를 bm25_retriever.py와 동일한 상세 형식으로 출력합니다."""
    if not final_results:
        print("검색 결과가 없습니다.")
        return

    print("\n\n--- 최종 하이브리드 검색 결과 (재정렬 방식) ---")
    for i, result in enumerate(final_results, 1):
        doc_id = result['id']
        document = documents_map.get(doc_id)
        
        if not document:
            continue

        print(f"\n[ 최종 순위 {i} ] (ID: {doc_id})")
        print(f" 최종 Score: {result['final_score']:.4f}")
        print(f"  - Dense  점수: {result['dense_score']:.4f} (정규화: {result['norm_dense_score']:.4f})")
        print(f"  - Sparse 점수: {result['sparse_score']:.2f} (정규화: {result['norm_sparse_score']:.4f})")
        print("-" * 100)
        
        print(f"제목: {document.get('title', '정보 없음')}")
        print(f"회사명: {document.get('company_name', '정보 없음')}")
        print(f"직무: {document.get('title', '정보 없음')}")
        print(f"상세 내용: {document.get('position_detail', '정보 없음')}")
        print(f"주요 업무: {document.get('main_tasks', '정보 없음')}")
        print(f"자격요건: {document.get('qualifications', '정보 없음')}")
        print(f"우대사항: {document.get('preferred_qualifications', '정보 없음')}")
        print(f"혜택 및 복지: {document.get('benefits', '정보 없음')}")
        print(f"채용 전형: {document.get('hiring_process', '정보 없음')}")
        
        tech_stack_list = document.get('tech_stack', [])
        if isinstance(tech_stack_list, list):
             print(f"기술스택: {', '.join(tech_stack_list) if tech_stack_list else '정보 없음'}")
        else:
             print(f"기술스택: {tech_stack_list or '정보 없음'}")

        print(f"직무 카테고리: {document.get('job_category', '정보 없음')}")
        print(f"지역: {document.get('location', '정보 없음')}")
        print("-" * 100)

# --- [수정된 main 함수] ---
def main():
    parser = argparse.ArgumentParser(description="Advanced Hybrid Retriever with Re-ranking.")
    parser.add_argument("user_json", help="Path to the user JSON file.")
    parser.add_argument("--top_k", type=int, default=10, help="Final number of results to display.")
    args = parser.parse_args()

    with open(args.user_json, 'r', encoding='utf-8') as f:
        user_data = json.load(f)

    overall_start_time = time.time()

    # --- 1단계: 후보군 생성 ---
    stage1_start_time = time.time()
    print(f"🔍 1단계: 각 리트리버에서 Top {CANDIDATE_K} 후보군을 가져옵니다...")
    dense_results_raw = dense.search(user_data, top_k=CANDIDATE_K)
    sparse_results_raw = sparse.search(user_data, top_k=CANDIDATE_K)
    stage1_duration = time.time() - stage1_start_time
    print(f"   (1단계 소요 시간: {stage1_duration:.3f}초)")

    dense_results = {str(doc_id).replace("doc-", ""): score for doc_id, score in dense_results_raw}
    sparse_results = {str(doc_id).replace("doc-", ""): score for doc_id, score in sparse_results_raw}

    # --- 2단계: 재정렬 ---
    stage2_start_time = time.time()
    print("\n🔄 2단계: 점수 정규화 및 가중합으로 재정렬을 수행합니다...")
    
    all_candidate_ids = set(dense_results.keys()) | set(sparse_results.keys())
    print(f"-> 총 {len(all_candidate_ids)}개의 고유 후보군 생성 완료.")

    overlap_count = len(set(dense_results.keys()) & set(sparse_results.keys()))
    print(f"-> 그 중 {overlap_count}개가 Dense와 Sparse 목록에 공통으로 존재합니다.")
    
    norm_dense_scores = dense_results
    norm_sparse_scores = normalize_scores(sparse_results)

    fused_results = []
    for doc_id in all_candidate_ids:
        d_score = dense_results.get(doc_id, 0.0)
        s_score = sparse_results.get(doc_id, 0.0)
        norm_d_score = norm_dense_scores.get(doc_id, 0.0)
        norm_s_score = norm_sparse_scores.get(doc_id, 0.0)
        final_score = (ALPHA * norm_d_score) + ((1 - ALPHA) * norm_s_score)
        
        fused_results.append({
            "id": doc_id, "final_score": final_score, "dense_score": d_score,
            "sparse_score": s_score, "norm_dense_score": norm_d_score,
            "norm_sparse_score": norm_s_score
        })

    fused_results.sort(key=lambda x: x["final_score"], reverse=True)
    stage2_duration = time.time() - stage2_start_time
    print(f"   (2단계 소요 시간: {stage2_duration:.3f}초)")
    
    # --- 3단계: 결과 조회 및 출력 ---
    stage3_start_time = time.time()
    print("\n✅ 3단계: 최종 순위에 따라 상세 정보를 조회하고 출력합니다...")
    
    final_top_k_results = fused_results[:args.top_k]
    final_doc_ids = [res['id'] for res in final_top_k_results]

    opensearch_client = OpenSearchDB()
    documents_map = fetch_documents_by_ids(opensearch_client, final_doc_ids)

    print_reranked_results(final_top_k_results, documents_map)
    stage3_duration = time.time() - stage3_start_time
    print(f"   (3단계 소요 시간: {stage3_duration:.3f}초)")

    total_time = time.time() - overall_start_time
    print(f"\n⏱ 총 소요 시간: {total_time:.3f} 초")

if __name__ == "__main__":
    main()