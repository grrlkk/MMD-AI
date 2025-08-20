# evaluation_script.py
import json
import os
from collections import defaultdict
from hybrid_retriever import hybrid_search

def run_comparison_evaluation(data_path: str, top_k: int = 5):
    try:
        with open(data_path, 'r', encoding='utf-8') as f:
            dataset = json.load(f)
    except Exception as e:
        print(f"파일 처리 중 오류: {e}")
        return

    # 리랭킹 ON/OFF 시나리오별 성능 지표를 저장할 변수
    performance_metrics = {
        'with_reranker': {'hit_count': 0, 'reciprocal_rank_sum': 0},
        'without_reranker': {'hit_count': 0, 'reciprocal_rank_sum': 0}
    }
    total_queries = len(dataset)
    
    # 공유를 위한 상세 비교 로그를 저장할 리스트
    comparison_logs = []

    print(f"--- 리랭킹 ON/OFF 비교 평가 시작 (총 {total_queries}개 쿼리, top_k={top_k}) ---")

    for i, data_point in enumerate(dataset, 1):
        query_info = data_point['query']
        gold_doc_id = data_point['gold_doc_id']
        
        print(f"\n▶ 쿼리 {i}/{total_queries} 평가 중...")

        # 1. 리랭킹 OFF로 실행 (1차 리트리버 결과)
        scores_no_rerank, ids_no_rerank, docs_no_rerank = hybrid_search(user_profile=query_info, top_k=top_k, use_reranker=False)
        
        # 2. 리랭킹 ON으로 실행 (최종 결과)
        scores_with_rerank, ids_with_rerank, docs_with_rerank = hybrid_search(user_profile=query_info, top_k=top_k, use_reranker=True)

        # 3. 두 결과를 하나의 로그로 종합
        combined_log = {
            "qid": data_point.get("qid"),
            "query": query_info, # 전체 쿼리
            "gold_doc_id": gold_doc_id, # 정답 문서
            "without_reranker_results": [ # 리랭킹 안 한 결과 (리트리버 결과)
                {"rank": r+1, "doc_id": doc_id, "score": score, "title": doc.get("title")}
                for r, (score, doc_id, doc) in enumerate(zip(scores_no_rerank, ids_no_rerank, docs_no_rerank))
            ],
            "with_reranker_results": [ # 리랭킹 한 결과
                {"rank": r+1, "doc_id": doc_id, "score": score, "title": doc.get("title")}
                for r, (score, doc_id, doc) in enumerate(zip(scores_with_rerank, ids_with_rerank, docs_with_rerank))
            ]
        }
        comparison_logs.append(combined_log)

        # 4. 각 시나리오별 성능 지표 계산
        # 리랭킹 OFF
        rank_no_rerank = next((idx + 1 for idx, doc_id in enumerate(ids_no_rerank) if doc_id == gold_doc_id), 0)
        if rank_no_rerank > 0:
            performance_metrics['without_reranker']['hit_count'] += 1
            performance_metrics['without_reranker']['reciprocal_rank_sum'] += 1 / rank_no_rerank
            
        # 리랭킹 ON
        rank_with_rerank = next((idx + 1 for idx, doc_id in enumerate(ids_with_rerank) if doc_id == gold_doc_id), 0)
        if rank_with_rerank > 0:
            performance_metrics['with_reranker']['hit_count'] += 1
            performance_metrics['with_reranker']['reciprocal_rank_sum'] += 1 / rank_with_rerank
            
        print(f"  - 리랭킹 OFF 순위: {'미발견' if rank_no_rerank == 0 else rank_no_rerank}  |  리랭킹 ON 순위: {'미발견' if rank_with_rerank == 0 else rank_with_rerank}")

    # --- 최종 결과 요약 및 저장 ---
    summary_results = {}
    print("\n" + "="*80)
    print("⭐ 최종 종합 평가 결과: 리랭킹 적용 전/후 성능 비교 ⭐")
    print("="*80)

    for scenario, metrics in performance_metrics.items():
        hit_rate = metrics['hit_count'] / total_queries if total_queries > 0 else 0
        mrr = metrics['reciprocal_rank_sum'] / total_queries if total_queries > 0 else 0
        
        summary_results[scenario] = {'hit_rate': hit_rate, 'mrr': mrr}
        
        print(f"▷ 시나리오: {scenario.upper()}")
        print(f"  - Hit Rate (적중률): {hit_rate:.4f}")
        print(f"  - MRR (평균 역순위): {mrr:.4f}")
        print("-" * 80)
        
    # 요약 파일 저장
    summary_file_name = "evaluation_comparison_summary.json"
    with open(summary_file_name, "w", encoding="utf-8") as f:
        json.dump(summary_results, f, ensure_ascii=False, indent=2)
    print(f"\n✅ 비교 평가 요약이 '{summary_file_name}' 파일에 저장되었습니다.")

    # 상세 로그 파일 저장
    detailed_log_file_name = "evaluation_comparison_logs.json"
    with open(detailed_log_file_name, "w", encoding="utf-8") as f:
        json.dump(comparison_logs, f, ensure_ascii=False, indent=2)
    print(f"✅ 공유용 상세 비교 로그가 '{detailed_log_file_name}' 파일에 저장되었습니다.")

if __name__ == "__main__":
    dataset_file = "retriever_sample_data.json"
    run_comparison_evaluation(dataset_file, top_k=5)