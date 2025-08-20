# evaluation_script.py
import json
import os
from hybrid_retriever import hybrid_search

def run_full_comparison_evaluation(data_path: str, top_k: int = 5):
    try:
        with open(data_path, 'r', encoding='utf-8') as f:
            dataset = json.load(f)
    except Exception as e:
        print(f"파일 처리 중 오류: {e}")
        return

    # 4가지 시나리오(2x2)의 성능 지표를 모두 저장할 변수
    performance_metrics = {
        'sincere': {
            'without_reranker': {'hit_count': 0, 'reciprocal_rank_sum': 0},
            'with_reranker': {'hit_count': 0, 'reciprocal_rank_sum': 0},
            'total_queries': 0
        },
        'insincere': {
            'without_reranker': {'hit_count': 0, 'reciprocal_rank_sum': 0},
            'with_reranker': {'hit_count': 0, 'reciprocal_rank_sum': 0},
            'total_queries': 0
        }
    }
    
    comparison_logs = []
    print(f"--- 최종 비교 평가 시작 (총 {len(dataset)}개 쿼리, top_k={top_k}) ---")

    for i, data_point in enumerate(dataset, 0):
        query_info = data_point['query']
        gold_doc_id = data_point['gold_doc_id']
        
        # 1. 쿼리 유형 식별 (순서 기반)
        query_type = 'sincere' if i % 2 == 0 else 'insincere'
        performance_metrics[query_type]['total_queries'] += 1
        
        print(f"\n▶ 쿼리 {i+1}/{len(dataset)} 평가 중... (유형: {query_type.upper()})")

        # 2. 리랭킹 OFF/ON 시나리오 모두 실행
        scores_no_rerank, ids_no_rerank, _ = hybrid_search(user_profile=query_info, top_k=top_k, use_reranker=False)
        scores_with_rerank, ids_with_rerank, _ = hybrid_search(user_profile=query_info, top_k=top_k, use_reranker=True)

        # 3. 공유용 상세 로그 생성
        combined_log = {
            "qid": data_point.get("qid"),
            "query_type": query_type, # 성격 정보 추가
            "query": query_info,
            "gold_doc_id": gold_doc_id,
            "without_reranker_results": [{"rank": r+1, "doc_id": doc_id, "score": score} for r, (score, doc_id) in enumerate(zip(scores_no_rerank, ids_no_rerank))],
            "with_reranker_results": [{"rank": r+1, "doc_id": doc_id, "score": score} for r, (score, doc_id) in enumerate(zip(scores_with_rerank, ids_with_rerank))]
        }
        comparison_logs.append(combined_log)

        # 4. 4가지 시나리오에 대한 성능 지표 계산 및 업데이트
        # 리랭킹 OFF
        rank_no_rerank = next((idx + 1 for idx, doc_id in enumerate(ids_no_rerank) if doc_id == gold_doc_id), 0)
        if rank_no_rerank > 0:
            performance_metrics[query_type]['without_reranker']['hit_count'] += 1
            performance_metrics[query_type]['without_reranker']['reciprocal_rank_sum'] += 1 / rank_no_rerank
            
        # 리랭킹 ON
        rank_with_rerank = next((idx + 1 for idx, doc_id in enumerate(ids_with_rerank) if doc_id == gold_doc_id), 0)
        if rank_with_rerank > 0:
            performance_metrics[query_type]['with_reranker']['hit_count'] += 1
            performance_metrics[query_type]['with_reranker']['reciprocal_rank_sum'] += 1 / rank_with_rerank
            
        print(f"  - 결과: [리랭킹 OFF 순위: {'미발견' if rank_no_rerank == 0 else rank_no_rerank}] -> [리랭킹 ON 순위: {'미발견' if rank_with_rerank == 0 else rank_with_rerank}]")

    # --- 최종 결과 요약 및 저장 ---
    summary_results = {}
    print("\n" + "="*80)
    print("⭐ 최종 종합 평가 결과: 유저 성격 및 리랭킹 적용에 따른 성능 비교 ⭐")
    print("="*80)

    for q_type, type_metrics in performance_metrics.items():
        summary_results[q_type] = {}
        print(f"#### {q_type.upper()} QUERIES (총 {type_metrics['total_queries']}개) ####")
        
        for scenario, metrics in type_metrics.items():
            if scenario == 'total_queries': continue
            
            total = type_metrics['total_queries']
            hit_rate = metrics['hit_count'] / total if total > 0 else 0
            mrr = metrics['reciprocal_rank_sum'] / total if total > 0 else 0
            
            summary_results[q_type][scenario] = {'hit_rate': hit_rate, 'mrr': mrr}
            
            print(f"▷ 시나리오: {scenario}")
            print(f"  - Hit Rate (적중률): {hit_rate:.4f}")
            print(f"  - MRR (평균 역순위): {mrr:.4f}")
            print("-" * 40)
        print("\n")
        
    # 요약 파일 저장
    summary_file_name = "evaluation_final_summary.json"
    with open(summary_file_name, "w", encoding="utf-8") as f:
        json.dump(summary_results, f, ensure_ascii=False, indent=2)
    print(f"\n✅ 최종 비교 평가 요약이 '{summary_file_name}' 파일에 저장되었습니다.")

    # 상세 로그 파일 저장
    detailed_log_file_name = "evaluation_final_logs.json"
    with open(detailed_log_file_name, "w", encoding="utf-8") as f:
        json.dump(comparison_logs, f, ensure_ascii=False, indent=2)
    print(f"✅ 공유용 상세 비교 로그가 '{detailed_log_file_name}' 파일에 저장되었습니다.")


if __name__ == "__main__":
    dataset_file = "retriever_sample_data.json"
    run_full_comparison_evaluation(dataset_file, top_k=5)