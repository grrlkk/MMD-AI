import json
import sys
import os

# 상위 폴더 경로 추가 (db.opensearch 임포트를 위해)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from db.opensearch import OpenSearchDB

# --- 설정 ---
INDEX_NAME = "wanted_job_new"
# 확인할 샘플 문서 개수
SAMPLE_SIZE = 5
# ----------------

def check_opensearch_data():
    """OpenSearch 인덱스의 총 문서 개수와 샘플 데이터를 확인합니다."""
    
    print(f"OpenSearch 인덱스 '{INDEX_NAME}'의 데이터 샘플(상위 {SAMPLE_SIZE}개)을 확인합니다...")
    
    try:
        opensearch = OpenSearchDB()
        
        query = {
            "size": SAMPLE_SIZE,
            "query": { "match_all": {} }
        }
        
        response = opensearch.search(query, size=SAMPLE_SIZE)
        
        total_docs = response.get("hits", {}).get("total", {}).get("value", 0)
        print(f"\n--- [인덱스 요약] ---")
        print(f"총 {total_docs}개의 문서가 '{INDEX_NAME}' 인덱스에 저장되어 있습니다.")
        print(f"----------------------")

        hits = response.get("hits", {}).get("hits", [])
        
        if not hits:
            print("\n결과 없음: 인덱스가 비어있습니다.")
            return
            
        print(f"\n✅ 그 중 {len(hits)}개의 샘플 문서는 아래와 같습니다.")
        
        for i, hit in enumerate(hits, 1):
            print(f"\n==================== [문서 {i}] ID: {hit['_id']} ====================")
            
            source_to_print = hit['_source'].copy()
            if 'embedding' in source_to_print:
                embedding_len = len(source_to_print['embedding'])
                source_to_print['embedding'] = f"[... {embedding_len}개의 숫자로 이루어진 벡터 ...]"

            print(json.dumps(source_to_print, indent=2, ensure_ascii=False))
        print("================================================================")

    except Exception as e:
        print(f"\n오류 발생: OpenSearch에 연결하거나 데이터를 조회하는 중 문제가 발생했습니다.")
        print(f"   (에러 상세: {e})")

if __name__ == "__main__":
    check_opensearch_data()