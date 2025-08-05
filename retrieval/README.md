# Retrieval System

하이브리드 검색 시스템 구현 모듈입니다.

## 주요 기능
- Dense 검색 (임베딩 기반)
- Sparse 검색 (BM25)
- 하이브리드 검색 (Dense + Sparse)
- 검색 결과 재순위화

## 구성 요소
- `dense.py`: 임베딩 기반 검색
- `sparse.py`: BM25 기반 키워드 검색
- `hybrid.py`: 하이브리드 검색 구현 + Rerank
- `reranker.py`: Cross-Encoder 기반 재순위화

## 사용 예시
```python

python -m hybrid.py ../fake/fake_user/user_01.json --top_k N
# 검색 실행
results = ensemble_retriever(
    query="서울 지역 프론트엔드 개발자 채용",
    top_k=5
)
```

## 특징
- Dense와 Sparse 검색 결과의 앙상블
- 검색 결과 캐싱으로 성능 최적화
- Cross-Encoder 기반 정확한 순위 조정 


## Evaluation
- ../retrieval/eval.py
- "python -m retrieval.evaluation --alpha [알파값] --candidate_k [후보군_크기] --k [평가할_순위]"
- ../fake/retriever_evaluation_dataset_final.json으로 총 200개의 정답 데이터 활용
- nDCG@10, Recall@10으로 최적의 Alpha 값 선택 (Dense, Sparse 반영 비율 조절)