#!/usr/bin/env python3
import os
import re
import json
import argparse
import time
from dotenv import load_dotenv
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer

# ====== 전역 초기화 ======
load_dotenv()
api_key = os.getenv("PINECONE_API_KEY")
env = os.getenv("PINECONE_ENV")
index_name = os.getenv("PINECONE_INDEX")


print("Dense: 모델 로딩 중...")
model_load_start = time.time()
MODEL = SentenceTransformer("intfloat/multilingual-e5-large-instruct")
model_load_duration = time.time() - model_load_start
print(f"Dense: 모델 로딩 완료. (소요 시간: {model_load_duration:.3f}초)")

print("Dense: Pinecone 연결 중...")
pc = Pinecone(api_key=api_key, environment=env)
INDEX = pc.Index(index_name)
print("Dense: Pinecone 연결 완료.")


def build_full_query(user: dict) -> str:
    conv = user.get("conversation", "").strip()
    edu = user.get("education", {})
    level = edu.get("level")
    major = edu.get("major")
    car = user.get("career", {})
    years = car.get("years")
    cat = car.get("job_category")
    skills = user.get("skills", {}).get("tech_stack", [])
    salary = user.get("preferences", {}).get("desired_salary")

    parts = [conv]
    if level or major:
        parts.append(f"학력:{level or ''}({major or ''})".strip("():"))
    if years is not None or cat:
        parts.append(f"{years or ''}년차 {cat or ''}".strip())
    if skills:
        parts.append("기술:" + ", ".join(skills))
    if salary:
        parts.append(f"희망연봉:{salary}")

    return " | ".join(filter(None, parts))


def search(user_data: dict, top_k: int) -> list[tuple[str, float]]:
    """
    사용자 데이터를 기반으로 Pinecone에서 Dense 검색을 수행하고 (ID, 점수) 리스트를 반환합니다.
    """
    full_query = build_full_query(user_data)
    if not full_query:
        print("❌ Error: Dense Retriever - 검색할 내용이 없습니다.")
        return []
    prefix_query = f"query: {full_query}"

    # ========== 임베딩 ==========
    query_vec = MODEL.encode(prefix_query, normalize_embeddings=True).tolist()

    # ========== 검색 ==========
    res = INDEX.query(
        vector=query_vec,
        top_k=top_k,
        include_metadata=False, # 메타데이터는 필요 없으므로 False로 변경
        metric="cosine"
    )

    # ========== (ID, 점수) 형식으로 정리 ==========
    results = []
    for match in res.get("matches", []):
        job_id = match.get("id", "").replace("doc-", "")
        score = match.get("score", 0.0)
        if job_id:
            results.append((job_id, score))
            
    return results

def main():
    """테스트를 위한 메인 함수"""
    parser = argparse.ArgumentParser(description="Dense Retriever for job recommendations")
    parser.add_argument("user_json", help="Path to fake user JSON file")
    parser.add_argument("--top_k", type=int, default=10, help="Number of top results to return")
    args = parser.parse_args()

    # 유저 JSON 로딩
    with open(args.user_json, encoding="utf-8") as f:
        user = json.load(f)

    start = time.time()
    results = search(user, args.top_k)
    print(f"⏱ Dense search time: {time.time() - start:.3f} sec")

    # ========== JSON 형식으로 출력 ==========
    output = [{"id": doc_id, "score": score} for doc_id, score in results]
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()