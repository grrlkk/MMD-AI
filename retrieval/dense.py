#!/usr/bin/env python3
"""
빠르게 실행 가능한 Dense Retriever.
- SentenceTransformer 모델은 전역에서 한 번만 로딩
- Pinecone Index도 한 번만 연결
- 결과: Top-K 공고 ID, 회사, 직무, URL, 스니펫 등 출력

사용법:
  $ python new_run.py path/to/user.json --top_k 5
"""

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

print("모델 로딩 중...")
MODEL = SentenceTransformer("intfloat/multilingual-e5-large-instruct")
print("모델 로딩 완료.")

print("Pinecone 연결 중...")
pc = Pinecone(api_key=api_key, environment=env)
INDEX = pc.Index(index_name)
print("Pinecone 연결 완료.")


# ====== 유저 쿼리 생성 함수 ======
def build_full_query(user: dict) -> str:
    conv   = user.get("conversation", "").strip()
    edu    = user.get("education", {})
    level  = edu.get("level")
    major  = edu.get("major")
    car    = user.get("career", {})
    years  = car.get("years")
    cat    = car.get("job_category")
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

    return " | ".join(parts)


# ====== 메인 실행 ======
def main():
    parser = argparse.ArgumentParser(description="Dense Retriever for job recommendations")
    parser.add_argument("user_json", help="Path to fake user JSON file")
    parser.add_argument("--top_k", type=int, default=5, help="Number of top results to return")
    args = parser.parse_args()

    # 유저 JSON 로딩
    with open(args.user_json, encoding="utf-8") as f:
        user = json.load(f)

    full_query = build_full_query(user)
    if not full_query:
        print("Error: conversation and profile fields are empty.")
        return
    prefix_query = f"[Query] {full_query}"

    # ========== 임베딩 생성 ==========
    start = time.time()
    query_vec = MODEL.encode(prefix_query, normalize_embeddings=True).tolist()
    print(f"⏱ Embedding time: {time.time() - start:.3f} sec")

    # ========== Pinecone 검색 ==========
    start = time.time()
    res = INDEX.query(
        vector=query_vec,
        top_k=args.top_k,
        include_metadata=True,
        metric="cosine"
    )
    print(f"⏱ Pinecone query time: {time.time() - start:.3f} sec")

    # ========== 결과 출력 ==========
    print(f"\n=== TOP-{args.top_k} DENSE RESULTS for Query: '{full_query}' ===\n")

    for idx, match in enumerate(res.get("matches", []), start=1):
        job_id = match.get("id")
        text_full = match.get("metadata", {}).get("text", "")
        score = match.get("score", 0)

        # 직무, 회사 추출
        m = re.search(r"직무:\s*([^\n]+?)\s+회사:\s*([^\n]+)", text_full)
        role = m.group(1).strip() if m else "(Unknown)"
        comp = m.group(2).strip() if m else "(Unknown)"

        # URL 추출
        u = re.search(r"채용공고 URL:\s*(https?://\S+)", text_full)
        job_url = u.group(1) if u else "(No URL)"

        # 스니펫 준비
        snippet = text_full
        snippet = re.sub(r"\[document\]\s*", "", snippet)
        snippet = re.sub(r"직무:[^\n]+", "", snippet)
        snippet = re.sub(r"회사:[^\n]+", "", snippet)
        snippet = snippet.replace("\n", " ").strip()
        if len(snippet) > 200:
            snippet = snippet[:200] + "..."

        print(f"{idx}. {job_id} | URL: {job_url} | 직무: {role} | 회사: {comp} | Score:{score:.4f}")
        print(f"   {snippet}\n")


if __name__ == "__main__":
    main()