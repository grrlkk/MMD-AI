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

print("모델 로딩 중...")
MODEL = SentenceTransformer("intfloat/multilingual-e5-large-instruct")
print("모델 로딩 완료.")

print("Pinecone 연결 중...")
pc = Pinecone(api_key=api_key, environment=env)
INDEX = pc.Index(index_name)
print("Pinecone 연결 완료.")


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
        print("❌ Error: conversation and profile fields are empty.")
        return
    prefix_query = f"[Query] {full_query}"

    # ========== 임베딩 ==========
    start = time.time()
    query_vec = MODEL.encode(prefix_query, normalize_embeddings=True).tolist()
    print(f"⏱ Embedding time: {time.time() - start:.3f} sec")

    # ========== 검색 ==========
    start = time.time()
    res = INDEX.query(
        vector=query_vec,
        top_k=args.top_k,
        include_metadata=True,
        metric="cosine"
    )
    print(f"⏱ Pinecone query time: {time.time() - start:.3f} sec")

    # ========== JSON 형식으로 정리 ==========
    scores = []
    docs = []

    for match in res.get("matches", []):
        score = match.get("score", 0)
        scores.append(score)

        text_full = match.get("metadata", {}).get("text", "")
        job_id = match.get("id")

        m = re.search(r"직무:\s*([^\n]+?)\s+회사:\s*([^\n]+)", text_full)
        role = m.group(1).strip() if m else "(Unknown)"
        comp = m.group(2).strip() if m else "(Unknown)"

        u = re.search(r"채용공고 URL:\s*(https?://\S+)", text_full)
        job_url = u.group(1) if u else "(No URL)"

        snippet = text_full
        snippet = re.sub(r"\[document\]\s*", "", snippet)
        snippet = re.sub(r"직무:[^\n]+", "", snippet)
        snippet = re.sub(r"회사:[^\n]+", "", snippet)
        snippet = snippet.replace("\n", " ").strip()
        if len(snippet) > 200:
            snippet = snippet[:200] + "..."

        docs.append({
            "id": job_id,
            "role": role,
            "company": comp,
            "url": job_url,
            "snippet": snippet,
            "score": score
        })

    result = {"scores": scores, "docs": docs}
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()