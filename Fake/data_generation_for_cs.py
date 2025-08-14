# data_generation_from_opensearch.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, re, json, uuid, random
from collections import defaultdict
from typing import Dict, List, Tuple, Optional

from langchain_core.prompts import PromptTemplate
from WorkFlow.config import get_llm
from DB.opensearch import OpenSearchDB

# ---------- 유틸 ----------
def format_hit_to_text(hit_source: dict) -> str:
    if not hit_source: return ""
    field_order_map = [
        ('title','직무'),('company_name','회사'),('job_category','직무 카테고리'),
        ('location','위치'),('career','경력'),('dead_line','마감일'),
        ('position_detail','포지션 상세'),('main_tasks','주요 업무'),
        ('qualifications','자격 요건'),('preferred_qualifications','우대 사항'),
        ('benefits','혜택 및 복지'),('hiring_process','채용 과정'),('url','채용공고 URL')
    ]
    lines = ["[document]"]
    for k,d in field_order_map:
        v = hit_source.get(k)
        if not v: continue
        if k in ['main_tasks','qualifications','preferred_qualifications','benefits'] and isinstance(v, list):
            lines.append(f"{d}:\n" + "\n".join(f"- {it}" for it in v))
        elif isinstance(v, list):
            lines.append(f"{d}: {', '.join(map(str, v))}")
        else:
            lines.append(f"{d}: {v}")
    return "\n\n".join(lines)

def extract_url_from_text(text: str) -> Optional[str]:
    m = re.search(r"채용공고 URL:\s*(.+)", text)
    return m.group(1).strip() if m else None

# ---------- 매핑 로드 ----------
def load_mapping_table(path: str) -> Dict[str, Dict[str, str]]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def flatten_code_to_role(mapping_json: Dict[str, Dict[str, str]]) -> Dict[int, str]:
    """
    {대분류코드: {소분류코드: 직무명}} → {소분류코드(int): 직무명}
    """
    code2role: Dict[int, str] = {}
    for big, sub in mapping_json.items():
        for code_str, role_name in sub.items():
            try:
                code2role[int(code_str)] = role_name
            except ValueError:
                # 숫자 아닌 코드가 있으면 string 그대로도 보관
                code2role[code_str] = role_name
    return code2role

def group_codes_by_big(mapping_json: Dict[str, Dict[str, str]]) -> Dict[str, List[int]]:
    """
    {대분류코드: [소분류코드(int), ...]}
    """
    group: Dict[str, List[int]] = {}
    for big, sub in mapping_json.items():
        arr = []
        for code_str in sub.keys():
            try: arr.append(int(code_str))
            except ValueError: pass
        group[big] = arr
    return group

# ---------- OpenSearch 샘플링(리트리버 X) ----------
def _random_sample_by_category_code(code: int, size: int, seed: Optional[int] = None, exclude_ids: List[str] = []) -> dict:
    seed_val = seed if seed is not None else random.randint(1, 10_000_000)
    
    query_body = {
        "size": size,
        "query": {
            "function_score": {
                "query": {"bool": {"filter": [{"term": {"job_category": code}}], "must_not": []}},
                "random_score": {"seed": seed_val}
            }
        },
        "_source": {"excludes": ["content_embedding"]}
    }

    if exclude_ids:
        query_body["query"]["function_score"]["query"]["bool"]["must_not"].append({"ids": {"values": exclude_ids}})

    return query_body

def collect_candidates_by_mapping(mapping_path: str,
                                  per_code: int = 4, # 이제 이 값은 각 소분류에서 한 번에 가져오는 개수입니다.
                                  max_total: int = 2000,
                                  seed: Optional[int] = None) -> Dict[str, Dict]:
    """
    매핑의 '소분류 카테고리 코드' 별로 랜덤 샘플링하여 후보 수집 (URL dedup)
    - 개발 직무(대분류 코드 '518')에 해당하는 공고만 수집하도록 수정됨.
    - 소분류별 공고 수에 비례하여 샘플링하도록 로직이 변경됨.
    """
    mapping_json = load_mapping_table(mapping_path)
    code2role = flatten_code_to_role(mapping_json)
    grouped = group_codes_by_big(mapping_json)

    db = OpenSearchDB()
    pool: Dict[str, Dict] = {}
    
    dev_codes = grouped.get('518', [])
    
    print(f"매핑 로드 완료: 총 소분류 {len(code2role)}개 코드")
    print(f"-> '개발' 직무(518)에 해당하는 소분류 {len(dev_codes)}개 코드만 처리합니다.")

    # 1. 각 소분류의 총 문서 수를 먼저 카운트합니다.
    code_counts = {}
    for code in dev_codes:
        try:
            body = {"query": {"bool": {"filter": [{"term": {"job_category": code}}]}}}
            count_resp = db.count(body) # db.count() 함수가 있다고 가정
            count = count_resp.get('count', 0)
            code_counts[code] = count
            print(f"코드 {code} (직무: {code2role.get(code, '기타')}) - 총 문서 수: {count}개")
        except Exception as e:
            print(f"[경고] 코드 {code} 문서 수 카운트 중 오류: {e}")
            code_counts[code] = 0

    total_docs_in_dev = sum(code_counts.values())
    print(f"-> '개발' 직무 전체 문서 수: {total_docs_in_dev}개")
    if total_docs_in_dev == 0:
        print("경고: 개발 직무의 문서가 OpenSearch에 없습니다.")
        return {}

    # 2. 총 목표 문서 수만큼 가중치 샘플링을 진행합니다.
    total_target = min(max_total, total_docs_in_dev)
    if not dev_codes:
        return {}

    # 샘플링할 소분류 코드와 가중치(문서 수) 리스트 생성
    sampling_codes = list(code_counts.keys())
    weights = [code_counts[code] for code in sampling_codes]

    # 이미 뽑힌 문서 ID를 추적하여 중복 방지
    sampled_doc_ids = set()

    print(f"\n총 {total_target}개의 문서를 가중치 기반으로 샘플링합니다.")

    while len(pool) < total_target:
        # 가중치에 따라 소분류 코드를 하나 선택
        chosen_code = random.choices(sampling_codes, weights=weights, k=1)[0]
        
        try:
            # 선택된 코드에서 문서 1개 가져오기 (이미 뽑힌 ID 제외)
            body = _random_sample_by_category_code(chosen_code, size=1, exclude_ids=list(sampled_doc_ids))
            resp = db.search(body, size=1)
            hits = (resp or {}).get("hits", {}).get("hits", [])
            
            if hits:
                h = hits[0]
                _id = h.get("_id", "")
                if _id not in sampled_doc_ids:
                    src = h.get("_source", {}) or {}
                    src["_mapped_role_name"] = code2role.get(chosen_code, "기타")
                    src["_mapped_role_code"] = chosen_code
                    text = format_hit_to_text(src)
                    url = extract_url_from_text(text) or src.get("url") or f"doc:{_id}"
                    
                    if url not in pool:
                        pool[url] = {"doc_id": _id, "source": src, "text": text}
                        sampled_doc_ids.add(_id)
                        print(f"  - 문서 추가: 코드 {chosen_code} (직무: {code2role.get(chosen_code, '기타')})")
            else:
                # 해당 카테고리에서 더 이상 뽑을 문서가 없으면 가중치를 0으로 설정
                print(f"  - 경고: 코드 {chosen_code}에서 더 이상 문서를 찾을 수 없습니다. 가중치를 0으로 설정합니다.")
                index = sampling_codes.index(chosen_code)
                weights[index] = 0
                if sum(weights) == 0:
                    print("모든 카테고리에서 문서를 모두 뽑았습니다. 샘플링을 종료합니다.")
                    break
        except Exception as e:
            print(f"[경고] 코드 {chosen_code} 샘플링 중 오류: {e}")
            
    print(f"\n최종 {len(pool)}개의 고유 후보 문서를 수집했습니다. (가중치 기반 랜덤)")
    return pool

# ---------- 직무별 균형 샘플링 ----------
def sample_diverse_documents_from_pool(pool_by_url: Dict[str, Dict], k: int) -> List[Dict]:
    """
    매핑된 직무명(_mapped_role_name) 기준으로 균형 샘플링
    """
    buckets = defaultdict(list)
    for item in pool_by_url.values():
        role = item["source"].get("_mapped_role_name") or "기타"
        buckets[role].append(item)

    print(f"{len(buckets)}개의 직무 그룹으로 분류 완료. (매핑 기반)")
    final: List[Dict] = []
    roles = list(buckets.keys())
    random.shuffle(roles)

    # 1차: 직무별로 1개씩
    for role in roles:
        if len(final) >= k: break
        if buckets[role]:
            final.append(buckets[role].pop(random.randrange(len(buckets[role]))))

    # 2차: 남은 슬롯 채우기
    if len(final) < k:
        remain = []
        for lst in buckets.values(): remain.extend(lst)
        random.shuffle(remain)
        final.extend(remain[:(k - len(final))])

    print(f"총 {len(final)}개의 문서 샘플링을 완료했습니다.\n")
    return final

# ---------- LLM ----------
QUESTION_GENERATION_PROMPT = PromptTemplate(
    input_variables=["document", "gold_doc_id", "gold_url"],
    template=r"""
당신은 주어진 채용 공고에 적합한 **'가상의 구직자'**입니다. 아직 이 공고의 내용을 보지 못한 상태에서, 자신의 스펙과 커리어 목표를 바탕으로 취업 챗봇에게 첫 질문을 생성해야 합니다.
아래 규칙을 반드시 지키세요.

[강제 규칙]
- 아래 [DOC]만 사용하여 가상의 구직자 프로필을 추론하고, 이 프로필에 가장 적합한 질문을 만드세요. 외부 지식 금지.
- 원문 문구를 그대로 복사하지 말고, **동의어/상위개념**으로 바꿔 쓰세요.
- 직무명/지역은 **일반화**하세요. (예: '메인넷 백엔드 개발자'→'백엔드 개발자', '강남구 테헤란로'→'서울 강남')
- 질문은 문서 내용을 요약하는 형태가 아니라, **자신의 스펙으로 어떤 직무를 찾고 있는지**를 나타내는 형태여야 합니다.
- 출력은 **아래 JSON 스키마만** 출력하세요. 그 외 텍스트 금지.

[금지 규칙]
- 가상의 구직자는 입력으로 들어온 공고의 내용을 모른다고 가정합니다.
- 질문에 아래와 같은 표현을 절대로 사용하지 마세요.
  - "이 회사", "귀사", "여기"
  - "이 공고", "이 채용", "이 포지션"
  - "이 직무", "담당 업무"
  - "자격 요건", "우대 사항", "필요 기술"

[골드 정보]
- gold_doc_id: {gold_doc_id}
- gold_url: {gold_url}

[DOC]
{document}

[출력 JSON 스키마]
{{
  "sincere_question": {{
    "candidate_major": "추론한 전공",
    "candidate_interest": "**[일반화된 직무명]**",
    "candidate_career": "추론한 경력",
    "candidate_tech_stack": ["추론한 핵심 기술 스택 리스트"],
    "candidate_location": "**[일반화된 위치]**",
    "candidate_question": "추론한 프로필을 바탕으로, 챗봇에게 던질 만한 구체적이고 자연스러운 질문 문장 (예: '저는 5년차 백엔드 개발자인데, 이런 스펙을 가진 사람이 지원할만한 공고가 있을까요?')"
  }},
  "insincere_question": {{
    "candidate_major": "",
    "candidate_interest": "**[일반화된 직무명]**",
    "candidate_career": "",
    "candidate_tech_stack": [],
    "candidate_location": "**[일반화된 위치]**",
    "candidate_question": "[일반화된 정보]의 핵심 키워드만 조합한 짧은 검색 질문"
  }}
}}
""".strip()
)

def _robust_json_from_text(text: str) -> Optional[dict]:
    text = re.sub(r"```(?:json)?\s*|\s*```", "", str(text)).strip()
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m: return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None

def _passes_simple_checks(q: dict) -> bool:
    s = str((q or {}).get("candidate_question", "")).strip()
    if len(s) < 8: return False
    if re.search(r"\b\d{2,4}[-.\s]?\d{3,4}[-.\s]?\d{4}\b", s): return False
    if re.search(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", s): return False
    return True

def generate_questions_with_llm(doc_text: str, gold_doc_id: str, gold_url: str) -> Tuple[Optional[Dict], Optional[Dict]]:
    if not doc_text: return None, None
    llm = get_llm()  # ← 여기서 실제 OpenAI 모델 사용 (프로젝트 설정에 따름)
    chain = QUESTION_GENERATION_PROMPT | llm
    resp = chain.invoke({"document": doc_text, "gold_doc_id": gold_doc_id, "gold_url": gold_url})

    data = _robust_json_from_text(getattr(resp, "content", resp))
    if not data:
        print("[오류] LLM이 유효한 JSON을 반환하지 않았습니다."); return None, None

    sincere_q = data.get("sincere_question"); insincere_q = data.get("insincere_question")
    if not (sincere_q and insincere_q):
        print(f"[오류] LLM 결과에 필요한 키가 없습니다: {data}"); return None, None
    if not _passes_simple_checks(sincere_q) or not _passes_simple_checks(insincere_q):
        print("[필터] 부적합한 질문(너무 짧거나 연락처/이메일 포함) 제거"); return None, None
    return sincere_q, insincere_q

# ---------- 메인 ----------
def main():
    NUM_SAMPLES = 300
    PER_CODE = 5                    
    FILE_NAME = "retriever_sample_data_forcs.json"
    MAPPING_PATH = r"C:/Users/PC/MMD-AI/Frontend/mapping_table.json"

    print("=== 1단계: 매핑 기반 랜덤 샘플링 (리트리버 미사용) ===")
    pool = collect_candidates_by_mapping(MAPPING_PATH, per_code=PER_CODE, max_total=4000, seed=None)

    print(f"=== 1.5단계: 직무 다양성 보장 샘플링 ({NUM_SAMPLES}개) ===")
    samples = sample_diverse_documents_from_pool(pool, k=NUM_SAMPLES)

    print(f"=== 2단계: LLM 질문 쌍 생성 시작 (총 {len(samples)}개 문서) ===")
    dataset = []
    for i, item in enumerate(samples, 1):
        gold_id = item["doc_id"]; src = item["source"]; doc_text = item["text"]
        gold_url = src.get("url", extract_url_from_text(doc_text) or "")
        print(f"  - {i}/{len(samples)}: doc_id={gold_id}  mapped_role={src.get('_mapped_role_name')} (code {src.get('_mapped_role_code')})")

        sincere_q, insincere_q = generate_questions_with_llm(doc_text, gold_id, gold_url)
        if sincere_q and insincere_q:
            dataset.append({"qid": f"q_{uuid.uuid4().hex[:12]}", "query": sincere_q,
                            "gold_doc_id": gold_id, "gold_url": gold_url, "document_text": doc_text})
            dataset.append({"qid": f"q_{uuid.uuid4().hex[:12]}", "query": insincere_q,
                            "gold_doc_id": gold_id, "gold_url": gold_url, "document_text": doc_text})

    print("\n=== 최종 데이터셋 생성 완료! ===")
    print(f"총 {len(dataset)}개의 (query, gold_doc) 데이터 포인트가 생성되었습니다.")
    with open(FILE_NAME, "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)
    print(f"\n{FILE_NAME} 파일로 저장되었습니다.")

if __name__ == "__main__":
    main()