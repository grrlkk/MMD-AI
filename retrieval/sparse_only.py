from db.opensearch import OpenSearchDB

def build_query(user_data: dict, top_k: int) -> dict:
    """핵심 정보 및 필터 조건으로 OpenSearch 쿼리를 생성합니다."""
    desired_job = user_data.get("preferences", {}).get("desired_job") or user_data.get("career", {}).get("job_category", "")
    tech_stack = user_data.get("skills", {}).get("tech_stack", [])
    
    should_clauses = []
    if desired_job:
        should_clauses.append({"multi_match": {"query": desired_job, "fields": ["title^3", "job_category^2"], "boost": 10}})
    if tech_stack:
        should_clauses.append({"multi_match": {"query": " ".join(tech_stack), "fields": ["tech_stack^3", "qualifications^2"], "boost": 5}})

    if not should_clauses:
        return {"query": {"match_all": {}}, "size": top_k}
          
    filter_clauses = []
    locations = user_data.get("preferences", {}).get("desired_location", [])
    if locations:
        processed_locations = [loc.split()[0] for loc in locations if loc]
        if processed_locations:
            filter_clauses.append({"terms": {"location": processed_locations}})

    years = user_data.get("career", {}).get("years", 0)
    experience_filter = ["경력", "신입/경력"] if years > 0 else ["신입", "신입/경력"]
    filter_clauses.append({"terms": {"experience_level": experience_filter}})

    search_query = {
        "query": {"bool": {"should": should_clauses, "minimum_should_match": 1, "filter": filter_clauses}},
        "size": top_k,
        "_source": False,
    }
    return search_query

def search(user_data: dict, top_k: int) -> list[tuple[str, float]]:
    opensearch = OpenSearchDB()
    search_query = build_query(user_data, top_k)
    response = opensearch.search(search_query)
    
    results = []
    for hit in response.get("hits", {}).get("hits", []):
        results.append((hit["_id"], hit["_score"]))
    
    return results