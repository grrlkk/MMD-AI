# DB/test_sigv4_check.py
import os, re, sys, json
from dotenv import load_dotenv

# 1) .env 로드 (프로젝트 루트/현재 폴더)
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "..", ".env"))
load_dotenv() # 현재 폴더도 시도

def masked(v: str, keep=4):
    if not v: return "<EMPTY>"
    return v[:keep] + "*" * max(0, len(v)-keep)

# 2) 환경변수 읽기
HOST = os.getenv("OPENSEARCH_HOST")
PORT = int(os.getenv("OPENSEARCH_PORT", "443"))
INDEX = os.getenv("OPENSEARCH_INDEX", "opensearch_job")

AK = os.getenv("AWS_OPENSEARCH_ACCESS_KEY_ID")
SK = os.getenv("AWS_OPENSEARCH_SECRET_ACCESS_KEY")
ST = os.getenv("AWS_SESSION_TOKEN") # 선택
REGION = os.getenv("AWS_REGION")

# 호스트에서 region 추출 fallback
if not REGION and HOST:
    m = re.search(r"\.(ap|us|eu|sa|ca|me|af)-[a-z0-9-]+-\d\.", HOST)
    if m:
        REGION = m.group(0).strip(".")[:-1] # 예: ap-northeast-2
if not REGION:
    REGION = "ap-northeast-2"

print("ENV CHECK ========")
print("HOST:", HOST)
print("PORT:", PORT)
print("INDEX:", INDEX)
print("AWS_REGION:", REGION)
print("ACCESS_KEY_ID:", masked(AK))
print("SECRET_ACCESS_KEY:", masked(SK))
print("SESSION_TOKEN:", "<SET>" if ST else "<NONE>")
print("===================\n")

# 3) 값 검증
missing = []
if not HOST: missing.append("OPENSEARCH_HOST")
if not AK: missing.append("AWS_OPENSEARCH_ACCESS_KEY_ID")
if not SK: missing.append("AWS_OPENSEARCH_SECRET_ACCESS_KEY")
if missing:
    print("❌ 누락된 환경변수:", ", ".join(missing))
    print("➡ .env 위치/키 이름/값을 확인하세요. (.env를 로드하려면 python-dotenv 필요)")
    sys.exit(1)

# 4) SigV4로 접속
from opensearchpy import OpenSearch, RequestsHttpConnection
from requests_aws4auth import AWS4Auth

awsauth = AWS4Auth(AK, SK, REGION, "es", session_token=ST)

client = OpenSearch(
    hosts=[{"host": HOST, "port": PORT}],
    http_auth=awsauth,
    use_ssl=True,
    verify_certs=True,
    connection_class=RequestsHttpConnection,
)

try:
    c = client.count(index=INDEX)["count"]
    print(f"📊 '{INDEX}' 문서 수:", c)
    res = client.search(index=INDEX, body={"query":{"match_all":{}}, "size":3})
    hits = res.get("hits", {}).get("hits", [])
    print(f"\n=== 예시 {len(hits)}건 ===")
    
    for i, h in enumerate(hits, 1):
        print(f"--- Document {i} ---")
        print("ID:", h["_id"])
        
        source_data = h.get("_source")
        if source_data:
            # ▼▼▼▼▼ 변경된 부분 ▼▼▼▼▼
            # 출력하기 전 'content_embedding' 필드를 제거
            source_data.pop("content_embedding", None)
            # ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲

            print(json.dumps(source_data, indent=2, ensure_ascii=False))
        else:
            print("_source: <NONE>")
        
        print("-" * 40)

except Exception as e:
    print("❌ 요청 실패:", repr(e))
    print("➡ 원인 예시: 네트워크/VPC 접근, 도메인 정책, 자격증명 권한, 인덱스 없음 등")