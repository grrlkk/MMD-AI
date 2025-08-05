import os
from dotenv import load_dotenv
from opensearchpy import OpenSearch, RequestsHttpConnection
from requests_aws4auth import AWS4Auth
import boto3
from db.logger import setup_logger

logger = setup_logger(__name__)
load_dotenv()
AWS_ACCESS_KEY_ID = os.getenv("AWS_OPENSEARCH_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_OPENSEARCH_SECRET_ACCESS_KEY")
AWS_REGION = os.getenv("AWS_REGION")
OPENSEARCH_HOST = os.getenv("OPENSEARCH_HOST")
OPENSEARCH_PORT = int(os.getenv("OPENSEARCH_PORT", "443"))
OPENSEARCH_INDEX = os.getenv("OPENSEARCH_INDEX", "wanted_job_new")


class OpenSearchDB:
    def __init__(self):
        self.access_key = AWS_ACCESS_KEY_ID
        self.secret_key = AWS_SECRET_ACCESS_KEY
        self.region = AWS_REGION
        self.host = OPENSEARCH_HOST
        self.port = OPENSEARCH_PORT
        self.index_name = OPENSEARCH_INDEX
        self._validate_environment()
        credentials = boto3.Session(aws_access_key_id=self.access_key, aws_secret_access_key=self.secret_key, region_name=self.region).get_credentials()
        self.awsauth = AWS4Auth(credentials.access_key, credentials.secret_key, self.region, 'es', session_token=credentials.token)
        self.client = OpenSearch(
            hosts=[{'host': self.host, 'port': self.port}], http_auth=self.awsauth, use_ssl=True,
            verify_certs=True, connection_class=RequestsHttpConnection, timeout=30,
            max_retries=3, retry_on_timeout=True
        )
        logger.info(f"OpenSearch class initialized. Host: {self.host}, Index: {self.index_name}")
    
    def _validate_environment(self):
        missing_vars = []
        if not AWS_ACCESS_KEY_ID: missing_vars.append("AWS_OPENSEARCH_ACCESS_KEY_ID")
        if not AWS_SECRET_ACCESS_KEY: missing_vars.append("AWS_OPENSEARCH_SECRET_ACCESS_KEY")
        if not AWS_REGION: missing_vars.append("AWS_REGION")
        if not OPENSEARCH_HOST: missing_vars.append("OPENSEARCH_HOST")
        if missing_vars: raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")
    
    def test_connection(self):
        try:
            health = self.client.cluster.health()
            logger.info(f"OpenSearch cluster health: {health['status']}")
            return True
        except Exception as e:
            logger.error(f"Connection test failed: {str(e)}")
            return False
    
    def create_index(self, index_name=None, mapping=None):
        """하이브리드 검색을 위한 OpenSearch 인덱스를 생성합니다."""
        if index_name is None:
            index_name = self.index_name
        
        # [수정] 하이브리드 검색용 매핑 및 설정으로 교체
        if mapping is None:
            mapping = {
                "settings": {
                    "index": { "knn": True, "knn.algo_param.ef_search": 100 }
                },
                "mappings": {
                    "properties": {
                        "embedding": {
                            "type": "knn_vector", "dimension": 1024,
                            "method": { "name": "hnsw", "space_type": "cosinesimil", "engine": "nmslib" }
                        },
                        "title": { "type": "text" }, "company_name": { "type": "keyword" },
                        "position_detail": { "type": "text" }, "main_tasks": { "type": "text" },
                        "qualifications": { "type": "text" }, "preferred_qualifications": { "type": "text" },
                        "tech_stack": { "type": "text" }, "location": { "type": "keyword" },
                        "experience_level": { "type": "keyword" }, "job_category": { "type": "keyword" },
                        "benefits": { "type": "text" }, "hiring_process": { "type": "text" }
                    }
                }
            }
        
        try:
            if not self.client.indices.exists(index=index_name):
                response = self.client.indices.create(index=index_name, body=mapping)
                logger.info(f"Successfully created hybrid index: {index_name}")
                return response
            else:
                logger.info(f"Index {index_name} already exists")
                return None
        except Exception as e:
            logger.error(f"Error creating index {index_name}: {str(e)}")
            raise
    
    def bulk_index_with_ids(self, documents, doc_ids, index_name=None):
        if index_name is None: index_name = self.index_name
        if len(documents) != len(doc_ids): raise ValueError("Documents and doc_ids must have the same length")
        actions = []
        for doc, doc_id in zip(documents, doc_ids):
            actions.append({"index": {"_index": index_name, "_id": doc_id}})
            actions.append(doc)
        try:
            response = self.client.bulk(body=actions, refresh=True)
            logger.info(f"Successfully bulk indexed {len(documents)} documents with IDs")
            return response
        except Exception as e:
            logger.error(f"Error bulk indexing documents with IDs: {str(e)}")
            raise

    def search(self, query, index_name=None, size=10):
        if index_name is None: index_name = self.index_name
        try:
            response = self.client.search(index=index_name, body=query, size=size)
            logger.info(f"Search completed. Found {response['hits']['total']['value']} documents")
            return response
        except Exception as e:
            logger.error(f"Error searching documents: {str(e)}")
            raise
    
    def delete_index(self, index_name=None):
        if index_name is None: index_name = self.index_name
        try:
            response = self.client.indices.delete(index=index_name)
            logger.info(f"Successfully deleted index: {index_name}")
            return response
        except Exception as e:
            logger.error(f"Error deleting index {index_name}: {str(e)}")
            raise