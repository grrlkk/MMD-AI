import sys
import os
import time
import boto3
import re
from typing import Dict, Any
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from opensearchpy.helpers import bulk
from boto3.dynamodb.types import TypeDeserializer

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from db.dynamodb import DynamoDB
from db.opensearch import OpenSearchDB
from db.logger import setup_logger

logger = setup_logger(__name__)

MODEL_NAME = "intfloat/multilingual-e5-large-instruct"

def extract_experience_level(qualifications: list) -> str:
    if not qualifications or not isinstance(qualifications, list): return "신입/경력"
    text = " ".join(qualifications)
    if "신입" in text and "경력" not in text: return "신입"
    if re.search(r"경력\s*\d+\s*년", text) or re.search(r"\d+\s*년\s*이상", text): return "경력"
    return "신입/경력"

def extract_location(location_text: str) -> str:
    if not location_text: return None
    return location_text.split()[0]

def get_doc_id_from_url(url: str) -> str:
    match = re.search(r'/wd/(\d+)', url)
    return match.group(1) if match else None

class DynamoToOpenSearchMigrator:
    def __init__(self, batch_size: int = 100):
        self.dynamodb = DynamoDB()
        self.opensearch = OpenSearchDB()
        self.batch_size = batch_size
        self.success_count = 0
        self.error_count = 0
        
        logger.info("Initializing embedding model...")
        self.embedding_model = SentenceTransformer(MODEL_NAME)
        logger.info("✅ Embedding model initialized.")
        logger.info(f"Migrator initialized with batch size: {batch_size}")

    def transform_to_hybrid_document(self, dynamo_item: Dict[str, Any]) -> Dict[str, Any]:
        text_to_embed = " ".join(filter(None, [
            dynamo_item.get("title", ""), dynamo_item.get("position_detail", ""),
            " ".join(dynamo_item.get("main_tasks", []) if isinstance(dynamo_item.get("main_tasks"), list) else [str(dynamo_item.get("main_tasks", ""))]),
            " ".join(dynamo_item.get("qualifications", []) if isinstance(dynamo_item.get("qualifications"), list) else [str(dynamo_item.get("qualifications", ""))]),
            " ".join(dynamo_item.get("preferred_qualifications", []) if isinstance(dynamo_item.get("preferred_qualifications"), list) else [str(dynamo_item.get("preferred_qualifications", ""))])
        ]))
        experience = extract_experience_level(dynamo_item.get("qualifications", []))
        location = extract_location(dynamo_item.get("location"))
        embedding_vector = self.embedding_model.encode(f"passage: {text_to_embed}", normalize_embeddings=True)

        return {
            "title": dynamo_item.get("title"), "company_name": dynamo_item.get("company_name"),
            "location": location, "experience_level": experience,
            "tech_stack": dynamo_item.get("tag_name"), "job_category": dynamo_item.get("job_category"),
            "position_detail": dynamo_item.get("position_detail"), "main_tasks": dynamo_item.get("main_tasks"),
            "qualifications": dynamo_item.get("qualifications"), "preferred_qualifications": dynamo_item.get("preferred_qualifications"),
            "benefits": dynamo_item.get("benefits"), "hiring_process": dynamo_item.get("hiring_process"),
            "embedding": embedding_vector.tolist()
        }

    # migrate.py의 create_actions_generator 함수를 이 코드로 교체

    def create_actions_generator(self):
        """DynamoDB에서 문서를 읽어 OpenSearch bulk action을 생성하는 제너레이터"""
        deserializer = TypeDeserializer()
        
        # [수정] boto3 클라이언트를 생성할 때, self.dynamodb에 저장된 region 정보를 명시적으로 전달
        dynamodb_client = boto3.client('dynamodb', region_name=self.dynamodb.region)
        paginator = dynamodb_client.get_paginator('scan')
        
        processed_count = 0

        for page in tqdm(paginator.paginate(TableName=self.dynamodb.table_name), desc="Scanning DynamoDB"):
            for item in page.get("Items", []):
                try:
                    doc = {k: deserializer.deserialize(v) for k, v in item.items()}
                    doc_id = get_doc_id_from_url(doc.get("url"))
                    if not doc_id:
                        self.error_count += 1
                        logger.warning(f"Skipping item due to missing URL/ID: {doc.get('title')}")
                        continue
                    
                    processed_count += 1
                    hybrid_doc = self.transform_to_hybrid_document(doc)
                    
                    yield {
                        "_op_type": "index",
                        "_index": self.opensearch.index_name,
                        "_id": doc_id, 
                        "_source": hybrid_doc
                    }
                except Exception as e:
                    logger.error(f"Error processing item. Error: {e}")
                    self.error_count += 1
    
    def migrate_all(self) -> Dict[str, int]:
        logger.info("Starting migration from DynamoDB to OpenSearch")
        
        self.opensearch.create_index()
        
        start_time = time.time()
        
        try:
            success, failed = bulk(
                self.opensearch.client,
                self.create_actions_generator(),
                chunk_size=self.batch_size,
                raise_on_error=False,
                request_timeout=60
            )
            self.success_count = success
            # failed는 오류가 발생한 action의 리스트이므로, 개수를 세어야 함
            self.error_count += len(failed)

        except Exception as e:
            logger.error(f"Error during bulk migration: {str(e)}")
            # 이 경우 처리된 문서가 없으므로 error_count를 전체로 잡기보다 로깅에 집중
            raise
        
        total_time = time.time() - start_time
        stats = {
            'total_migrated': self.success_count,
            'total_errors': self.error_count,
            'total_time_seconds': total_time,
            'migration_rate': self.success_count / total_time if total_time > 0 else 0
        }
        
        logger.info(f"Migration completed. Stats: {stats}")
        return stats

def main():
    """메인 실행 함수"""
    try:
        migrator = DynamoToOpenSearchMigrator(batch_size=100)
        stats = migrator.migrate_all()
        
        print("\n" + "="*50)
        print("마이그레이션 완료!")
        print("="*50)
        print(f"총 성공: {stats['total_migrated']}")
        print(f"총 오류: {stats['total_errors']}")
        print(f"총 소요 시간: {stats['total_time_seconds']:.2f}초")
        print(f"처리 속도: {stats['migration_rate']:.2f} 문서/초")
        print("="*50)
            
    except Exception as e:
        logger.error(f"Migration failed: {str(e)}")
        print(f"❌ 마이그레이션 실패: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()