import datetime
import json
import os
import time

import requests
from bs4 import BeautifulSoup
from driver import get_chrome_driver
from dynamodb import save_job_to_dynamodb
from logger import get_logger
from selenium.common.exceptions import TimeoutException, NoSuchElementException, ElementClickInterceptedException
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait

logger = get_logger(__name__)

# 전역 카운터 변수들
NOT_ELEMENT_COUNT = 0
TIMEOUT_EXCEPTION_COUNT = 0
ELEMENT_CLICK_INTERCEPT_COUNT = 0
EXCEPTION_COUNT = 0

class Crawler:
    def __init__(
            self,
            data_path=os.path.join(os.getcwd(), "data_collection", "backup"),
            site_name: str = "wanted"
    ):
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:136.0) Gecko/20100101 Firefox/136.0",
            "Referer": "https://www.wanted.co.kr/",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "ko,en-US;q=0.9,en;q=0.8",
            "Connection": "keep-alive"
        }
        self.driver = get_chrome_driver()
        if not os.path.exists(data_path):
            os.mkdir(data_path)
        self.data_path = data_path
        self.filenames = {
                    "url_list": os.path.join(data_path, f"{site_name}.url_list.json"),
                }
        self.site_name = site_name
        self.endpoint = f"https://www.{site_name}.co.kr"

        # mapping_table.json에서 직업 카테고리 정보 로딩
            # mapping_table.json 파일에서 직업 카테고리 매핑 정보 로드
        with open("data_collection/crawler/mapping_table.json") as f:
            raw_mapping = json.load(f)
            
            # 직업 ID를 이름으로 매핑하는 딕셔너리 생성
            self.job_category_id2name = {}
            for parent_category, job_map in raw_mapping.items():
                for job_id, name in job_map.items():
                    self.job_category_id2name[int(job_id)] = name
                    
        logger.info(f"직업 카테고리 매핑 로드 완료: {len(self.job_category_id2name)}개 항목")
        


        # 섹션명을 필드명으로 매핑하는 딕셔너리
        self.map_section_to_field = {
            "포지션 상세": "position_detail",
            "주요업무": "main_tasks",
            "자격요건": "qualifications",
            "우대사항": "preferred_qualifications",
            "혜택 및 복지": "benefits",
            "채용 전형": "hiring_process",
            "기술 스택 • 툴": "tech_stack",
        }

    def requests_get(self, url: str) -> requests.Response:
        with requests.Session() as s:
            response = s.get(url, headers=self.headers)
        return response
    
    def run(self):
        """
        전체 크롤링 프로세스를 실행하는 함수
        """
        logger.info("=== 크롤링 프로세스 시작 ===")
        
        try:
            # 1. URL 리스트 수집
            logger.info("1단계: URL 리스트 수집 시작")
            job_dict = self.get_url_list()
            logger.info(f"URL 리스트 수집 완료: {len(job_dict)}개 카테고리")
            
            # 2. 채용공고 정보 크롤링 및 DB 저장
            logger.info("2단계: 채용공고 크롤링 및 DB 저장 시작")
            processed_count = self.crawling_job_info(job_dict)
            
            logger.info("=== 크롤링 프로세스 완료 ===")
            logger.info(f"총 처리된 채용공고: {processed_count}개")
            
            return processed_count
            
        except Exception as e:
            logger.error(f"크롤링 프로세스 중 오류 발생: {e}")
            raise
        finally:
            # 드라이버 종료
            if hasattr(self, 'driver') and self.driver:
                self.driver.quit()
                logger.info("Chrome driver 종료 완료")

    def scroll_down_page(self, driver) -> str:
        page_source = ""
        while True:
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(1)
            if page_source == driver.page_source:
                break
            else:
                page_source = driver.page_source

        return page_source

    def get_url_list(self):
        filename = self.filenames["url_list"]
        driver = self.driver

        job_dict = {}
        if os.path.exists(filename):
            with open(filename) as f:
                job_dict = json.load(f)

        with open("data_collection/crawler/mapping_table.json") as f:
            mapping_table = json.load(f)

        for job_parent_category, job_category_id2name in mapping_table.items():
            for job_category in job_category_id2name:
                if job_category in job_dict:
                    continue

                driver.get(
                    f"{self.endpoint}/wdlist/{job_parent_category}/{job_category}"
                )
                logger.info("job_category로 이동")

                logger.info("scroll_down_page 함수 호출 시작")
                page_source = self.scroll_down_page(driver)

                try:
                    soup = BeautifulSoup(page_source, "html.parser")
                    ul_element = soup.find("ul", {"data-cy": "job-list"})
                    position_list = [
                        a_tag["href"]
                        for a_tag in ul_element.find_all("a")
                        if a_tag.get("href", "").startswith("/wd/")
                    ]
                except Exception:
                    position_list = []

                job_dict[job_category] = {
                    "position_list": position_list,
                }

                with open(
                    os.path.join(self.data_path, f"{self.site_name}.url_list.json"), "w"
                ) as f:
                    logger.info("wanted.url_list.json 파일에 저장")
                    json.dump(job_dict, f)

        return job_dict
    
    def crawling_job_info(self, job_dict=None):
        """
        채용공고 정보를 크롤링하고 바로 DB에 저장하는 함수
        get_recruit_content_info와 postprocess 함수를 합친 버전
        """
        global NOT_ELEMENT_COUNT, TIMEOUT_EXCEPTION_COUNT, ELEMENT_CLICK_INTERCEPT_COUNT, EXCEPTION_COUNT
        
        logger.info("crawling_job_info 함수 실행 - 파일 저장 없이 바로 DB 저장")
        
        if job_dict is None:
            if os.path.exists(self.filenames["url_list"]):
                with open(self.filenames["url_list"]) as f:
                    job_dict = json.load(f)
            else:
                job_dict = {}
        
        driver = self.driver
        processed_count = 0

        for job_category, job_info in job_dict.items():
            logger.info(f"처리 중인 직업 카테고리: {job_category}")
            
            for position_url in job_info["position_list"]:
                try:
                    # 채용공고 페이지로 이동
                    driver.get(f"{self.endpoint}{position_url}")
                    time.sleep(1)

                    # 추가 정보를 위해 더보기 창 클릭 시도
                    try:
                        elements = driver.find_elements(By.XPATH, "//span[text()='상세 정보 더 보기']/ancestor::button")
                        logger.info(f"상세 정보 버튼 요소 개수: {len(elements)}")

                        if not elements: 
                            logger.info(f"{position_url} -> 버튼 요소 없음")
                            NOT_ELEMENT_COUNT += 1
                        else:
                            wait = WebDriverWait(driver, 5)
                            more_button = wait.until(
                                EC.element_to_be_clickable(
                                    (By.XPATH, "//span[text()='상세 정보 더 보기']/ancestor::button")
                                )
                            )
                            more_button.click()
                            logger.info(f"{position_url}의 상세 정보 더 보기 버튼 클릭")
                            time.sleep(1)

                    except TimeoutException:
                        logger.warning(f"{position_url} -> 버튼이 5초 내에 clickable 상태가 되지 않음")
                        TIMEOUT_EXCEPTION_COUNT += 1
                    except ElementClickInterceptedException:
                        logger.warning(f"🚫 {position_url} ▶️ 클릭 시 다른 요소에 가려짐")
                        ELEMENT_CLICK_INTERCEPT_COUNT += 1
                    except Exception as e:
                        logger.error(f"❌ {position_url} ▶️ 클릭 중 알 수 없는 오류: {e}")
                        EXCEPTION_COUNT += 1

                    # 페이지 소스 가져와서 바로 파싱
                    page_source = driver.page_source
                    soup = BeautifulSoup(page_source, "html.parser")

                    # 파싱 시작
                    # Job Title
                    job_title = (
                        soup.find("h1", class_="wds-jtr30u").text.strip()
                        if soup.find("h1", class_="wds-jtr30u")
                        else None
                    )

                    # Company Name and ID
                    company_name_element = soup.find(
                        "strong", class_="CompanyInfo_CompanyInfo__name__sBeI6"
                    )
                    company_name = (
                        company_name_element.text.strip() if company_name_element else None
                    )
                    company_link = soup.find(
                        "a", class_="JobHeader_JobHeader__Tools__Company__Link__NoBQI"
                    )
                    company_id = (
                        company_link["href"].split("/")[-1] if company_link else None
                    )

                    # Tags
                    tags_article = soup.find(
                        "article", class_="CompanyTags_CompanyTags__OpNto"
                    )
                    tag_name_list = []
                    tag_id_list = []
                    if tags_article:
                        tag_buttons = tags_article.find_all(
                            "button", class_="Button_Button__root__MS62F"
                        )
                        for tag_button in tag_buttons:
                            tag_name_span = tag_button.find("span", class_="wds-1m3gvmz")
                            if tag_name_span:
                                tag_name = tag_name_span.text.strip()
                                tag_id = tag_button.get("data-tag-id")
                                if tag_name and tag_id:
                                    tag_name_list.append(tag_name)
                                    tag_id_list.append(tag_id)

                    # Job Description
                    job_description_article = soup.find(
                        "article", class_="JobDescription_JobDescription__s2Keo"
                    )
                    detailed_content = {}
                    if job_description_article:
                        # Position Detail
                        position_detail_h2 = job_description_article.find(
                            "h2", class_="wds-qfl364"
                        )
                        if (
                            position_detail_h2
                            and position_detail_h2.text.strip() == "포지션 상세"
                        ):
                            position_detail_div = position_detail_h2.find_next_sibling(
                                "div",
                                class_="JobDescription_JobDescription__paragraph__wrapper__WPrKC",
                            )
                            if position_detail_div:
                                position_detail_span = position_detail_div.find(
                                    "span", class_="wds-wcfcu3"
                                )
                                if position_detail_span:
                                    position_detail_text = position_detail_span.get_text(
                                        separator="\n"
                                    ).strip()
                                    detailed_content["position_detail"] = position_detail_text

                        # Subsections
                        section_divs = job_description_article.find_all(
                            "div", class_="JobDescription_JobDescription__paragraph__87w8I"
                        )
                        for section_div in section_divs:
                            h3 = section_div.find("h3", class_="wds-1y0suvb")
                            if h3:
                                section_title = h3.text.strip()
                                content_span = section_div.find("span", class_="wds-wcfcu3")
                                if content_span:
                                    content_text = content_span.get_text(
                                        separator="\n"
                                    ).strip()
                                    content_lines = [
                                        line.strip()
                                        for line in content_text.split("\n")
                                        if line.strip()
                                    ]
                                    if content_lines and content_lines[0].startswith("•"):
                                        items = [
                                            line.lstrip("• ").strip()
                                            for line in content_lines
                                        ]
                                        detailed_content[
                                            self.map_section_to_field.get(
                                                section_title,
                                                section_title.lower().replace(" ", "_"),
                                            )
                                        ] = items
                                    else:
                                        detailed_content[
                                            self.map_section_to_field.get(
                                                section_title,
                                                section_title.lower().replace(" ", "_"),
                                            )
                                        ] = content_text

                    # 마감일
                    try:
                        deadline = soup.find("span", class_="wds-lgio6k").get_text()
                    except:
                        deadline = "no_data"
                    
                    # 위치
                    try:
                        location = soup.find("span", class_="wds-1o4yxuk").get_text()
                    except:
                        location = "no_data"

                    # 경력 정보 추출 (HTML 구조에 맞게 수정)
                    career_info = "no_data"
                    try:
                        # JobHeader 섹션에서 경력 정보 찾기
                        job_header_spans = soup.find_all("span", class_="JobHeader_JobHeader__Tools__Company__Info__b9P4Y wds-1pe0q6z")
                        
                        for span in job_header_spans:
                            span_text = span.text.strip()
                            if "경력" in span_text or "신입" in span_text:
                                career_info = span_text
                                logger.info(f"경력 정보 추출 성공: {career_info}")
                                break
                        
                        # 위에서 찾지 못한 경우 일반적인 방법으로 다시 시도
                        if career_info == "no_data":
                            # 경력 관련 키워드가 포함된 모든 span 태그 검색
                            career_elements = soup.find_all("span", string=lambda text: text and ("경력" in text or "신입" in text or "년차" in text))
                            if career_elements:
                                career_info = career_elements[0].text.strip()
                                logger.info(f"일반 검색으로 경력 정보 추출: {career_info}")
                                
                    except Exception as e:
                        logger.warning(f"경력 정보 추출 실패: {e}")

                    # 결과 구성
                    result = {
                        "url": f"https://www.wanted.co.kr{position_url}",
                        "crawled_at": datetime.datetime.utcnow().isoformat(),
                        "job_category": job_category,
                        "job_name": self.job_category_id2name.get(
                            int(job_category), job_category
                        ),
                        "title": job_title,
                        "company_name": company_name,
                        "company_id": company_id,
                        "tag_name": tag_name_list,
                        "tag_id": tag_id_list,
                        "dead_line": deadline,
                        "location": location,
                        "career": career_info,  # 경력 정보 추가
                        **detailed_content,
                    }

                    # DB에 바로 저장
                    save_job_to_dynamodb(result)
                    processed_count += 1
                    
                    logger.info(f"✅ {position_url} 처리 완료 - DB 저장 성공 (총 {processed_count}개 처리)")

                except Exception as e:
                    logger.error(f"❌ {position_url} 처리 중 오류 발생: {e}")
                    continue

        # 최종 통계 출력
        logger.info("=== 크롤링 완료 통계 ===")
        logger.info(f"총 처리된 채용공고: {processed_count}개")
        logger.info(f"버튼 없음: {NOT_ELEMENT_COUNT}")
        logger.info(f"Timeout: {TIMEOUT_EXCEPTION_COUNT}")
        logger.info(f"클릭 차단: {ELEMENT_CLICK_INTERCEPT_COUNT}")
        logger.info(f"기타 오류: {EXCEPTION_COUNT}")
        
        return processed_count
        