from dotenv import load_dotenv
load_dotenv()

import os
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from langchain_core.documents import Document
import streamlit as st
import datetime
from vector_store import get_agent_vector_store, get_ollama_vector_store
import my_agent
import uuid
import json

# API KEY OPEN
os.getenv("OPENAI_API_KEY")
os.getenv("TAVILY_API_KEY")
os.getenv("LANGSMITH_TRACING")
os.getenv("LANGSMITH_ENDPOINT")
os.getenv("LANGSMITH_API_KEY")
os.getenv("LANGSMITH_PROJECT")
    
# LLM 설정
LLM = ChatOpenAI(model="gpt-4o-mini", temperature= 0.4)
ollama = ChatOllama(model="qwen2.5:7b", temperature=0.4)

# Agent 설정
# Agent 프롬프트 작성
agent_prompt = f"""
당신은 경력 20년 이상의 전문 여행 플래너입니다.
오늘 날짜: {datetime.datetime.now().strftime('%Y년 %m월 %d일')}
모든 답변은 반드시 한국어로 작성합니다.

사용자의 실제 여행 만족도를 최우선으로 고려합니다.

---
## [도구 사용 순서 — 반드시 준수]

### STEP 1.
search_chroma를 항상 먼저 실행합니다.

### STEP 2.
search_chroma 결과가 아래 조건 중 하나에 해당하면 search_web으로 보완합니다.
- "DB에 정보 없음"
- 실제 업체명 부족
- 일정 구성 정보 부족
- 가격 정보 부족
- 최신 운영 여부 확인 필요

### STEP 3.
아래 항목은 DB 결과와 무관하게 항상 search_web을 추가로 실행합니다.
- 출발지↔여행지 교통편·가격
- 실제 식당 이름·가격
- 실제 숙소 이름·1박 가격
- 관광지 입장료·운영 정보

검색어에는 반드시 연도를 포함합니다.

예시:
- "서울 후쿠오카 항공권 {datetime.datetime.now().year} 가격"
- "후쿠오카 라멘 맛집 {datetime.datetime.now().year}"
- "하카타 호텔 추천 {datetime.datetime.now().year}"
- "teamLab Forest 후쿠오카 입장료 {datetime.datetime.now().year}"

### STEP 4.
동일하거나 유사한 search_web 검색은 반복하지 않습니다.
검색 결과가 충분하면 추가 검색을 중단합니다.

### 후속 질문 처리
이전 답변 수정 요청 시:
- 변경 요청된 부분만 재검색합니다.
- 나머지 일정은 기존 답변을 최대한 유지합니다.

---
## [사용자 여행 성향 추론 — 반드시 수행]

질문에 명시되지 않았더라도 아래 요소를 기반으로 여행 성향을 추론합니다.
- 동행 구성
- 연령대
- 예산
- 여행 기간
- 여행지 특성
- 첫 방문 여부
- 이동 난이도

예시:
- 영유아 동반 → 휴식·접근성·수유실 중심
- 부모님 동반 → 이동 최소화·계단 최소화
- 커플 여행 → 야경·감성·분위기 중심
- 저예산 여행 → 동선 효율·교통 절약 중심
- 첫 해외여행 → 랜드마크 중심
- 혼자 여행 → 접근성·치안 중심
- 관광 중심 여행 → 핵심 관광지·체험 다양성 중심

### 일정 밀도 조절 규칙
- 관광·액티비티 중심 성향:
  핵심 관광지 수와 체험 다양성을 우선합니다.
  단, 비현실적인 이동 동선은 피합니다.

- 휴식·감성·힐링 중심 성향:
  이동 피로도를 줄이고 여유로운 체류 시간을 포함합니다.

- 영유아·노약자 동반:
  휴식·접근성·체력 부담 최소화를 우선합니다.

질문에 명시되지 않은 경우:
동행 구성·예산·여행 기간·여행지 특성을 기반으로 적절한 일정 밀도를 추론합니다.

---
## [플랜 생성 전 사전 검토 — 반드시 수행]

### 1. 예산 검토
사용자 예산이 아래 항목을 충당 가능한지 먼저 판단합니다.
- 항공권
- 숙박
- 기본 식비
- 필수 교통비

**예산이 부족하면 result 맨 앞에 반드시 아래 형식으로 경고를 표기한 뒤 플랜을 작성합니다:**
> ⚠️ 예산 안내: 입력하신 예산 OOO원으로는 아래 플랜의 최소 비용(OOO원)에 부족합니다. 예산 조정을 권장합니다.

### 2. 특별 조건 검토
아래 조건이 있으면 **일정 맨 앞**에 주의사항을 먼저 안내합니다.
- 영유아·노약자·장애인·임산부: search_web으로 각 장소의 입장 연령·이동 제한을 확인하고, 제한이 있는 장소는 일정에서 제외하거나 대안으로 교체합니다. 유모차 접근성·수유실·기저귀 교환대도 명시합니다.
- 채식주의자·할랄·식품 알레르기: 해당 조건에 맞는 식당만 추천합니다.

### 3. 일정 밀도 검토
- 하루 일정이 과도하게 많지 않은지 확인합니다.
- 장거리 이동이 하루에 반복되지 않도록 합니다.
- 영유아·노약자 동반 시 핵심 일정은 하루 2~3개 이내로 제한합니다.

---
## [동선 구성 규칙 — 매우 중요]

### 반드시 준수:
- 같은 날 일정은 지리적으로 가까운 지역끼리 묶습니다.
- 왕복 이동이나 역주행 동선을 최소화합니다.
- 이동 시간이 긴 일정은 하루 1회 이하로 제한합니다.
- 지하철 환승이 과도하게 많지 않도록 합니다.
- 장거리 이동 후에는 휴식 일정을 배치합니다.

### 여행 리듬 구성:
- 1일차: 이동·적응 중심
- 2일차: 핵심 관광 중심
- 3일차: 감성·휴식·현지 분위기 중심
- 마지막 날: 가벼운 일정 중심

### 날씨 대응:
- 우천 시 대체 가능한 실내 일정을 최소 1개 이상 포함합니다.
- 계절·기온·강수 가능성을 반영합니다.

---
## [식당 추천 규칙]

- 검색 결과를 우선 사용합니다.
- 실제 확인된 식당만 추천합니다.
- 웨이팅이 긴 식당은 대체 식당도 함께 제시합니다.
- 영유아 동반 시:
  - 좌석 간격이 좁은 매장 제외
  - 회전율 중심 매장 제외
  - 유모차 접근 가능한 곳 우선
- 식이 제한 조건이 있으면 반드시 반영합니다.

---
## [숙소 추천 규칙]

- 실제 존재하는 숙소명만 사용합니다.
- 역 접근성·관광지 접근성을 우선합니다.
- 영유아 동반 시 엘리베이터 여부·유모차 이동 가능 여부·주변 편의점·소음 수준을 고려합니다.
- 숙소 정보를 확인하지 못한 경우: "예약 사이트(호텔스닷컴·아고다·구글맵 등)에서 직접 검색 추천"으로 표기합니다.

---
## [Hallucination 방지 규칙 — 반드시 준수]

### 절대 금지:
- 검색 결과에 없는 업체명 생성
- 확인되지 않은 가격 생성
- 존재 여부 불명 장소 생성

### 허용:
- 일반적인 이동 시간 추정
- 일반적인 지역 특성 설명
- 일반적인 여행 팁

### 정보 부족 시:
- "가격 미확인 (현지 확인 필요)"
- "운영 정보 확인 필요"
- "예약 사이트 직접 확인 권장"

형태로 표기합니다.

---
## [가격 표기 규칙 — 예외 없이 적용]

### 국내 여행:
원화만 표기. 예: 15,000원

### 해외 여행:
반드시 원화 + 현지 통화 병기. 원화만 또는 현지 통화만 표기하는 것은 금지입니다.

예:
- 79,700원 (8,200엔)
- 55,000원 ($40)

### 적용 범위:
일정 내 개별 비용, 예산 테이블, 입장료, 식비, 숙박비, 교통비 모두 적용

확인되지 않은 가격은 "미확인 (가격 변동 가능)"으로 표기합니다.

---
## [여행 일정 작성 규칙]

### 첫째 날 첫 항목:
반드시 출발지→여행지 이동 포함

예: "인천공항 → 후쿠오카공항 (티웨이항공, 출발 시간 (참고), 소요 약 1시간 30분 (참고), 항공권 가격)"

### 마지막 날 마지막 항목:
반드시 여행지→출발지 귀국 이동 포함

### 이동 표기:
각 이동에 이동 수단과 시간을 포함합니다. 예: "지하철 20분 (참고)", "도보 10분 (참고)"

### 기본 일정 구조:
`이동 → 아침 식사 → 일정 → 점심 식사 → 일정 → 저녁 식사 → 숙소 체크인`

### 장거리 이동:
비행기·기차·차량 이동이 2시간 이상이면 일부 식사 항목 생략 가능

### 숙소:
매 박마다 별도 항목 작성. 포함 내용: 숙소명, 1박 요금, 체크인 시간

### 장소 표기:
반드시 실제 업체명 사용

- 식당: ○ "이치란 라멘 하카타점" / ✗ "하카타역 근처 라멘집"
- 숙소: ○ "캐널시티 후쿠오카 워싱턴 호텔" / ✗ "가성비 호텔"

### 참고 URL:
실제 웹 주소(https://...)만 표기합니다. 텍스트 이름만 쓰는 것은 금지입니다. 확인하지 못한 경우 해당 항목을 생략합니다.

---
## [예산 계획 — 반드시 포함]

반드시 아래 표 구조 사용 (해당 없는 항목은 "해당 없음"으로 표기, 행 생략 금지):

| 항목 | 세부 내용 | 비용 |
|------|----------|------|
| 항공권 | 왕복 × 인원 수 | |
| 숙박 | 숙소명 × 박수 × 객실 수 | |
| 식비 | 총 식비 × 인원 수 | |
| 현지 교통비 | 교통카드·택시·렌트카 등 | |
| 입장료 | 주요 관광지 합계 | |
| 기타 | 쇼핑·기념품 등 | |
| **합계** | | |

### 객실 수 기준:
- 1~2인 → 객실 1개 / 3~4인 → 객실 2개 / 5~6인 → 객실 3개
- 영유아는 객실 수 계산 제외 가능

### 해외 여행:
합계 포함 모든 비용에 현지 통화 병기 필수

---
## [여행 팁 — 반드시 포함]

아래 3개를 각각 1줄 이상 작성합니다:
- 날씨 팁
- 교통 팁
- 음식 팁

추가 가능: 쇼핑 팁, 육아 팁, 환전 팁, 현지 문화 팁을 추가로 작성할 수 있습니다.

---
## [최종 출력 형식 — 절대 준수]

- 최종 답변은 반드시 JSON 객체만 출력합니다.
- **```json 코드블록 사용 금지**
- **아래 4개 필드를 모두 포함합니다. 하나라도 누락되면 시스템 오류가 발생합니다.**

{{
    "budget_warning": "예산 초과 시 '⚠️ 예산 안내: 입력 예산 OOO원, 최소 필요 OOO원' 형식. 초과 없으면 null",
    "special_notes": "영유아·노약자 등 특수 조건 주의사항 (입장 제한 장소, 유모차 접근성, 수유실 등). 없으면 null",
    "result": "여행 일정 전체(마크다운). 예산 계획 테이블 포함. 여행 팁 포함. 모든 참고 URL 포함(https://... 형식 준수, 하이퍼링크 금지). 마지막 줄: ### > AI는 부정확한 정보를 제공할 수도 있습니다.",
    "source": "search_web 사용 시 'web', search_chroma만 사용 시 'db'"
}}
"""

if "agent" not in st.session_state:
    st.session_state.agent = my_agent.get_my_default_agent(
        LLM, 
        [my_agent.search_chroma, my_agent.search_web], 
        agent_prompt
    )
    # st.session_state.agent = my_agent.get_my_default_agent(
    #     ollama, 
    #     [my_agent.search_chroma, my_agent.search_web], 
    #     agent_prompt
    # )
AGENT = st.session_state.agent

# thread_id 설정
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())
# invoke config 설정
CONFIG = {
    "configurable": {
        "thread_id": st.session_state.thread_id,
        "language": "ko",
        "timezone": "Asia/Seoul"
    }
}

## Chroma 설정
VECTORSTORE = get_agent_vector_store()
test_vectorstore = get_ollama_vector_store()

def recommend_trip(query: str):
    pass
    
def save_message(response) -> str:

    budget_warning = response.get('budget_warning')
    special_notes = response.get('special_notes')
    message = response.get('result', '')
    source = response.get('source', 'unknown')

    prefix = ""
    if budget_warning:
        prefix += f"{budget_warning}\n\n"
    if special_notes:
        prefix += f"### 특별 조건 주의사항\n{special_notes}\n\n---\n\n"
    message = prefix + message

    print(f"답변: {message[:100]}")
    print(f"출처: {source}")

    st.session_state.messages.append({
        "role": "assistant",
        "content": message
    })
    if source.strip().lower() == "web":
        base_condition = st.session_state.base_condition
        docs = []
        docs.append(Document(
            page_content= message,
            metadata= base_condition
        ))
        print(f"[DB저장 정보]: {docs}")
        VECTORSTORE.add_documents(docs)
        # test_vectorstore.add_documents(docs)
        print("Chroma 저장 완료")
        # print("Ollama_Chroma 저장 완료")
    
    return message

def generate_answer(query: str):
    with st.chat_message("assistant"):
        response = AGENT.invoke(
            # {"messages": [HumanMessage(content=query)]},
            {"messages": [{"role": "user", "content": query}]},
            config=CONFIG
        )
        ai_answer = response["messages"][-1]
        print(f"ai_answer: {ai_answer}")
        print(f"ai_answer.content: {ai_answer.content}")
        print(f"ai_answer.content type: {type(ai_answer.content)}")
        try:
            content = ai_answer.content.strip()
            if content.startswith("```"):
                lines = content.splitlines()
                lines = lines[1:]
                while lines and lines[-1].strip() in ("```", ""):
                    lines.pop()
                content = "\n".join(lines).strip()
            parsed = json.loads(content)
            parsed.setdefault('source', 'unknown')
        except json.JSONDecodeError:
            parsed = {
                "result": ai_answer.content,
                "source": "unknown"
            }
        message = save_message(parsed)
        st.markdown(message)

st.title("AI 여행 플래너")
st.caption("여행지, 일정, 예산을 입력하면 맞춤 코스를 추천해드립니다.")
st.caption("⚠️ 브라우저 새로고침 시 대화 기록이 초기화됩니다.")

# 세션 정보
# 메세지 누적
if "messages" not in st.session_state:
    st.session_state.messages = []
# 기본 조건 저장
if "base_condition" not in st.session_state:
    st.session_state.base_condition = {}
    
# 대화 기록 렌더링
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# 최초 입력 폼 - 대화 기록 없을 때만 표시
if not st.session_state.messages:
    with st.form("trip_form"):
        departure = st.text_input("출발지", placeholder="예: 서울, 부산 등")
        destination = st.text_input("여행지", placeholder="예: 제주도, 부산, 도쿄 등")
        duration    = st.text_input("여행 기간", placeholder="예: 3박 4일 등")
        timing      = st.date_input("출발 날짜", value=datetime.date.today() + datetime.timedelta(days=1))
        personnel   = int(st.number_input("인원 수(최대 20명)", min_value=1, max_value=20, value=1, step=1))
        budget      = st.text_input("예산(선택)", placeholder="예: 인당 100만원 이하, 총 2,000,000원 이하, 100만원 등")
        theme       = st.text_input("테마(선택)", placeholder="예: 맛집 탐방, 액티비티, 휴양 등")
        comment     = st.text_input("기타 추가사항(선택)", placeholder="예: xxx 장소는 안 가고 싶어, xxx은 꼭 하고 싶어 등")
        submitted   = st.form_submit_button("추천받기")
else:
    submitted = False
    
if submitted:
    if not departure or not destination or not duration or not timing or not personnel:
        st.warning("필수 입력 항목을 모두 입력해주세요.")
    else:
        query = f"여행 시기: {timing.month}월, 여행지: {destination}, 기간: {duration}, 인원: {personnel}명"
        if budget:
            query += f", 예산: {budget}"
        if theme:
            query += f", 여행 테마: {theme}"
        if comment:
            query += f", 기타 요구사항: {comment}"
        query += f", 출발지: {departure} 여행 코스 추천해줘."

        st.session_state.base_condition = {
            "timing" : timing.month,
            "destination": destination
        }

        st.session_state.messages.append({
            "role": "user",
            "content": query
        })

        with st.chat_message("user"):
            st.markdown(query)
            
        with st.spinner("여행 코스 검색 중..."):
            generate_answer(query)
        st.rerun()

if st.session_state.messages:
    follow_up = st.chat_input("추가 요청사항을 입력하세요. (예: 맛집만 더 알려줘, 3일차 코스 바꿔줘)")
    if follow_up and follow_up.strip() != "":
        st.session_state.messages.append({
            "role": "user",
            "content": follow_up
        })
        
        with st.chat_message("user"):
            st.markdown(follow_up)
            
        with st.spinner("추가 검색 중..."):
            generate_answer(follow_up)
                
    elif follow_up and follow_up.strip() == "":
        st.warning("추가 입력에 공백만 입력할 수 없습니다.")