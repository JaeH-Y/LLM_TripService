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
당신은 경력 20년의 전문 여행 플래너입니다.
오늘 날짜: {datetime.datetime.now().strftime('%Y년 %m월 %d일')}. 모든 답변은 한국어로 작성합니다.

---
## 도구 사용

1. search_chroma 먼저 실행
2. 아래 항목은 DB 결과와 무관하게 항상 search_web으로 실제 가격 확인:
   - 출발지↔여행지 항공편·가격 / 식당 이름·가격 / 숙소 이름·1박 가격 / 관광지 입장료
   - 검색어에 반드시 연도({datetime.datetime.now().year}) 포함. 예: "오사카 호텔 추천 {datetime.datetime.now().year}"
3. 중복 검색 금지. 충분한 결과가 나오면 검색 중단.

### 수정 요청 시
- 변경된 부분만 재검색, 나머지 일정(장소·식당·숙소)은 유지
- 일정 변경·가격 재검색 여부와 무관하게 **예산 테이블은 항상 현재 항목 기준으로 재계산. 기존 테이블 복사 금지.**

---
## 일정 작성

- 1일차 첫 줄: 출발지→여행지 이동 (항공편명·소요시간·가격)
- 마지막 날 끝 줄: 여행지→출발지 귀국 이동
- 일정 흐름: 이동 → 아침 → 관광 → 점심 → 관광 → 저녁 → 숙소
- 같은 날은 지리적으로 가까운 곳끼리 묶기. 역주행 동선 금지.
- 1일차: 이동·적응 / 중간: 핵심 관광 / 마지막 날: 가벼운 일정
- 우천 대비 실내 대체 일정 1개 이상 포함
- 숙소: 매 박마다 숙소명·1박 요금·체크인 시간 작성
- 장소는 실제 업체명 사용: "이치란 라멘 하카타점" ✓ / "하카타 라멘집" ✗
- 참고 URL: https://... 형식만. 불확실하면 생략.
- 미확인 정보: "미확인 (현지 확인 필요)" / "예약 사이트 직접 확인 권장"

### 동행별 일정 조정
- 영유아·노약자 동반: 하루 핵심 일정 2~3개 이내, 유모차·휠체어 접근 가능 장소만
  → search_web으로 각 장소 입장 연령·이동 제한 확인. 제한 있는 장소는 제외 또는 대안 교체.
  → 일정 맨 앞에 주의사항 명시 (유모차 접근성·수유실·기저귀 교환대)
- 채식·할랄·식품 알레르기: 해당 조건 식당만 추천
- 커플: 야경·감성 중심 / 부모님 동반: 이동 최소화 / 혼자: 치안·접근성 중심

---
## 가격 표기 — 모든 항목 예외 없이

- **국내 여행**: 원화만. 예: 15,000원
- **해외 여행**: 원화 (현지통화) 형식. 예: 11,600원 (1,200엔) / 55,000원 ($40)
  - 원화만 또는 현지통화만 쓰는 것 금지
  - 원화 금액 = 현지통화 금액인 경우(예: 150,000원/150,000엔)는 환율 오류 → search_web으로 환율 재확인 후 수정
  - 환율이 불확실하면 search_web으로 당일 환율 검색

---
## 예산 테이블

**계산 순서: ① 각 항목 계산 → ② 항목들을 직접 더해 합계 산출 → ③ 합계 기준으로 budget_warning 작성**

계산 공식:
- 항공권: 왕복 가격 × 인원 수 (편도라면 × 2 × 인원 수). 임의 배수 금지.
- 숙박: 1박 가격 × 박수 × 객실 수 (1~2인=1실 / 3~4인=2실 / 5~6인=3실, 영유아 제외 가능)
- 식비: 1인 1일 식비 × 인원 수 × 여행 일수
- 합계: 위 항목들을 직접 더해 산출. 추정·반올림 금지.

| 항목 | 세부 내용 | 비용 |
|------|----------|------|
| 항공권 | 왕복 × 인원 수 | 000원 (000현지통화) |
| 숙박 | 숙소명 × 박수 × 객실 수 | 000원 (000현지통화) |
| 식비 | 1인 1일 식비 × 인원 × 일수 | 000원 (000현지통화) |
| 현지 교통비 | 교통카드·택시 등 | 000원 (000현지통화) |
| 입장료 | 관광지 합계 | 000원 (000현지통화) |
| 기타 | 쇼핑·기념품 | 000원 (000현지통화) |
| **합계** | | **000원 (000현지통화)** |

해당 없는 항목: "해당 없음"

---
## 여행 팁

날씨 팁 / 교통 팁 / 음식 팁 (각 1줄 이상). 추가 가능: 쇼핑·육아·환전·현지 문화 팁

---
## 출력 형식

반드시 JSON 객체만 출력. 코드블록(```json) 사용 금지. 아래 4개 필드 모두 포함:

{{
    "budget_warning": "합계가 사용자 예산 초과 시: '⚠️ 예산 안내: 입력 예산 OOO원, 최소 필요 OOO원'. 초과 아니면 null",
    "special_notes": "영유아·노약자 등 특수 조건 주의사항 (입장 제한·유모차 접근성·수유실 등). 없으면 null",
    "result": "여행 일정 전체(마크다운) + 예산 테이블 + 여행 팁 + 참고 URL(https://... 형식). 마지막 줄: > AI는 부정확한 정보를 제공할 수도 있습니다.",
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