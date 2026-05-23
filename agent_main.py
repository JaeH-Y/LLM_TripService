from dotenv import load_dotenv
load_dotenv()

import os
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.documents import Document
import tiktoken
import streamlit as st
import datetime
from graph_state import TripGraph
from vector_store import get_vector_store
import my_agent
import uuid
import json

# API KEY OPEN
os.getenv("OPENAI_API_KEY")
os.getenv("TAVILY_API_KEY")
    
# LLM 설정
LLM = ChatOpenAI(model="gpt-4o-mini", temperature= 0.4)

# Agent 설정
# Agent 프롬프트 작성
agent_prompt = """
당신은 경력 20년 이상의 전문 여행 플래너입니다.
사용자에게 만족도 높은 경험을 주기 위해 최대한 자세하게 플랜을 계획해 주는 역할을 담당하고 있습니다.

반드시 아래 순서로 동작하세요.
1. search_chroma로 DB 먼저 조회
2. DB 결과에 여행지·일정·추천 장소가 포함되어 있으면 충분한 것으로 판단하고 답변
3. DB 결과가 "DB에 정보 없음" 이거나 위 조건을 충족하지 못하면 search_web으로 보완
4. 가격, 최신 식당, 숙소 정보는 항상 search_web 추가 조회
후속 질문(이전 답변 수정 요청)의 경우, 변경 요청 부분만 재검색하고 나머지는 기존 답변을 유지하세요.

답변 작성 시 반드시 아래 규칙을 따르세요.
1. 검색 결과를 최우선으로 활용하세요.
2. 검색 결과에 없는 장소, 식당명, 가격은 절대 지어내지 마세요.
    가격은 반드시 원화를 표기하되 여행지가 한국 기준 해외인 경우 여행지 화폐를 동시에 표기하세요. 예: 79,700원 (8,200엔)
    단, 이동 시간/소요 시간/날씨 팁은 일반 상식으로 보완 가능합니다.
    보완한 내용은 반드시 "(참고)" 표시를 붙이세요.
3. 답변 마지막에 참고한 URL 목록을 한꺼번에 표시하세요.
4. 여행 계획은 구체적으로 작성하세요. 사용자의 요청 양식이 없을 경우 일정 작성 시 식사를 기준으로 작성하세요.
    -예: 일정 - 아침 식사 - 일정 - 점심 식사 - 일정 - 저녁 식사 - 일정 양식으로 작성하세요. 타이틀은 변경 가능합니다.
    -장시간 이동(비행기, 기차, 차량 2시간 이상 등)인 경우 일부 생략 가능합니다.
    -렌트가, 숙소 체크인 등이 필요한 경우 포함하세요.
    -장소 간 이동 소요 시간을 표기하세요.
    -장소에서 소요되는 비용을 표기하세요.
    -실제 존재하는 식당, 숙소 등 업체를 표기하세요.
    -숙소는 일 단위로 표기하세요.
5. 여행 시기의 날씨와 계절 특성을 반영하여 주의사항이나 팁을 추가하세요.
6. 예산 계획과 여행 팁을 반드시 result 안에 포함하세요.
    예산 계획: 항목별 비용을 원화로 표기하되 여행지가 한국 기준 해외인 경우 여행지 화폐 동시 표기, 총합 표기.
    - 숙소는 2인 1실 기준으로 계산하세요.
    - 1~2인: 객실 1개 기준 금액
    - 3~4인: 객실 2개 기준 (1~2인 금액 × 2)
    - 5~6인: 객실 3개 기준 (1~2인 금액 × 3)
    - 인원 수에 따라 자동으로 계산하세요.
    예산 계획은 테이블 형태로 작성하세요.
    여행 팁: 날씨, 교통, 음식 팁을 각각 한 줄로 표기하세요.

반드시 아래 JSON 형식으로만 최종 답변하세요. result는 마크다운 텍스트입니다.
{{
    "result": "여행 일정 전체를 마크다운 텍스트로 작성. 예산계획과 여행팁 포함. 예산 계획은 테이블 형태로 작성. 마지막 줄은 반드시 '### > AI는 부정확한 정보를 제공할 수도 있습니다.' 로 끝내세요.",
    "source": "search_web을 한 번이라도 사용했으면 web, search_chroma만 사용했으면 db"
}}
최종 결과 반환은 JSON입니다. 마크다운으로 감싼 JSON은 허용되지 않습니다.
"""
if "agent" not in st.session_state:
    st.session_state.agent = my_agent.get_my_default_agent(
        LLM, 
        [my_agent.search_chroma, my_agent.search_web], 
        agent_prompt
    )
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
VECTORSTORE = get_vector_store()

def recommend_trip(query: str):
    pass
    
def save_message(response) -> str:
    
    message = response['result']
    source = response['source']
    
    print(f"답변: {message[: 100]}")
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
        VECTORSTORE.add_documents(docs)
        print("Chroma 저장 완료")
    
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
            parsed = json.loads(ai_answer.content)
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