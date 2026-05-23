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

# API KEY OPEN
os.getenv("OPENAI_API_KEY")
os.getenv("TAVILY_API_KEY")
    
# DB 설정
vectorstore = get_vector_store()
    
# LLM 설정
LLM = ChatOpenAI(model="gpt-4o-mini", temperature= 0.4)

# Tavily 검색 설정
SEARCH = TavilySearch(max_result=5, include_raw_content=True) # raw 데이터 없으면 LLM이 데이터 만들어냄

# 프롬프트
PROMPT = ChatPromptTemplate.from_messages([
    ("system", 
    """
        당신은 경력 20년 이상의 전문 여행 플래너입니다.
        사용자에게 만족도 높은 경험을 주기 위해 최대한 자세하게 플랜을 계획해 주는 역할을 담당하고 있습니다.
        반드시 아래 규칙을 따르세요.
        1. 제공된 검색 결과를 최우선으로 활용하세요.
        2. 검색 결과에 없는 장소, 식당명, 가격은 절대 지어내지 마세요.
            가격은 반드시 원화를 표기하되 여행지가 한국 기준 해외인 경우 여행지 화폐를 동시에 표기하세요. 예: 79,700원 (8,200엔)
            단, 이동 시간/소요 시간/날씨 팁은 일반 상식으로 보완 가능합니다.
            보완한 내용은 반드시 "(참고)" 표시를 붙이세요.
        3. 답변 마지막에 참고한 URL 목록을 한꺼번에 표시하세요.
        4. 여행 계획은 구체적으로 작성하세요.
        5. 여행 시기의 날씨와 계절 특성을 반영하여 주의사항이나 팁을 추가하세요.
        6. 답변 마지막에 아래 형식으로 예산 계획과 여행 팁을 반드시 표기하세요.
            예산 계획은 항상 재계산 하세요.
            예산 계획과 여행 팁은 별도 JSON 키가 아닌 result 안에 포함해서 작성하세요.
            예산 계획: 항목별 비용을 원화로 표기하되 여행지가 한국 기준 해외인 경우 여행지 화폐 동시 표기하고 총합을 표기하세요.
            예: 항공권: xxx원
            여행지가 해외인 경우 추가 표기 (xxx 여행지 화폐단위)
            여행 팁: 날씨, 교통, 음식 팁을 각각 한 줄로 표기하세요.
        반드기 아래 JSON 형식으로만 답하세요. result는 마크다운 텍스트입니다.
        {{
            "result": "여행 일정 전체를 마크다운 텍스트로 작성. 예산계획과 여행팁 포함. 예산 계획은 테이블 형태로 마크다운 텍스트 작성. 마지막 줄은 반드시 '##> AI는 부정확한 정보를 제공할 수도 있습니다.' 로 끝내세요.",
            "compression": "이 답변의 핵심 요약 (300자 이내). 반드시 포함: 여행지/일정/예산/사용자가 요청한 제외, 추가 조건 및 특별 요구사항"
        }}
        result 안에 절대 JSON을 넣지 마세요. 마크다운 텍스트만 허용합니다.    
    """),
    ("system", "이전 대화 기록:\n{chat_history}"),
    ("user", "검색 결과:\n{search_results}\n\n질문:\n{query}")
])

# 세션 데이터 관리
# 첫 여행 계획 저장
if "base_query" not in st.session_state:
    st.session_state.base_query = ""
if "meta" not in st.session_state:
    st.session_state.meta = {}
# 대화 전체 저장
if "messages" not in st.session_state:
    st.session_state.messages = []
# 대화 전체 요약 저장
if "history" not in st.session_state:
    st.session_state.history = []
# 대화 최근 요약 저장(LLM 활용)
if "current_history" not in st.session_state:
    st.session_state.current_history = ""
if "last_plan" not in st.session_state:
    st.session_state.last_plan = ""
if "warning_msg" not in st.session_state:
    st.session_state.warning_msg = ""

def recommend_trip(query: str):
    
    # Chroma 검색
    base_condition = st.session_state.meta
    base_condition_str = " ".join(f"{k}: {v}" for k, v in base_condition.items())
    search_query = f"{base_condition_str}\n{query}"
    print(f"[DB 검색 쿼리]: {search_query}")
    search_db = TripGraph()
    chroma_result = search_db.invoke(search_query)
    print(f"[DB 검색 내용]: {chroma_result}")
    chroma_answer = chroma_result['answer']
    if chroma_result.get('evaluate') != "no_result":
        if needs_search(query, chroma_answer) is False:
            print("[DB 저장 내용 재활용]")
            # PROMPT와 같은 형식으로 만들어 온 다음 검사받음
            return chroma_answer
        
    # 웹 검색
    print("[DB 내용 부족, 웹 검색 시작]")
    print(f"DB 조회 내용 검사: {chroma_answer}")
    with st.spinner("웹 검색 중..."):
        result = SEARCH.invoke(search_query)

        search_results = "\n\n".join([
            f"제목: {rs['title']}\nURL: {rs['url']}\n내용: {rs['raw_content'] or rs['content']}"
            for rs in result['results']
        ])

        use_token = cal_token(search_results)
        if use_token > 100000:
            st.session_state.messages.pop(len(st.session_state.messages) -1)
            st.session_state.warning_msg = "요청 토큰량이 기준치를 초과하였습니다. 다른 조건으로 실행해주세요."
            st.rerun()
        print(f"[예상 LLM 투입 토큰 수]: {use_token:,}/128,000")
        
        chain = PROMPT | LLM | JsonOutputParser()
        llm_result = chain.invoke({
            "search_results": search_results,
            "query": query,
            "chat_history": st.session_state.current_history
        })
    
        print(f"[LLM 답변]: {llm_result}")
    with st.spinner("Saving..."):
        docs = []
        docs.append(Document(
            page_content=f"""
                시기: {base_condition['timing']}월, 
                여행지: {base_condition['destination']}, 
                기간: {base_condition['duration']}, 
                인원: {base_condition['personnel']},
                예산: {base_condition['budget']}
                테마: {base_condition['theme']}
                기타 요구사항: {base_condition['comment']}
                \n\n
                {llm_result['result']}
            """,
            metadata={
                "timing": base_condition['timing'],
                "destination": base_condition['destination'],
                "duration": base_condition['duration'],
                "personnel": base_condition['personnel'],
                "budget": base_condition['budget'],
                "theme": base_condition['theme'],
                "comment": base_condition['comment'],
            }
        ))
        vectorstore.add_documents(docs)
    
    return llm_result
    
def needs_search(query: str, chroma_answer: str) -> bool:
    with st.spinner("답변 검사중..."):
        base_condition = st.session_state.meta
        base_condition_str = "\n".join(f"{k}: {v}" for k, v in base_condition.items())
        prompt = ChatPromptTemplate.from_messages([
            ("system",
            """
                검색 결과가 질문에 충분히 답할 수 있는지 판단하세요.
                답변이 아래 JSON 형식인지 확인하세요.
                {{
                    "result": "여행 일정 전체를 마크다운 텍스트로 작성. 예산계획과 여행팁 포함. 예산 계획은 테이블 형태로 마크다운 텍스트 작성. 마지막 줄은 반드시 '##> AI는 부정확한 정보를 제공할 수도 있습니다.' 로 끝내세요.",
                    "compression": "이 답변의 핵심 요약 (300자 이내). 반드시 포함: 여행지/일정/예산/사용자가 요청한 제외, 추가 조건 및 특별 요구사항"
                }}
                충분하면 "satisfy", 부족하면 "retry"만 출력하세요.
            """),
            ("system","기본 조건: {base_condition}"),
            ("user", "질문: {question} \n\n검색 결과:\n{context}")
        ])
        
        chain = prompt | LLM | StrOutputParser()
        result = chain.invoke({
            "base_condition": base_condition_str,
            "question": query,
            "context": chroma_answer
        })
    
    if result.strip().lower() == "retry":
        return True
    return False
    
def cal_token(search_text:str) -> int:
    enc = tiktoken.encoding_for_model("gpt-4o-mini")
    return len(enc.encode(search_text))

def save_search(response) -> str:
    result = response['result']
    compression = response['compression']
    st.session_state.last_plan = result
    st.session_state.messages.append({
        "role": "assistant",
        "content": result
    })
    st.session_state.history.append({
        "role": "assistant",
        "content": compression
    })
    return result

def all_clean():
    st.session_state.clear()
    st.session_state.base_query = ""
    st.session_state.meta = {}
    st.session_state.messages = []
    st.session_state.history = []
    st.session_state.current_history = ""
    st.session_state.last_plan = ""

st.title("AI 여행 플래너")
st.caption("여행지, 일정, 예산을 입력하면 맞춤 코스를 추천해드립니다.")
st.caption("⚠️ 브라우저 새로고침 시 대화 기록이 초기화됩니다.")

# 대화 기록 렌더링
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        
st.session_state.current_history = "\n".join(
    f'{msg["role"]}: {msg["content"]}'
    for msg in st.session_state.history
)
        
if st.session_state.warning_msg:
    st.warning(st.session_state.warning_msg)
    st.session_state.warning_msg = ""  # 한 번 보여주고 초기화

# 최초 입력 폼 - 대화 기록 없을 때만 표시
if not st.session_state.messages:
    with st.form("trip_form"):
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
    if not destination or not duration or not timing or not personnel:
        st.warning("필수 입력 항목을 모두 입력해주세요.")
    else:
        query = f"여행 시기: {timing.month}월, 여행지: {destination}, 기간: {duration}, 인원: {personnel}명"
        if budget:
            query += f", 예산: {budget}"
        if theme:
            query += f", 여행 테마: {theme}"
        if comment:
            query += f", 기타 요구사항: {comment}"
        query += " 여행 코스 추천해줘."
        
        # 내용 초기화
        # all_clean()
        
        # 여행 계획 저장
        st.session_state.base_query = query
        # metadata 카테고리 저장
        st.session_state.meta = {
            "timing": timing.month,
            "destination": destination,
            "duration": duration,
            "personnel": personnel,
            "budget": budget if budget else None,
            "theme": theme if theme else None,
            "comment": comment if comment else None
        }
        
        user_request = f"""
            여행지: {destination}
            기간: {duration}
            인원: {personnel}명
            여행 시기: {timing.month}월
            예산: {budget}
            여행 테마: {theme}
            기타 요구사항: {comment}
        """
        st.session_state.messages.append({
            "role": "user",
            "content": user_request
        })
        st.session_state.history.append({
            "role": "user",
            "content": query
        })
        
        with st.chat_message("assistant"):
            with st.spinner("여행 코스 검색 중..."):
                response = recommend_trip(query)
                result = save_search(response)
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
        
        with st.chat_message("assistant"):
            with st.spinner("검색 중..."):
                response = recommend_trip(follow_up)
                result = save_search(response)
            st.markdown(result)
            
    elif follow_up and follow_up.strip() == "":
        st.warning("추가 입력에 공백만 입력할 수 없습니다.")