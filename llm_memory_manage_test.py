import os
import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.output_parsers import JsonOutputParser
import datetime
import tiktoken


# API 설정
with open("C:\\Users\yjh21\OneDrive\Desktop\윤재훈\LLM\open_api.txt") as f:
    os.environ["OPENAI_API_KEY"] = f.read().strip()

with open("C:\\Users\yjh21\OneDrive\Desktop\윤재훈\LLM\\tavily_trip_reco.txt") as f:
    os.environ["TAVILY_API_KEY"] = f.read().strip()

# LLM 설정
LLM = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)

# 검색 도구 설정
SEARCH = TavilySearch(max_result=5, include_raw_content=True)

# 프롬프트 템플릿
PROMPT = ChatPromptTemplate.from_messages([
    ("system", """당신은 여행 플래너입니다.
        반드시 아래 규칙을 따르세요.
        1. 제공된 검색 결과를 최우선으로 활용하세요.
        2. 검색 결과에 없는 장소, 식당명, 가격은 절대 지어내지 마세요.
            가격은 반드시 원화와 현지 화폐를 동시에 표기하세요. 예: 79,700원 (8,200엔)
            단, 이동 시간/소요 시간/날씨 팁은 일반 상식으로 보완 가능합니다.
            보완한 내용은 반드시 "(참고)" 표시를 붙이세요.
        3. 답변 마지막에 참고한 URL 목록을 한꺼번에 표시하세요.
        4. 여행 계획은 구체적으로 작성하세요.
        5. 여행 시기의 날씨와 계절 특성을 반영하여 주의사항이나 팁을 추가하세요.
        6. 답변 마지막에 아래 형식으로 예산 계획과 여행 팁을 반드시 표기하세요.
            예산 계획은 항상 재계산 하세요.
            예산 계획과 여행 팁은 별도 JSON 키가 아닌 result 안에 포함해서 작성하세요.
            예산 계획: 항목별 비용을 원화와 현지 화폐 동시 표기하고 총합을 표기하세요.
            예: 항공권: 358,400원 (36,840엔)
            여행 팁: 날씨, 교통, 음식 팁을 각각 한 줄로 표기하세요.
        반드기 아래 JSON 형식으로만 답하세요. result는 마크다운 텍스트입니다.
        {{
            "result": "여행 일정 전체를 마크다운 텍스트로 작성. 예산계획과 여행팁 포함. 마지막 줄은 반드시 '> AI는 부정확한 정보를 제공할 수도 있습니다.' 로 끝내세요.",
            "compression": "이 답변의 핵심 요약 (200자 이내). 반드시 포함: 여행지/일정/예산/사용자가 요청한 제외, 추가 조건 및 특별 요구사항"
        }}
        result 안에 절대 JSON을 넣지 마세요. 마크다운 텍스트만 허용합니다.    
        """),
        # 4. 검색 결과만으로 답변이 부족하면 "검색 결과에 해당 정보가 없습니다"라고 명시하세요.
    ("system", "이전 대화 기록:\n{chat_history}"),
    ("user", "검색 결과:\n{search_results}\n\n질문: {query}")
])

def recommend_trip(query: str, need_search: bool) -> str:
    search_text = ""
    if need_search:
        print("추가 검색 필요")
        results = SEARCH.invoke(query)
        # print(f"Tavily 검색 결과: {results['results']}")
        search_text = "\n\n".join([
            f"제목: {r['title']}\nURL: {r['url']}\n내용: {r['raw_content'] or r['content']}"
            for r in results['results']
        ])
    else:
        print("추가 검색 불필요")
        search_text = f"이전 여행 일정:\n{st.session_state.last_result}"
        
    print(st.session_state.chat_history)
    
    use_token = count_token(search_text+st.session_state.chat_history)
    print(f"[LLM에 적용 되는 토큰 수]: {use_token}/ 128,000")
    
    chain = PROMPT | LLM | JsonOutputParser()
    return chain.invoke({
        "search_results": search_text,
        "query": query,
        "chat_history": st.session_state.chat_history
    })
    
def count_token(text: str) -> int:
    enc = tiktoken.encoding_for_model("gpt-4o-mini")
    return len(enc.encode(text))
    
def save_response(response):
    result = response["result"]
    compression = response["compression"]
    print(f"response keys: {response.keys()}")
    st.session_state.messages.append({
            "role": "assistant",
            "content": result
        })
    st.session_state.history.append({
        "role": "assistant",
        "content": compression
    })
    st.session_state.last_result = result  # ← 최근 result 전체 저장

def needs_search(follow_up: str) -> bool:
    response = LLM.invoke(
        f"""이전 대화 기록과 새 질문을 보고 웹 검색이 필요한지 판단하세요.
        새로운 장소, 맛집, 관광지 정보가 필요하면 "yes"
        기존 일정 수정, 시간 변경, 항목 제거라면 "no"
        반드시 "yes" 또는 "no" 중 하나만 답하세요.

        이전 대화:
        {st.session_state.chat_history}

        새 질문: {follow_up}
        """
    )
    
    return response.content.strip().lower() == "yes"

# session_state 초기화
# 화면 출력용 (전체 내용)
if "messages" not in st.session_state:
    st.session_state.messages = []
# LLM에 넘기는 대화 기록용 (요약본)
if "history" not in st.session_state:
    st.session_state.history = []
if "chat_history" not in st.session_state:
    st.session_state.chat_history = ""
if "base_query" not in st.session_state:
    st.session_state.base_query = ""
if "last_result" not in st.session_state:
    st.session_state.last_result = ""

st.title("AI 여행 플래너")
st.caption("여행지, 일정, 예산을 입력하면 맞춤 코스를 추천해드립니다.")
st.caption("⚠️ 브라우저 새로고침 시 대화 기록이 초기화됩니다.")

# 기존 대화 출력
for i, msg in enumerate(st.session_state.messages):
    if i == 0:
        continue
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# 기존 요약본 저장
st.session_state.chat_history = "\n".join([
    f"{h['role']}: {h['content']}"
    for h in st.session_state.history[-6:]
])

# 최초 입력 폼 - 대화 기록 없을 때만 표시
if not st.session_state.messages:
    with st.form("trip_form"):
        destination = st.text_input("여행지", placeholder="예: 제주도, 부산, 도쿄 등")
        duration    = st.text_input("여행 기간", placeholder="예: 3박 4일 등")
        timing      = st.date_input("출발 날짜", value=datetime.date.today() + datetime.timedelta(days=1))
        personnel   = int(st.number_input("인원 수", min_value=1, max_value=20, value=1, step=1))
        budget      = st.text_input("예산(선택)", placeholder="예: 인당 100만원 이하 등")
        theme       = st.text_input("테마(선택)", placeholder="예: 맛집 탐방, 액티비티 등")
        comment     = st.text_input("기타 추가사항", placeholder="예: 유니버셜 스튜디오는 안 가고 싶어 등")
        submitted   = st.form_submit_button("추천받기")
else:
    submitted = False
    
if submitted:
    if not destination or not duration or not timing or not personnel:
        st.warning("필수 입력 항목을 모두 입력해주세요.")
    else:
        query = f"여행 시기: {timing.month}월, 여행지: {destination}, 기간: {duration}, 인원: {personnel}명, "
        if budget:
            query += f"예산: {budget}, "
        if theme:
            query += f"여행 테마: {theme} "
        if comment:
            query += f"기타 요구사항: {comment}"
        query += "여행 코스 추천해줘."
        
        # 새 검색 시 대화 초기화
        st.session_state.base_query = query
        st.session_state.messages = [] 
        st.session_state.history = []
        st.session_state.last_result = ""
        
        st.session_state.messages.append({
            "role": "user",
            "content": query
        })

        # with st.chat_message("user"):
        #     st.markdown(query)
        
        with st.chat_message("assistant"):
            with st.spinner('여행 코스 검색 중...'):
                response = recommend_trip(query, True)
            st.markdown("## 추천 여행 코스")
            st.markdown(response['result'])

        save_response(response)
            
        
        
# 최초 입력 완료 후
if st.session_state.messages:
    follow_up = st.chat_input("추가 요청사항을 입력하세요. (예: 맛집만 더 알려줘, 3일차 코스 바꿔줘)")
    if follow_up and follow_up.strip() != "":
        st.session_state.messages.append({
            "role": "user",
            "content": follow_up
        })
        
        with st.chat_message("user"):
            st.markdown(follow_up)
        
        # 후속 질문은 base_query 맥락 유지하면서 새 검색
        new_query = f"{st.session_state.base_query} 추가 요청: {follow_up}"
        
        need = needs_search(follow_up)
            
        with st.chat_message("assistant"):
            with st.spinner("검색 중..."):
                response = recommend_trip(new_query, need)
            st.markdown(response['result'])
            
        save_response(response)
    elif follow_up and follow_up.strip() == "":
        st.warning("추가 입력에 공백만 입력할 수 없습니다.")