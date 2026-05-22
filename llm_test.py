import os
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


# API 설정
with open("C:\\Users\yjh21\OneDrive\Desktop\윤재훈\LLM\open_api.txt") as f:
    os.environ["OPENAI_API_KEY"] = f.read().strip()

with open("C:\\Users\yjh21\OneDrive\Desktop\윤재훈\LLM\\tavily_trip_reco.txt") as f:
    os.environ["TAVILY_API_KEY"] = f.read().strip()
    
# LLM 설정
LLM = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)

# 검색 도구 설정
SEARCH = TavilySearch(max_result=3)

# 프롬프트 템플릿
PROMPT = ChatPromptTemplate.from_messages([
    ("system", """당신은 여행 플래너입니다. 
        반드시 아래 규칙을 따르세요.
        1. 답변은 반드시 제공된 검색 결과에 있는 정보만 사용하세요.
        2. 검색 결과에 없는 가격, 식당, 장소 등에 대한 정보는 절대 지어내지 마세요.
        3. 답변 마지막에 참고한 URL 목록을 한꺼번에 표시하세요. URL을 특정할 수 없으면 검색 결과로 넘어온 전체 URL을 나열하세요.
        """),
        # 4. 검색 결과만으로 답변이 부족하면 "검색 결과에 해당 정보가 없습니다"라고 명시하세요.
    ("user", "검색 결과:\n{search_results}\n\n질문: {query}")
])

def reccomend_trip(query: str) -> str:
    # 1. 검색 실행
    results = SEARCH.invoke(query)
    # print(type(results))
    # for i, r in enumerate(results):
    #     print(f"index: {i}, r: {r}")
    #     print(f"results[{r}]: {results[r]}")
    print(results["results"])
    
    search_text = "\n\n".join([
        f"제목: {r['title']}\nURL: {r['url']}\n내용: {r['content']}"
        for r in results["results"]
    ])
    
    # 2. LLM에 검색 결과 + 질문 넘기기
    chain = PROMPT | LLM | StrOutputParser()
    response = chain.invoke({
        "search_results": search_text,
        "query": query
    })
    
    return response

# 실행
result = reccomend_trip("현재 날짜, 시간 기준으로 제주도 3박 4일 코스 알려줘.")
print(result)