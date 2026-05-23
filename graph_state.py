from langgraph.graph import StateGraph, END
from typing import TypedDict
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from vector_store import get_vector_store

# 창의성 버려
LLM = ChatOpenAI(model="gpt-4o-mini", temperature=0)

class GraphState(TypedDict):
    question: str       # 사용자 질문
    filter: dict        # LLM이 추출한 필터 조건
    documents: list     # 검색 된 문서들
    answer: str         # 최종 답변
    evaluate: str       # 평가 결과
    
class TripGraph:
    def __init__(self):
        self.vectorstore = get_vector_store()
        self.defualt_k = 5
        self.THRESHOLD = 0.5
        self.graph = self.build_graph()
    
    def analyze_node(self, state: GraphState) -> GraphState:
        prompt = ChatPromptTemplate.from_messages([
            ("system", 
            """
                사용자 질문을 분석해서 검색 필터를 JSON으로 추출하세요.
                가능한 filter 키:
                -timing: 월 단위 숫자. 필수 정보 예: 1
                -destination: 여행 장소. 예: 서울
                해당 조건이 없으면 키 자체를 포함하지 마세요.
                반드시 JSON만 출력하세요 예:
                {{
                    "timing": 3,
                    "destination": 오사카,
                }}
            """),
            ("user", "{question}")
        ])
        
        chain = prompt | LLM | JsonOutputParser()
        filter_result = chain.invoke({
            "question": state['question']
        })
        
        print(f"[filter]: {filter_result}")
        return {**state, "filter" : filter_result}
    
    def search_node(self, state: GraphState) -> GraphState:
        filter_condition = self.build_chroma_filter(state['filter'])
        
        docs_score = self.vectorstore.similarity_search_with_relevance_scores(
            state['question'],
            k=self.defualt_k,
            filter=filter_condition
        )
        print(f"[검색]: {len(docs_score)}건 검색 완료.")
        
        for doc, score in docs_score:
            print(f"  - score: {score:.4f} | {doc.page_content[:50]}...")
            
        filtered = [doc for doc, score in docs_score if score >= self.THRESHOLD]
        return {**state, "documents": filtered}
    
    def build_chroma_filter(self, filter_condition: dict) -> dict:
        if not filter_condition:
            return None
        if len(filter_condition) == 1:
            return filter_condition
        return {"$and": [{k:v} for k, v in filter_condition.items()]}
    
    def check_score_node(self, state: GraphState) -> GraphState:
        prompt = ChatPromptTemplate.from_messages([
            ("system",
            """
                검색 결과가 질문에 충분히 답할 수 있는지 판단하세요.
                충분하면 "sufficient", 부족하면 "retry"만 출력하세요.
            """),
            ("user", "질문: {question}\n\n 검색 결과:\n{context}")
        ])
        
        context = "\n".join(doc.page_content for doc in state["documents"])
        chain = prompt | LLM | StrOutputParser()
        result = chain.invoke({
            "question": state['question'],
            "context": context
        })
        print(f"[답변 검수 결과]: {result}")
        
        return {**state, "evaluate": result.strip().lower()}
    
    def build_chroma_branch(self, state: GraphState) -> str:
        # 문서가 아예 없으면 → 웹 검색 유도
        # 문서가 그래도 50%는 신뢰할 수 있어야지.
        if not state.get("documents") or len(state.get('documents')) <= (self.defualt_k//2) +1:
            return "no_result"
        
        # 있으면 → LLM 검수
        if state.get("evaluate") == "sufficient":
            return "answer"
        
        return "search_wide"
    
    def search_wide_node(self, state: GraphState) -> GraphState:
        print("[재검색]: 검색 문서 수 확대 -> 20건")
        filter_condition = self.build_chroma_filter(state['filter'])
        docs = self.vectorstore.similarity_search(
            query= state["question"],
            k= self.defualt_k * 4,
            filter= filter_condition
        )
        
        print(f"[재검색 결과] {len(docs)}건")
        return {**state, "documents": docs}
    
    def no_result(self, state: GraphState) -> GraphState:
        return {**state, "answer": None, "evaluate": "no_result"}

    def answer_result_node(self, state: GraphState) -> GraphState:
        context = "\n".join(doc.page_content for doc in state["documents"])
        prompt = ChatPromptTemplate.from_messages([
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
            ("user", "검색 결과:\n{context}\n\n질문:\n{query}")
        ])
        
        chain = prompt | LLM | JsonOutputParser()
        answer = chain.invoke({
            "context": context,
            "query": state["question"]
        })
        return {**state, "answer": answer}
    
    def build_graph(self):
        builder = StateGraph(GraphState)
        
        # filter 생성
        builder.add_node("analyze", self.analyze_node)
        # Chroma 검색
        builder.add_node("search", self.search_node)
        # 체크
        builder.add_node("check", self.check_score_node)
        # 다시 검사
        builder.add_node("search_wide", self.search_wide_node)
        # 데이터 없음
        builder.add_node("no_result", self.no_result)
        # 답하기
        builder.add_node("answer", self.answer_result_node)
        
        # 연결
        builder.set_entry_point("analyze")
        builder.add_edge("analyze", "search")
        builder.add_edge("search", "check")
        builder.add_conditional_edges("check", self.build_chroma_branch)
        builder.add_edge("search_wide", "answer")
        builder.add_edge("answer", END)
        builder.add_edge("no_result", END)
        
        return builder.compile()
        
    def invoke(self, question: str):
        return self.graph.invoke({
            "question": question,
            "filter": {},
            "documents": [],
            "answer": "",
            "evaluate": ""
        })