from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain.agents import create_agent
from langchain.agents.middleware import SummarizationMiddleware, AgentMiddleware
from langchain_core.messages import HumanMessage, ToolMessage, AIMessage, RemoveMessage
from langchain_tavily import TavilySearch
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.message import REMOVE_ALL_MESSAGES
from typing_extensions import override
from vector_store import get_agent_vector_store
import tiktoken

VECTORSTORE = get_agent_vector_store()
DEFAULT_K = 5
THRESHOLD = 0.5

MEMORY = MemorySaver()

# SummarizationMiddleware before_model 오버라이딩
# 이유: ToolMessage 압축 대상 제외 및 관리 제외
class FilteredSummarizationMiddleware(SummarizationMiddleware):
    
    def _strip_tool_messages(self, messages):
        return [
            m for m in messages
            if not isinstance(m, ToolMessage)
            and not (isinstance(m, AIMessage) and m.tool_calls)
        ]
    
    @override
    def before_model(self, state, runtime):
        """Process messages before model invocation, potentially triggering summarization.

        Args:
            state: The agent state.
            runtime: The runtime environment.

        Returns:
            An updated state with summarized messages if summarization was performed.
        """
        messages = state["messages"]
        self._ensure_message_ids(messages)
        
        # ToolMessage가 마지막이면 현재 툴 사이클 진행 중 → 건드리지 않음
        in_active_cycle = isinstance(messages[-1], ToolMessage) if messages else False

        total_tokens = self.token_counter(messages)
        if not self._should_summarize(messages, total_tokens):
            if not in_active_cycle:
                has_tool = any(
                    isinstance(m, ToolMessage) or (isinstance(m, AIMessage) and m.tool_calls)
                    for m in messages
                )
                if has_tool:
                    return{
                        "messages": [
                            RemoveMessage(id=REMOVE_ALL_MESSAGES),
                            *self._strip_tool_messages(messages)
                        ]
                    }
            return None

        cutoff_index = self._determine_cutoff_index(messages)

        if cutoff_index <= 0:
            return None

        messages_to_summarize, preserved_messages = self._partition_messages(messages, cutoff_index)

        # ToolMessage, AIMessage(tool_calls)를 요약에서 제외
        # messages_to_summarize: 요약 대상 메세지들
        filtered = self._strip_tool_messages(messages_to_summarize)

        summary = self._create_summary(filtered)
        new_messages = self._build_new_messages(summary)

        # 사이클 중이 아닐 때만 preserved도 정리
        if not in_active_cycle:
            preserved_messages = self._strip_tool_messages(preserved_messages)
        return {
            "messages": [
                RemoveMessage(id=REMOVE_ALL_MESSAGES),
                *new_messages,
                *preserved_messages,
            ]
        }

# AIMessage(answer)가 생성 된 이후 처리
class FilteredToolMessageAfterModel(AgentMiddleware):
    def after_model(self, state, runtime):
        messages = state['messages']
        last_message = messages[-1] if messages else None
        
        # 마지막 메세지가 AI메세지인지 (최종 답변이 아니면 걍 넘어가)
        if isinstance(last_message, AIMessage) and not last_message.tool_calls:
            return {
                "messages": [
                    RemoveMessage(id=REMOVE_ALL_MESSAGES),  # 추가
                    *[
                        m for m in messages
                        if not isinstance(m, ToolMessage)
                        and not (isinstance(m, AIMessage) and m.tool_calls)
                    ]
                ]
            }
        
        return None

def get_my_default_agent(model: ChatOpenAI ,tools: list, prompt: str):
    return create_agent(
        model,
        tools,
        checkpointer=MEMORY,
        middleware=[
            FilteredToolMessageAfterModel(),
            FilteredSummarizationMiddleware(
                model,
                trigger=("tokens", 10000),
                keep=("messages", 12)
            )
        ],
        system_prompt=prompt
    )
    
@tool
def search_chroma(query: str, region: str, month: int) -> str:
    """
    여행 관련 정보를 DB에서 먼저 검색합니다.
    반드시 웹 검색 이전에 먼저 검색합니다.
    region: 광역 단위 지역명만 사용. 예) "제주도", "부산", "서울", "오사카", "베이징", "뉴욕" 등
    시/구/동 등 하위 단위는 절대 넣지 말 것.
    month: 여행 월 숫자. 예) 6, 7, 8
    가격, 장소 등에 대한 정보는 항상 웹 검색을 통해 최신 정보를 반영합니다.
    """
    
    conditions = []
    if region:
        conditions.append({"region": region})
    if month:
        conditions.append({"month": month})
        
    if len(conditions) == 0:
        search_filter = None
    elif len(conditions) == 1:
        search_filter = conditions[0]
    else:
        search_filter = {"$and": conditions}
        
    print(f"search_filter: {search_filter}")
    result = VECTORSTORE.similarity_search_with_relevance_scores(
        query,
        k=DEFAULT_K,
        filter=search_filter
    )
    return "\n".join(doc.page_content for doc, score in result if score >= THRESHOLD) if result else "DB에 정보 없음"

@tool
def search_web(query: str):
    """
    여행 관련 최신 정보를 웹에서 검색합니다.
    """
    result = TavilySearch(max_results=5).invoke(query)
    search_results = "\n\n".join([
        f"제목: {rs['title']}\nURL: {rs['url']}\n내용: {rs['raw_content'] or rs['content']}"
        for rs in result['results']
    ])
    enc = tiktoken.encoding_for_model("gpt-4o-mini")
    print(f"예상 소요 토큰: {len(enc.encode(search_results))}")
    
    return search_results