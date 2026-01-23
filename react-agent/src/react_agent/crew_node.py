"""Crew AI Hierarchical Process Node for LangGraph.

이 모듈은 Crew AI의 Hierarchical Process를 LangGraph 노드로 통합합니다.
Manager Agent가 도메인 전문가들에게 동적으로 업무를 할당하는 구조입니다.

키워드 기반 복잡도 평가 및 crew ai 사용 여부를 판단합니다
"""

from typing import cast
from crewai import Agent, Crew, Task, Process
from langchain_anthropic import ChatAnthropic
from langchain_core.runnables import RunnableConfig
from react_agent.state import State
from react_agent.tools import get_all_tools
from react_agent.crew_prompts import get_crew_agent_prompt


def create_carbon_trading_agent(tools: list) -> Agent:
    """탄소 배출권 거래 전문가 Agent 생성 (상세 프롬프트 적용)"""
    llm = ChatAnthropic(temperature=0.1, model="claude-haiku-4-5")

    # crew_prompts.py의 상세 프롬프트 가져오기
    detailed_prompt = get_crew_agent_prompt("carbon_trading")

    return Agent(
        role="탄소 배출권 거래 전문가",
        goal="배출권 시장 분석, 거래 전략 수립, 가격 예측 및 포트폴리오 최적화. 데이터 기반 분석과 시각화(AG Charts/Mermaid)를 필수로 활용.",
        backstory=detailed_prompt,  # 상세 프롬프트 적용
        verbose=True,
        allow_delegation=False,
        llm=llm,
        tools=tools,
    )


def create_regulation_expert_agent(tools: list) -> Agent:
    """온실가스 규제 대응 전문가 Agent 생성 (상세 프롬프트 적용)"""
    llm = ChatAnthropic(temperature=0.1, model="claude-haiku-4-5")

    # crew_prompts.py의 상세 프롬프트 가져오기
    detailed_prompt = get_crew_agent_prompt("regulation")

    return Agent(
        role="온실가스 규제 컨설턴트",
        goal="환경 법규 해석, Scope 1/2/3 배출량 산정, 제3자 검증 대응, ESG 공시 규제. 법규 근거와 절차 시각화(Mermaid)를 필수로 포함.",
        backstory=detailed_prompt,  # 상세 프롬프트 적용
        verbose=True,
        allow_delegation=False,
        llm=llm,
        tools=tools,
    )


def create_customer_service_agent(tools: list) -> Agent:
    """고객 상담 및 솔루션 설계 전문가 Agent 생성 (상세 프롬프트 적용)"""
    llm = ChatAnthropic(temperature=0.1, model="claude-haiku-4-5")

    # crew_prompts.py의 상세 프롬프트 가져오기
    detailed_prompt = get_crew_agent_prompt("customer_service")

    return Agent(
        role="맞춤형 솔루션 아키텍트",
        goal="기업 맞춤형 환경 솔루션 설계, ROI 분석, 비용-효과 모델링, 실질적 가치 창출. 서비스 비교(AG Grid)와 로드맵(Mermaid)을 필수로 제공.",
        backstory=detailed_prompt,  # 상세 프롬프트 적용
        verbose=True,
        allow_delegation=False,
        llm=llm,
        tools=tools,
    )


def create_manager_agent() -> Agent:
    """Manager Agent 생성 (Hierarchical Process용, 상세 프롬프트 적용)"""
    llm = ChatAnthropic(temperature=0.1, model="claude-sonnet-4-5")  # Manager는 Sonnet 사용

    # crew_prompts.py의 상세 프롬프트 가져오기
    detailed_prompt = get_crew_agent_prompt("manager")

    return Agent(
        role="탄소 환경 컨설팅 팀 매니저",
        goal="사용자 질문을 분석하고 적절한 도메인 전문가에게 명확한 지시를 내려 업무를 할당한 후, 전문가들의 답변을 통합하여 최적의 답변 제공",
        backstory=detailed_prompt,  # 상세 프롬프트 적용
        verbose=True,
        allow_delegation=True,  # Manager는 delegation 허용
        llm=llm,
    )


async def crew_hierarchical_node(state: State, config: RunnableConfig) -> dict:
    """Crew AI Hierarchical Process를 LangGraph 노드로 실행.

    Manager Agent가 사용자 질문을 분석하고,
    적절한 도메인 전문가(탄소배출권/규제/고객상담)에게 동적으로 업무를 할당합니다.

    Args:
        state: LangGraph State 객체
        config: RunnableConfig

    Returns:
        업데이트된 state dictionary
    """
    # 사용자 메시지 추출
    messages = state.get("messages", [])
    if not messages:
        return {"messages": [{"role": "assistant", "content": "질문을 입력해주세요."}]}

    last_user_message = None
    for msg in reversed(messages):
        if hasattr(msg, "type") and msg.type == "human":
            last_user_message = msg.content
            break
        elif isinstance(msg, dict) and msg.get("role") == "user":
            last_user_message = msg.get("content", "")
            break

    if not last_user_message:
        return {"messages": [{"role": "assistant", "content": "질문을 입력해주세요."}]}

    # 도구 로드 (async)
    tools = await get_all_tools()

    # 도메인 전문가 Agents 생성
    carbon_expert = create_carbon_trading_agent(tools)
    regulation_expert = create_regulation_expert_agent(tools)
    customer_service = create_customer_service_agent(tools)
    manager = create_manager_agent()

    # Task 생성 (Manager가 동적으로 전문가들에게 할당)
    task = Task(
        description=f"""사용자 질문: {last_user_message}

위 질문을 분석하고 적절한 전문가(탄소배출권/규제/고객상담)에게 업무를 할당하여 답변을 작성하세요.

**중요: 답변 시 반드시 시각화를 활용하세요:**
- 데이터/통계 → AG Charts (```agchart)
- 표 형식 데이터 → AG Grid (```aggrid)
- 프로세스/절차 → Mermaid (```mermaid)
- 지리 정보 → Map (```map)

시각화 예시는 각 전문가의 프롬프트에 포함되어 있습니다.""",
        expected_output="사용자 질문에 대한 전문적이고 구조화된 답변 (시각화 포함)",
        agent=manager,  # Manager가 이 Task를 받아서 전문가들에게 위임
    )

    # Crew 생성 (Hierarchical Process)
    crew = Crew(
        agents=[manager, carbon_expert, regulation_expert, customer_service],
        tasks=[task],
        process=Process.hierarchical,  # Hierarchical Process 사용
        manager_agent=manager,  # Manager Agent 지정
        verbose=True,
    )

    # Crew 실행
    try:
        result = crew.kickoff()

        # 결과를 LangGraph 메시지 형식으로 변환
        response_content = str(result)

        return {
            "messages": [
                {
                    "role": "assistant",
                    "content": response_content,
                }
            ]
        }
    except Exception as e:
        error_msg = f"Crew AI 실행 중 오류 발생: {str(e)}"
        print(error_msg)
        return {
            "messages": [
                {
                    "role": "assistant",
                    "content": f"죄송합니다. 답변 처리 중 오류가 발생했습니다: {str(e)}",
                }
            ]
        }


def should_use_crew(state: State) -> bool:
    """Crew AI Hierarchical Process를 사용할지 판단하는 함수.

    복잡한 질문이거나 여러 도메인이 필요한 경우 True 반환.

    Args:
        state: LangGraph State 객체

    Returns:
        Crew AI 사용 여부
    """
    messages = state.get("messages", [])
    if not messages:
        return False

    # 마지막 사용자 메시지 추출
    last_user_message = ""
    for msg in reversed(messages):
        if hasattr(msg, "type") and msg.type == "human":
            last_user_message = msg.content.lower()
            break
        elif isinstance(msg, dict) and msg.get("role") == "user":
            last_user_message = msg.get("content", "").lower()
            break

    # 복잡한 질문 키워드 (여러 도메인 필요)
    complex_keywords = [
        "전략 수립",
        "종합적으로",
        "통합",
        "모든",
        "전체적으로",
        "roi 분석",
        "비용 효과",
        "로드맵",
        "단계별",
        "검토",
        "평가",
    ]

    # 여러 도메인 키워드가 함께 등장하는 경우
    domain_keywords = {
        "carbon": ["배출권", "거래", "koc", "kcu", "kau", "시장"],
        "regulation": ["규제", "법규", "scope", "검증", "보고"],
        "service": ["서비스", "솔루션", "상담", "견적", "요금"],
    }

    domain_count = 0
    for keywords in domain_keywords.values():
        if any(kw in last_user_message for kw in keywords):
            domain_count += 1

    # 복잡한 키워드가 있거나, 2개 이상 도메인이 필요한 경우 Crew 사용
    return any(kw in last_user_message for kw in complex_keywords) or domain_count >= 2
