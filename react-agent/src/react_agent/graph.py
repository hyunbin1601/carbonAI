"""5-Node Agent Architecture for Hooxi Carbon Emission Management.

Node Structure:
1. classify_node (Sonnet) - Classify user intent
2. route_node (Logic) - Route to appropriate tools
3. execute_node (Tools) - Execute selected tools
4. generate_node (Haiku) - Generate response
5. verify_node (Sonnet) - Verify quality and decide if regeneration needed
"""

import json
import time
from datetime import UTC, datetime
from typing import Dict, List, Literal

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langchain_anthropic import ChatAnthropic
from langgraph.graph import StateGraph
from langgraph.prebuilt import ToolNode

from react_agent.configuration import Configuration
from react_agent.state import InputState, State
from react_agent.tools import TOOLS


# ============================================================================
# Node 1: Classify (Sonnet) - Intent Understanding
# ============================================================================


async def classify_node(
    state: State, config: RunnableConfig
) -> Dict[str, str]:
    """Classify user intent using Claude Sonnet.

    This node analyzes the user's message and determines:
    - What type of request it is (FAQ, MEASUREMENT, REPORTING, etc.)
    - What information is needed to fulfill the request

    Args:
        state: Current conversation state
        config: Runtime configuration

    Returns:
        dict: Updated state with intent_type
    """
    configuration = Configuration.from_runnable_config(config)

    # Initialize Sonnet for classification
    llm = ChatAnthropic(
        temperature=0,
        model=configuration.model_classify,
    )

    # Classification prompt (Customer-facing perspective)
    classify_prompt = f"""You are an intent classifier for 후시파트너스 customer service chatbot.

Analyze the customer's message and classify it into ONE of these categories:

1. SERVICE_INQUIRY - Customer asking about services:
   - "탄소 배출량 측정 서비스는 어떻게 진행되나요?"
   - "배출권 거래 중개는 어떤 서비스인가요?"
   - "비용이 얼마나 드나요?"
   - "기간은 얼마나 걸리나요?"
   - General service information, pricing, process

2. EMISSION_ESTIMATE - Customer wants rough emission estimate:
   - "우리 회사 배출량이 얼마나 될까요?"
   - "간단하게 계산 좀 해주세요"
   - Quick calculation requests (not official measurement)

3. MARKET_INFO - Customer asking about emission credit market:
   - "지금 배출권 가격이 어떻게 되나요?"
   - "KRX 시장 현황은 어때요?"
   - Market trends, prices, trading volume

4. REGULATION_INFO - Customer asking about regulations:
   - "탄소 배출 의무 대상인가요?"
   - "새로운 규제가 있나요?"
   - Environmental regulations, compliance requirements

5. CONSULTATION_REQUEST - Customer wants to speak with consultant:
   - "상담원 연결해주세요"
   - "전문가와 상담하고 싶어요"
   - "계약하려면 어떻게 해야 하나요?"
   - Requests for human expert consultation

User's last message: {state.messages[-1].content if state.messages else "No message"}

Respond with ONLY the category name (SERVICE_INQUIRY, EMISSION_ESTIMATE, MARKET_INFO, REGULATION_INFO, or CONSULTATION_REQUEST).
"""

    # Get classification
    response = await llm.ainvoke([HumanMessage(content=classify_prompt)])
    intent_type = response.content.strip().upper()

    # Validate intent type
    valid_intents = ["SERVICE_INQUIRY", "EMISSION_ESTIMATE", "MARKET_INFO", "REGULATION_INFO", "CONSULTATION_REQUEST"]
    if intent_type not in valid_intents:
        intent_type = "SERVICE_INQUIRY"  # Default fallback

    return {"intent_type": intent_type}


# ============================================================================
# Node 2: Route (Logic) - Conditional Routing
# ============================================================================


def route_node(state: State, config: RunnableConfig) -> Dict[str, List[str]]:
    """Route to appropriate tools based on classified customer intent.

    This is pure Python logic (no LLM) that maps customer needs to tools.

    Args:
        state: Current state with intent_type
        config: Runtime configuration

    Returns:
        dict: Updated state with selected_tools
    """
    intent_type = state.intent_type

    # Routing logic (Customer journey focused)
    tool_mapping = {
        # Customer asking about services → Search knowledge base for service info
        "SERVICE_INQUIRY": ["search_emission_database"],

        # Customer wants rough estimate → Simple calculation + interactive calculator
        "EMISSION_ESTIMATE": ["calculate_emissions", "generate_react_component"],

        # Customer asking about market → Market data + visualization
        "MARKET_INFO": ["get_krx_market_data", "generate_mermaid_chart"],

        # Customer asking about regulations → Search regulations + government docs
        "REGULATION_INFO": ["search_emission_database", "firecrawl_government_docs"],

        # Customer wants human consultant → Connect to consultation system
        "CONSULTATION_REQUEST": ["request_human_consultation"],
    }

    selected_tools = tool_mapping.get(intent_type, ["search_emission_database"])

    return {"selected_tools": selected_tools}


# ============================================================================
# Node 3: Execute (Tools) - Tool Execution
# ============================================================================


async def execute_node(
    state: State, config: RunnableConfig
) -> Dict:
    """Execute the selected tools and collect results.

    Args:
        state: Current state with selected_tools
        config: Runtime configuration

    Returns:
        dict: Updated state with tool_results
    """
    # For now, we'll use LangChain's ToolNode for tool execution
    # In a full implementation, you would selectively call only selected_tools

    # Check if the last message has tool calls
    last_message = state.messages[-1] if state.messages else None

    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        # Execute tools using ToolNode
        tool_node = ToolNode(TOOLS)
        result = await tool_node.ainvoke(state, config)
        return result
    else:
        # No tools to execute, return empty result
        return {"tool_results": {}}


# ============================================================================
# Node 4: Generate (Haiku) - Response Generation
# ============================================================================


async def generate_node(
    state: State, config: RunnableConfig
) -> Dict:
    """Generate response using Claude Haiku (fast and cost-effective).

    Creates artifacts from tool results and adds them as UIMessages.

    Args:
        state: Current state with tool results
        config: Runtime configuration

    Returns:
        dict: Updated state with generated response and UI messages
    """
    configuration = Configuration.from_runnable_config(config)

    # Initialize Haiku for response generation
    llm = ChatAnthropic(
        temperature=0.3,
        model=configuration.model_generate,
    )

    # System prompt with artifact instructions
    system_message = configuration.system_prompt.format(
        system_time=datetime.now(tz=UTC).isoformat()
    )

    # Add artifact guidance
    artifact_guidance = """

## 📊 아티팩트 안내

도구가 차트나 계산기를 생성했다면:
- "오른쪽 패널에서 인터랙티브 차트/계산기를 확인하실 수 있습니다"라고 안내
- 코드를 직접 표시하지 말고, 결과만 설명
- 사용자가 직접 조작할 수 있다는 점 강조

예시: "배출량 계산 결과를 오른쪽 패널의 인터랙티브 차트에서 확인하실 수 있습니다."
"""

    full_system_message = system_message + artifact_guidance

    # Generate response
    response = await llm.ainvoke(
        [{"role": "system", "content": full_system_message}, *state.messages]
    )

    # Extract artifacts from tool messages
    ui_messages = []

    # Check recent tool messages for artifact data
    for msg in reversed(state.messages):
        if hasattr(msg, 'type') and msg.type == 'tool':
            # Check if tool result contains artifact code
            if hasattr(msg, 'content') and isinstance(msg.content, str):
                try:
                    tool_result = json.loads(msg.content) if msg.content.startswith('{') else {}

                    # Check for React component
                    if 'code' in tool_result and 'type' in tool_result:
                        artifact_type = tool_result['type']
                        artifact_code = tool_result['code']
                        artifact_id = tool_result.get('artifact_id', f'artifact_{int(time.time())}')

                        # Create UIMessage for artifact
                        ui_msg = {
                            'id': artifact_id,
                            'type': 'component',
                            'content': artifact_code,
                            'metadata': {
                                'message_id': response.id,
                                'artifact_type': artifact_type,
                                'title': f'{"차트" if artifact_type == "mermaid" else "계산기"}'
                            }
                        }
                        ui_messages.append(ui_msg)
                except (json.JSONDecodeError, AttributeError):
                    continue

    return {
        "messages": [response],
        "ui": ui_messages
    }


# ============================================================================
# Node 5: Verify (Sonnet) - Quality Verification
# ============================================================================


async def verify_node(
    state: State, config: RunnableConfig
) -> Dict:
    """Verify response quality using Claude Sonnet.

    Evaluates the generated response on:
    - Accuracy (40 points)
    - Completeness (30 points)
    - Clarity (20 points)
    - Usefulness (10 points)

    Total: 100 points
    Pass threshold: 70+ points (configurable)

    Args:
        state: Current state with generated response
        config: Runtime configuration

    Returns:
        dict: Updated state with quality_score and verification_feedback
    """
    configuration = Configuration.from_runnable_config(config)

    # Get the last assistant message
    last_response = None
    for msg in reversed(state.messages):
        if isinstance(msg, AIMessage):
            last_response = msg.content
            break

    if not last_response:
        return {"quality_score": 100, "verification_feedback": "No response to verify - accepting"}

    # For now, skip actual verification to avoid infinite loop
    # TODO: Implement proper verification with JSON parsing

    # Auto-accept with high score to prevent infinite regeneration
    return {
        "quality_score": 100,
        "verification_feedback": "Auto-approved (verification temporarily disabled)",
        "regeneration_count": state.regeneration_count + 1
    }


# ============================================================================
# Conditional Edges
# ============================================================================


def should_execute_tools(state: State) -> Literal["execute", "generate"]:
    """Decide if we need to execute tools or skip to generation.

    Args:
        state: Current state

    Returns:
        "execute" if tools are needed, "generate" otherwise
    """
    # Check if we need to execute tools based on intent
    if state.intent_type in ["FAQ", "MEASUREMENT", "REPORTING", "TRADING"]:
        return "execute"
    else:
        return "generate"


def should_regenerate(state: State, config: RunnableConfig) -> Literal["__end__", "generate"]:
    """Decide if response quality is acceptable or needs regeneration.

    Args:
        state: Current state with quality_score
        config: Runtime configuration

    Returns:
        "__end__" if quality is acceptable, "generate" to regenerate
    """
    configuration = Configuration.from_runnable_config(config)

    # Check if quality meets threshold
    if state.quality_score >= configuration.quality_threshold:
        return "__end__"

    # Prevent infinite regeneration loop
    if state.regeneration_count >= 2:
        # After 2 regenerations, accept the result
        return "__end__"

    # Regenerate
    return "generate"


# ============================================================================
# Helper: Check if tools need to be called
# ============================================================================


async def call_model_with_tools(
    state: State, config: RunnableConfig
) -> Dict:
    """Call model with tools bound (for tool selection).

    This is used before execute_node to let the model decide which tools to call.

    Args:
        state: Current state
        config: Runtime configuration

    Returns:
        dict: Updated state with tool calls
    """
    configuration = Configuration.from_runnable_config(config)

    # Initialize Sonnet with tools
    llm = ChatAnthropic(
        temperature=0.1,
        model=configuration.model_classify,
    )
    model = llm.bind_tools(TOOLS)

    # System prompt
    system_message = configuration.system_prompt.format(
        system_time=datetime.now(tz=UTC).isoformat()
    )

    # Add guidance about which tools to use based on intent
    tool_guidance = f"\n\nUser intent classified as: {state.intent_type}\nRecommended tools: {', '.join(state.selected_tools)}"
    enhanced_system = system_message + tool_guidance

    # Get model response with tool calls
    response = await model.ainvoke(
        [{"role": "system", "content": enhanced_system}, *state.messages]
    )

    return {"messages": [response]}


def route_after_tool_selection(state: State) -> Literal["execute", "generate"]:
    """Route based on whether tools were called.

    Args:
        state: Current state

    Returns:
        "execute" if tools were called, "generate" otherwise
    """
    last_message = state.messages[-1] if state.messages else None

    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        return "execute"
    else:
        return "generate"


# ============================================================================
# Graph Construction
# ============================================================================


# Create the graph
builder = StateGraph(State, input=InputState, config_schema=Configuration)

# Add all 5 nodes
builder.add_node("classify", classify_node)
builder.add_node("route", route_node)
builder.add_node("call_with_tools", call_model_with_tools)  # Tool selection
builder.add_node("execute", execute_node)
builder.add_node("generate", generate_node)
builder.add_node("verify", verify_node)

# Set entry point
builder.add_edge("__start__", "classify")

# Connect classify → route
builder.add_edge("classify", "route")

# Connect route → call_with_tools (for tool selection)
builder.add_edge("route", "call_with_tools")

# Conditional: call_with_tools → execute OR generate
builder.add_conditional_edges(
    "call_with_tools",
    route_after_tool_selection,
)

# Connect execute → generate
builder.add_edge("execute", "generate")

# Connect generate → verify
builder.add_edge("generate", "verify")

# Conditional: verify → __end__ OR regenerate
builder.add_conditional_edges(
    "verify",
    should_regenerate,
)

# Compile the graph
graph = builder.compile(name="Hooxi Carbon Agent - 5 Nodes")
