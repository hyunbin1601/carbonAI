"""Define the state structures for the agent."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

from langchain_core.messages import AnyMessage
from langgraph.graph import add_messages
from langgraph.managed import IsLastStep
from typing_extensions import Annotated


@dataclass
class InputState:
    """Defines the input state for the agent, representing a narrower interface to the outside world.

    This class is used to define the initial state and structure of incoming data.
    """

    messages: Annotated[Sequence[AnyMessage], add_messages] = field(
        default_factory=list
    )
    """
    Messages tracking the primary execution state of the agent.

    Typically accumulates a pattern of:
    1. HumanMessage - user input
    2. AIMessage with .tool_calls - agent picking tool(s) to use to collect information
    3. ToolMessage(s) - the responses (or errors) from the executed tools
    4. AIMessage without .tool_calls - agent responding in unstructured format to the user
    5. HumanMessage - user responds with the next conversational turn

    Steps 2-5 may repeat as needed.

    The `add_messages` annotation ensures that new messages are merged with existing ones,
    updating by ID to maintain an "append-only" state unless a message with the same ID is provided.
    """


@dataclass
class State(InputState):
    """Represents the complete state of the agent, extending InputState with additional attributes.

    This class can be used to store any information needed throughout the agent's lifecycle.
    """

    is_last_step: IsLastStep = field(default=False)
    """
    Indicates whether the current step is the last one before the graph raises an error.

    This is a 'managed' variable, controlled by the state machine rather than user code.
    It is set to 'True' when the step count reaches recursion_limit - 1.
    """

    # Hooxi-specific attributes for carbon emission management
    artifacts: list[dict] = field(default_factory=list)
    """
    List of generated artifacts (charts, calculators, documents).

    Each artifact contains:
    - type: "mermaid" | "react" | "document"
    - artifact_id: Unique identifier
    - code/path: Content or file path
    - metadata: Additional information
    """

    emission_data: dict = field(default_factory=dict)
    """
    Cached emission calculation results.

    Contains:
    - scope1, scope2, scope3: Emission values
    - total: Total emissions
    - breakdown: Detailed breakdown by source
    """

    market_data: dict = field(default_factory=dict)
    """
    Cached KRX market data for session.

    Contains:
    - price: Current emission credit price
    - change: Price change percentage
    - volume: Trading volume
    - timestamp: Last update time
    """

    # 5-node structure workflow fields
    intent_type: str = field(default="")
    """
    User intent classification result from classify node.

    Possible values:
    - "FAQ": Frequently asked questions
    - "MEASUREMENT": Emission calculation
    - "REPORTING": Document generation
    - "TRADING": Market trading
    - "CONSULTATION": Human consultant request
    """

    selected_tools: list[str] = field(default_factory=list)
    """
    List of tool names selected by route node.
    """

    tool_results: dict = field(default_factory=dict)
    """
    Results from tool execution in execute node.
    """

    quality_score: int = field(default=0)
    """
    Quality score (0-100) from verify node.
    """

    verification_feedback: str = field(default="")
    """
    Feedback from verify node for improvement.
    """

    regeneration_count: int = field(default=0)
    """
    Number of times response has been regenerated due to low quality.
    """

    ui: list[dict[str, Any]] = field(default_factory=list)
    """
    UI messages for artifacts (charts, components, visualizations).

    Each UI message contains:
    - id: Unique identifier
    - type: "component"
    - content: Component code or data
    - metadata: Additional metadata (message_id, title, etc.)

    These are rendered as artifacts in the chat UI sidebar.
    """
