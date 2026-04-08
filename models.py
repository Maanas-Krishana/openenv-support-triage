from __future__ import annotations
from typing import Optional, Literal
try:
    from openenv.core.env_server.types import Action, Observation
    from pydantic import Field
except ImportError:
    try:
        from core.env_server.types import Action, Observation
        from pydantic import Field
    except ImportError:
        from pydantic import BaseModel as Action
        from pydantic import BaseModel as Observation
        from pydantic import Field

class CustomerSupportAction(Action):
    action_type: Literal["assign", "reply_and_resolve", "escalate", "search_knowledge_base"] = Field(
        ..., description="The action to take: assign to a department, reply to the customer and resolve, escalate, or search the internal knowledge base."
    )
    department: Optional[Literal["IT", "Billing", "Sales", "General"]] = Field(
        None, description="The department to assign to. Required if action_type is 'assign'."
    )
    reply_message: Optional[str] = Field(
        None, description="The reply message to send. Required if action_type is 'reply_and_resolve'."
    )
    search_query: Optional[str] = Field(
        None, description="The search query for looking up internal KB documents. Required if action_type is 'search_knowledge_base'."
    )

class CustomerSupportObservation(Observation):
    ticket_id: str = Field(..., description="Unique identifier for the ticket")
    subject: str = Field(..., description="Subject of the ticket")
    message: str = Field(..., description="Body of the customer's message")
    department: str = Field("General", description="Current assigned department")
    status: str = Field("Open", description="Status of the ticket: Open or Resolved")
    system_response: Optional[str] = Field(None, description="System data provided to the agent, like search engine results.")
    
    # Required OpenEnv spec fields
    reward: float = Field(0.0, description="Reward received from the last action")
    done: bool = Field(False, description="Whether the episode has ended")
