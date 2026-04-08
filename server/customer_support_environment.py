import uuid
import random
from typing import Dict, Any

try:
    from openenv.core.env_server import Environment
    from openenv.core.env_server.types import State
except ImportError:
    try:
        from core.env_server import Environment
        from core.env_server.types import State
    except ImportError:
        class Environment:
            def __init__(self): pass
        from dataclasses import dataclass
        @dataclass
        class State:
            episode_id: str
            step_count: int

from models import CustomerSupportAction, CustomerSupportObservation

# Example Tasks for Customer Support Environment
TASKS = [
    {
        "ticket_id": "TKT-100",
        "subject": "Can't login to my account",
        "message": "Hi, I have been trying to login to my dashboard but I keep getting a 'Password Incorrect' error. Help please!",
        "best_action": "assign",
        "best_department": "IT",
        "keywords": ["password", "reset", "login"]
    },
    {
        "ticket_id": "TKT-200",
        "subject": "Double charged on my credit card",
        "message": "I was looking at my bank statement and I got charged $49 twice this month for my subscription. I want a refund for the extra charge and an apology.",
        "best_action": "reply_and_resolve",
        "best_department": "Billing", # though it starts in General
        "keywords": ["sorry", "apologize", "refund", "$49"]
    },
    {
        "ticket_id": "TKT-300",
        "subject": "Why is the system acting weird and not showing my graphs?",
        "message": "Every time I go to the analytics tab, the browser freezes for 5 seconds and then the graphs show up completely blank. This has been happening since yesterday's update. FIX IT.",
        "best_action": "escalate",
        "best_department": "IT",
        "keywords": []
    }
]

KNOWLEDGE_BASE = {
    "refund": "Standard refunds process within 3-5 business days. Double charges are processed immediately.",
    "freeze": "Analytics graph freezing is a known P0 bug as of the recent update. Do not try to resolve, immediately escalate all tickets.",
    "password": "Password resets can be handled by IT manually if the self-service portal is down."
}

class CustomerSupportEnvironment(Environment):
    """
    A Customer Support Ticket Triage environment.
    
    The agent receives a ticket and must decide to assign it to the correct department, 
    resolve it with a proper reply, or escalate it based on its content.
    """

    def __init__(self):
        super().__init__()
        self._state = State(episode_id=str(uuid.uuid4()), step_count=0)
        self.current_task: Dict[str, Any] = {}
        self.ticket_id = ""
        self.subject = ""
        self.message = ""
        self.department = "General"
        self.status = "Open"
        self.system_response = None

        # Used to cycle through 3 tasks to provide reproducible evaluation
        self.task_idx = 0

    def reset(self) -> CustomerSupportObservation:
        self._state.step_count = 0
        self._state.episode_id = str(uuid.uuid4())
        
        # Cycle through our 3 predefined tasks (Easy, Medium, Hard)
        self.current_task = TASKS[self.task_idx % len(TASKS)]
        self.task_idx += 1
        
        self.ticket_id = self.current_task["ticket_id"]
        self.subject = self.current_task["subject"]
        self.message = self.current_task["message"]
        
        # Dynamic variable shuffling for TKT-200
        if self.ticket_id == "TKT-200":
            amount = random.choice([29, 49, 99, 149])
            self.message = self.message.replace("$49", f"${amount}")

        self.department = "General"
        self.status = "Open"
        self.system_response = None
        
        return CustomerSupportObservation(
            ticket_id=self.ticket_id,
            subject=self.subject,
            message=self.message,
            department=self.department,
            status=self.status,
            system_response=self.system_response,
            reward=0.0,
            done=False
        )

    def step(self, action: CustomerSupportAction) -> CustomerSupportObservation:
        self._state.step_count += 1
        
        reward = 0.0
        done = False
        message = self.message

        if self.status == "Resolved":
            return CustomerSupportObservation(
                ticket_id=self.ticket_id,
                subject=self.subject,
                message=self.message,
                department=self.department,
                status=self.status,
                system_response=self.system_response,
                reward=0.0,
                done=True
            )
            
        # Reset system response each turn unless it's a search
        self.system_response = None

        # Global Action: Search Knowledge Base
        if action.action_type == "search_knowledge_base":
            query = (action.search_query or "").lower()
            results = [val for key, val in KNOWLEDGE_BASE.items() if key in query]
            if results:
                self.system_response = "Search Results: " + " | ".join(results)
            else:
                self.system_response = "Search Results: No relevant documents found."
            
            # Searching provides a slight positive neutral token reward
            return CustomerSupportObservation(
                ticket_id=self.ticket_id,
                subject=self.subject,
                message=self.message,
                department=self.department,
                status=self.status,
                system_response=self.system_response,
                reward=0.05,
                done=False
            )

        # Task 1: "TKT-100" (Easy, Assign to IT)
        if self.current_task["ticket_id"] == "TKT-100":
            if action.action_type == "assign" and action.department == "IT":
                reward = 0.99
                done = True
                self.department = "IT"
                self.status = "Resolved"
            elif action.action_type == "assign":
                reward = 0.0
                self.department = action.department
                # Not done, wait for correct assignment or max steps
            else:
                reward = -0.5 # Penalty for bad action

        # Task 2: "TKT-200" (Medium, Reply & Resolve with correct context)
        elif self.current_task["ticket_id"] == "TKT-200":
            if action.action_type == "assign" and action.department == "Billing":
                reward = 0.2
                self.department = "Billing"
            elif action.action_type == "reply_and_resolve":
                # Check for keywords in reply
                reply = (action.reply_message or "").lower()
                has_apology = any(w in reply for w in ["sorry", "apologize"])
                has_refund = "refund" in reply
                
                if has_apology and has_refund:
                    reward = 0.99
                    done = True
                    self.status = "Resolved"
                else:
                    # Multi-turn interaction loop
                    if has_refund and not has_apology:
                        self.message += "\n\nCustomer: Thanks for the refund, but your service still caused me a lot of stress. Some apology would be nice!"
                    elif has_apology and not has_refund:
                        self.message += "\n\nCustomer: A simple sorry is not enough! Where is my refund??"
                    else:
                        self.message += "\n\nCustomer: You haven't helped me at all. Give me my money back and explain why this happened!"
                    
                    reward = -0.1
            else:
                reward = -0.5

        # Task 3: "TKT-300" (Hard, Escalate!)
        elif self.current_task["ticket_id"] == "TKT-300":
            if action.action_type == "escalate":
                reward = 0.99
                done = True
                self.status = "Resolved"
            elif action.action_type == "assign":
                reward = 0.1
                self.department = action.department
            else:
                reward = -0.5

        if self._state.step_count >= 8:
            done = True

        return CustomerSupportObservation(
            ticket_id=self.ticket_id,
            subject=self.subject,
            message=self.message,
            department=self.department,
            status=self.status,
            system_response=self.system_response,
            reward=reward,
            done=done
        )

    @property
    def state(self) -> State:
        return self._state
