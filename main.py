from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.sqlite import SqliteSaver
import base64
import glob
from pathlib import Path
import os
from typing import Any, Dict, Optional
import json
from datetime import datetime

# ===== CONTEXT MANAGEMENT WITH PERSISTENCE =====
class DisputeContext:
    """Manages conversation state with SQLite persistence"""
    
    def __init__(self, checkpointer, thread_id):
        self.checkpointer = checkpointer
        self.thread_id = thread_id
        self._state = self._load_state()
    
    def _load_state(self):
        """Load state from checkpointer or create default"""
        try:
            # Try to load existing state from checkpointer
            config = {"configurable": {"thread_id": self.thread_id}}
            checkpoint = self.checkpointer.get(config)
            
            if checkpoint and "context_state" in checkpoint.get("channel_values", {}):
                return checkpoint["channel_values"]["context_state"]
        except Exception as e:
            print(f"Could not load state: {e}")
        
        # Return default state
        return {
            "state": "initial",
            "user_role": None,
            "order_id": None,
            "evidence": {},
            "answers": {},
            "tools_used": [],
            "timestamp": datetime.now().isoformat()
        }
    
    def _save_state(self):
        """Save current state to checkpointer"""
        try:
            # We'll let the agent's natural checkpointing handle this
            # For now, just track in memory
            pass
        except Exception as e:
            print(f"Could not save state: {e}")
    
    def update_state(self, new_state: str):
        self._state["state"] = new_state
        self._state["timestamp"] = datetime.now().isoformat()
        self._save_state()
    
    def add_tool_usage(self, tool_name: str, args: dict, result: dict):
        """Track tool usage"""
        self._state["tools_used"].append({
            "tool": tool_name,
            "args": args,
            "result": result,
            "timestamp": datetime.now().isoformat()
        })
        self._save_state()
    
    def set_order_id(self, order_id: str):
        self._state["order_id"] = order_id
        self._save_state()
    
    def set_user_role(self, role: str):
        self._state["user_role"] = role
        self._save_state()
    
    def add_evidence(self, evidence: dict):
        self._state["evidence"] = evidence
        self._save_state()
    
    def add_answers(self, answers: str):
        self._state["answers"]["customer_responses"] = answers
        self._save_state()
    
    def get_summary(self) -> str:
        evidence_status = "Sufficient" if self._state["evidence"].get("confidence", 0) > 0.7 else "Insufficient"
        return f"""
=== DISPUTE CONTEXT STATE ===
Thread ID: {self.thread_id}
State: {self._state["state"]}
User Role: {self._state["user_role"] or 'Unknown'}
Order ID: {self._state["order_id"] or 'Not provided'}
Evidence Status: {evidence_status}
Tools Used: {len(self._state["tools_used"])}
Last Updated: {self._state["timestamp"]}
Ready for Verdict: {self.is_ready_for_verdict()}
============================="""
    
    def is_ready_for_verdict(self) -> bool:
        has_order = bool(self._state["order_id"])
        has_evidence = bool(self._state["evidence"]) and self._state["evidence"].get("confidence", 0) > 0.7
        has_answers = bool(self._state["answers"])
        return has_order and has_evidence and has_answers
    
    def get_state(self):
        return self._state.copy()

# Global context will be initialized per thread
dispute_context = None

# ===== UTILITY FUNCTIONS =====
def pause_order_completion(order_id=None):
    """Pause the order completion in your system."""
    if order_id:
        print(f"Order {order_id} paused (placeholder).")
    else:
        print("Order paused (placeholder).")

def trigger_ui_update():
    """Trigger a UI update so the front-end reflects the mediation state."""
    print("UI update triggered (placeholder).")

def collect_photos(paths: list[str] | None = None, dir: str | None = None):
    """Read one or more photos, base64-encode them, and return metadata."""
    exts = ["*.jpg", "*.jpeg", "*.png", "*.webp", "*.heic", "*.bmp"]

    resolved_paths: list[str] = []
    if paths:
        resolved_paths = paths
    else:
        search_dir = dir or os.getenv("IMAGES_DIR")
        if search_dir:
            p = Path(search_dir)
            for pattern in exts:
                for fp in p.glob(pattern):
                    resolved_paths.append(str(fp))
        else:
            resolved_paths = ["/home/nikhil/Desktop/unnamed (1).png", "/home/nikhil/Desktop/unnamed.png"]

    results: list[dict] = []
    for image_path in resolved_paths:
        try:
            with open(image_path, "rb") as f:
                image_bytes = f.read()
            image_b64 = base64.b64encode(image_bytes).decode("utf-8")
            results.append({"path": image_path, "b64": image_b64})
        except FileNotFoundError:
            results.append({"path": image_path, "b64": None, "error": "file_not_found"})
        except Exception as e:
            results.append({"path": image_path, "b64": None, "error": str(e)})
    return results

def _extract_json(text: str) -> dict | None:
    """Extract the first JSON object from a string safely."""
    import json, re
    match = re.search(r"\{[\s\S]*\}", text)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except Exception:
        return None

def _describe_with_vision(image_b64: str) -> dict:
    """Call a vision-capable model on the provided base64 image."""
    vision_llm = ChatOpenAI(model=VISION_MODEL, temperature=0)
    
    prompt_text = (
        "Analyze this delivery photo for packaging damage. Focus on:\n"
        "1. Packaging condition: intact, damaged, crushed, torn, wet, leaked\n"
        "2. Damage severity: none, minor, moderate, severe\n"
        "3. Product exposure: is the actual product visible or contaminated?\n"
        "4. Leak details: color, substance type, extent\n"
        "5. Photo quality: can you see the full package clearly?\n\n"
        
        "Return JSON with these exact fields:\n"
        "{\n"
        '  "summary": "brief description",\n'
        '  "packaging_colors": ["color1", "color2"],\n'
        '  "damage": {\n'
        '    "types": ["leakage", "crushing", "tearing"],\n'
        '    "severity": "none|minor|moderate|severe",\n'
        '    "areas": ["front", "corner", "bottom"],\n'
        '    "substance_color": ["red", "clear"],\n'
        '    "is_product_exposed": true/false\n'
        '  },\n'
        '  "confidence": 0.85,\n'
        '  "needs_more_photos": true/false\n'
        "}"
    )
    
    msg = HumanMessage(
        content=[
            {"type": "text", "text": prompt_text},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}},
        ]
    )
    
    try:
        res = vision_llm.invoke([msg])
        text = getattr(res, "content", str(res))
        data = _extract_json(text)
        if data:
            data["status"] = "ok"
            return data
    except Exception as e:
        print(f"Vision analysis error: {e}")
    
    return {
        "status": "error",
        "summary": "Could not analyze image",
        "packaging_colors": [],
        "damage": {
            "types": [],
            "severity": "unknown", 
            "areas": [],
            "substance_color": [],
            "is_product_exposed": False
        },
        "confidence": 0.0,
        "needs_more_photos": True
    }

def _merge_analyses(per_image: list[dict]) -> dict:
    """Aggregate multiple per-image analyses."""
    if not per_image:
        return {}
    
    # For simplicity, take the first analysis
    # In production, you'd merge multiple analyses
    return per_image[0]

# ===== TOOLS WITH CONTEXT TRACKING =====
@tool(parse_docstring=True)
def initiate_mediation_flow(order_id: str | None = None, user_role: str = "customer"):
    """Initialize the packaging dispute resolution process.

    Args:
        order_id: Order identifier for the disputed delivery
        user_role: Either 'customer' or 'driver' to identify who is reporting
    """
    global dispute_context
    
    if dispute_context:
        dispute_context.update_state("initiated")
        if order_id:
            dispute_context.set_order_id(order_id)
        dispute_context.set_user_role(user_role)
    
    pause_order_completion(order_id)
    trigger_ui_update()
    
    result = {
        "status": "initiated",
        "order_id": order_id,
        "user_role": user_role,
        "message": f"Dispute resolution started for order {order_id or 'TBD'}. Please share photos of the damaged package.",
        "next_action": "collect_evidence"
    }
    
    if dispute_context:
        dispute_context.add_tool_usage("initiate_mediation_flow", {"order_id": order_id, "user_role": user_role}, result)
        print(dispute_context.get_summary())
    
    return result

@tool(parse_docstring=True)
def collect_evidence(paths: list[str] | None = None, dir: str | None = None):
    """Analyze photos of damaged packaging to determine extent and cause of damage.

    Args:
        paths: Specific image file paths to analyze
        dir: Directory containing damage photos
    """
    global dispute_context
    
    photos = collect_photos(paths=paths, dir=dir)
    if not photos:
        result = {"status": "error", "reason": "no_photos_found", "next_action": "Please upload photos"}
        if dispute_context:
            dispute_context.add_tool_usage("collect_evidence", {"paths": paths, "dir": dir}, result)
            print(dispute_context.get_summary())
        return result

    valid = [p for p in photos if p.get("b64")]
    if not valid:
        result = {
            "status": "error", 
            "reason": "no_valid_photos", 
            "details": [p.get("error", "unknown") for p in photos],
            "next_action": "Please ensure photos are accessible"
        }
        if dispute_context:
            dispute_context.add_tool_usage("collect_evidence", {"paths": paths, "dir": dir}, result)
            print(dispute_context.get_summary())
        return result

    # Analyze each valid image
    per_image: list[dict] = []
    for p in valid:
        analysis = _describe_with_vision(p["b64"])
        analysis["path"] = p.get("path")
        per_image.append(analysis)

    # Merge analyses
    agg = _merge_analyses(per_image)
    
    # Generate summary
    dmg = agg.get("damage", {})
    colors = ", ".join(agg.get("packaging_colors", [])) or "unknown"
    types = ", ".join(dmg.get("types", [])) or "no visible damage"
    severity = dmg.get("severity", "unknown")
    confidence = agg.get("confidence", 0.0)
    
    description = f"Analysis: {types} (severity: {severity}) on {colors} packaging. Confidence: {confidence:.1%}"
    
    result = {
        "status": "ok",
        "description": description,
        "structured": agg,
        "confidence": confidence,
        "needs_more_photos": bool(agg.get("needs_more_photos")),
        "can_proceed_to_questions": confidence > 0.7,
        "analyzed_paths": [p.get("path") for p in valid],
    }
    
    if dispute_context:
        dispute_context.add_evidence(agg)
        dispute_context.update_state("evidence_collected")
        dispute_context.add_tool_usage("collect_evidence", {"paths": paths, "dir": dir}, result)
        print(dispute_context.get_summary())
    
    return result

@tool(parse_docstring=True)
def ask_follow_up_questions(target_role: str = "customer"):
    """Generate targeted questions based on evidence analysis.

    Args:
        target_role: 'customer' or 'driver' - who should answer
    """
    global dispute_context
    
    questions = [
        {
            "to": "customer",
            "question": "Have you consumed any of this product? If yes, are you experiencing any health issues?",
            "topic": "safety",
            "priority": "high"
        },
        {
            "to": "customer", 
            "question": "Please confirm you've set aside the product and won't consume it until resolution.",
            "topic": "safety",
            "priority": "high"
        },
        {
            "to": "customer",
            "question": "Did you notice any smell from the leaked substance? What does it smell like?",
            "topic": "contamination",
            "priority": "medium"
        },
        {
            "to": "customer",
            "question": "When did you first notice the damage - immediately upon delivery or later?",
            "topic": "timeline",
            "priority": "high"
        },
        {
            "to": "customer",
            "question": "How has this affected your meal/plans? Do you need immediate replacement?",
            "topic": "impact",
            "priority": "medium"
        }
    ]
    
    filtered_questions = [q for q in questions if q["to"] == target_role][:5]
    
    result = {
        "status": "ok",
        "questions": filtered_questions,
        "total_questions": len(filtered_questions),
        "next_action": "Please answer these questions to help determine the best resolution"
    }
    
    if dispute_context:
        dispute_context.update_state("questioning")
        dispute_context.add_tool_usage("ask_follow_up_questions", {"target_role": target_role}, result)
        print(dispute_context.get_summary())
    
    return result

@tool(parse_docstring=True)
def store_customer_responses(responses: str, order_id: str | None = None):
    """Store customer responses to follow-up questions.
    
    Args:
        responses: Customer's answers to the questions
        order_id: Order ID for this case
    """
    global dispute_context
    
    if dispute_context:
        dispute_context.add_answers(responses)
        if order_id:
            dispute_context.set_order_id(order_id)
        dispute_context.update_state("responses_collected")
    
    result = {
        "status": "stored",
        "responses": responses,
        "order_id": order_id,
        "message": "Responses recorded. Ready for verdict with order ID.",
        "next_action": "finalize_verdict"
    }
    
    if dispute_context:
        dispute_context.add_tool_usage("store_customer_responses", {"responses": responses, "order_id": order_id}, result)
        print(dispute_context.get_summary())
    
    return result

@tool(parse_docstring=True) 
def finalize_verdict(order_id: str):
    """Make final decision based on collected evidence and responses.

    Args:
        order_id: Order ID for this dispute (REQUIRED)
    """
    global dispute_context
    
    if not order_id:
        result = {
            "status": "error",
            "message": "Order ID is required to finalize verdict",
            "next_action": "Ask customer for order ID"
        }
        if dispute_context:
            dispute_context.add_tool_usage("finalize_verdict", {"order_id": order_id}, result)
            print(dispute_context.get_summary())
        return result
    
    # Get evidence from context if available
    evidence = {}
    if dispute_context:
        state = dispute_context.get_state()
        evidence = state.get("evidence", {})
        dispute_context.set_order_id(order_id)
    
    # Determine severity and exposure
    damage = evidence.get("damage", {})
    severity = damage.get("severity", "severe")  # Default to severe for demo
    is_exposed = damage.get("is_product_exposed", True)  # Default to exposed for demo
    
    # Decision logic
    if severity == "severe" or is_exposed:
        decision = "full_refund_replacement"
        payout_band = "high"
        rationale = "Severe damage with product exposure detected. Safety concern requires full compensation."
        next_steps = [
            "Full refund and replacement order will be processed within 30 minutes",
            "Keep damaged items for pickup by our team", 
            "Refund will be processed to original payment method within 1-2 business days"
        ]
    elif severity == "moderate":
        decision = "replacement"
        payout_band = "medium"
        rationale = "Moderate packaging damage confirmed. Product integrity compromised."
        next_steps = [
            "Replacement order will be processed within 30 minutes",
            "Keep damaged items for pickup by our team"
        ]
    elif severity == "minor":
        decision = "partial_refund"
        payout_band = "low"
        rationale = "Minor cosmetic damage only. Product remains functional and safe."
        next_steps = [
            "Partial refund of 50% will be processed within 24 hours",
            "Product is safe to consume despite cosmetic damage"
        ]
    else:
        decision = "deny"
        payout_band = "none"
        rationale = "No significant damage detected or insufficient evidence."
        next_steps = ["Claim reviewed and denied due to insufficient evidence"]
    
    trigger_ui_update()
    
    result = {
        "status": "finalized",
        "order_id": order_id,
        "decision": decision,
        "rationale": rationale,
        "payout_band": payout_band,
        "next_steps": next_steps,
        "resolution_time": datetime.now().isoformat()
    }
    
    if dispute_context:
        dispute_context.update_state("completed")
        dispute_context.add_tool_usage("finalize_verdict", {"order_id": order_id}, result)
        print(dispute_context.get_summary())
    
    return result

# ===== AGENT SETUP =====
load_dotenv()

if not os.getenv("OPENAI_API_KEY"):
    print("Missing OPENAI_API_KEY. Add it to a .env file in the project root.")
    raise SystemExit(1)

ASSISTANT_MODEL = os.getenv("ASSISTANT_MODEL", "gpt-4o-mini")
VISION_MODEL = os.getenv("VISION_MODEL", "gpt-4o-mini")

SYSTEM_MESSAGE = """You are GrabFood's AI Dispute Resolution Agent, specialized in resolving packaging damage claims.

**WORKFLOW**:
1. When customer reports damage, call `initiate_mediation_flow` 
2. When photo paths provided, call `collect_evidence`
3. After evidence collected (confidence >70%), call `ask_follow_up_questions`
4. When customer provides answers, call `store_customer_responses` 
5. When order_id available, call `finalize_verdict`

**DECISION FRAMEWORK**:
- SEVERE damage + product exposed = Full refund + replacement
- MODERATE damage = Replacement only  
- MINOR damage = Partial refund
- NO damage = Deny claim

**TOOL USAGE RULES**:
- ALWAYS call initiate_mediation_flow when damage is first reported
- ALWAYS call collect_evidence when photo paths are provided
- ALWAYS call store_customer_responses when customer answers questions
- ALWAYS call finalize_verdict with order_id when ready

Be proactive with tools - don't just ask for information, use the tools to collect and process it.
"""

llm = ChatOpenAI(model=ASSISTANT_MODEL, temperature=0.3)
tools = [initiate_mediation_flow, collect_evidence, ask_follow_up_questions, store_customer_responses, finalize_verdict]

if __name__ == "__main__":
    try:
        with SqliteSaver.from_conn_string(os.getenv("CHECKPOINT_DB", "state.db")) as memory:
            # Initialize context for this thread
            thread_id = os.getenv("THREAD_ID", "grabhack_demo")
            dispute_context = DisputeContext(memory, thread_id)
            
            agent_executor = create_react_agent(llm, tools, checkpointer=memory, prompt=SYSTEM_MESSAGE)
            config = {"configurable": {"thread_id": thread_id}}
            
            # Show initial context
            print("=== GrabFood Dispute Resolution Agent ===")
            print("Context Engineering Enhanced Version")
            print("==========================================")
            print(dispute_context.get_summary())
            
            demo_message = {
                "role": "user", 
                "content": """No, I haven’t consumed it, so no health issues.

Yes, I’ve set it aside and won’t consume it.

Yes, there’s a smell — it’s unpleasant, like spoiled food/oil.

I noticed the damage immediately upon delivery.

This has affected my meal plans — I would like a replacement as soon as possible."""
            }
            
            for step in agent_executor.stream(
                {"messages": [demo_message]}, config, stream_mode="values"
            ):
                step["messages"][-1].pretty_print()
            
            print("\n=== FINAL CONTEXT STATE ===")
            print(dispute_context.get_summary())
        
    except Exception as e:
        print(f"Agent execution failed: {e}")
    
    print("\nAgent run complete.")