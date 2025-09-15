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
import sqlite3
from datetime import datetime

# ===== ENHANCED CONTEXT MANAGEMENT WITH PROPER PERSISTENCE =====
class DisputeContext:
    """Manages conversation state with proper SQLite persistence"""
    
    def __init__(self, checkpointer, thread_id):
        self.checkpointer = checkpointer
        self.thread_id = thread_id
        self.db_path = "dispute_context.db"
        self._init_db()
        self._state = self._load_state()
    
    def _init_db(self):
        """Initialize SQLite database for context persistence"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS dispute_contexts (
                    thread_id TEXT PRIMARY KEY,
                    state_json TEXT,
                    last_updated TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS context_backups (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    thread_id TEXT,
                    state_json TEXT,
                    backup_timestamp TEXT,
                    reason TEXT
                )
            """)
            conn.commit()
    
    def _load_state(self):
        """Load state from SQLite database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(
                    "SELECT state_json FROM dispute_contexts WHERE thread_id = ?", 
                    (self.thread_id,)
                )
                row = cursor.fetchone()
                
                if row:
                    state_data = json.loads(row[0])
                    print(f"✅ Loaded existing context for thread: {self.thread_id}")
                    print(f"📊 Previous state: {state_data.get('state', 'unknown')}")
                    return state_data
                    
        except Exception as e:
            print(f"⚠️ Could not load state: {e}")
        
        # Return default state
        print(f"🆕 Creating new context for thread: {self.thread_id}")
        return {
            "state": "initial",
            "user_role": None,
            "order_id": None,
            "evidence": {},
            "answers": {},
            "tools_used": [],
            "timestamp": datetime.now().isoformat(),
            "session_count": 1
        }
    
    def _save_state(self):
        """Save current state to SQLite database with backup"""
        try:
            self._state["timestamp"] = datetime.now().isoformat()
            
            with sqlite3.connect(self.db_path) as conn:
                # Create backup of previous state
                cursor = conn.execute(
                    "SELECT state_json FROM dispute_contexts WHERE thread_id = ?", 
                    (self.thread_id,)
                )
                row = cursor.fetchone()
                if row:
                    conn.execute("""
                        INSERT INTO context_backups 
                        (thread_id, state_json, backup_timestamp, reason) 
                        VALUES (?, ?, ?, ?)
                    """, (
                        self.thread_id,
                        row[0],
                        datetime.now().isoformat(),
                        "auto_backup_before_update"
                    ))
                
                # Save current state
                conn.execute("""
                    INSERT OR REPLACE INTO dispute_contexts 
                    (thread_id, state_json, last_updated) 
                    VALUES (?, ?, ?)
                """, (
                    self.thread_id,
                    json.dumps(self._state, indent=2),
                    datetime.now().isoformat()
                ))
                conn.commit()
                
        except Exception as e:
            print(f"❌ Could not save state: {e}")
    
    def update_state(self, new_state: str):
        """Update the current state and persist"""
        old_state = self._state["state"]
        self._state["state"] = new_state
        self._save_state()
        print(f"🔄 State transition: {old_state} → {new_state}")
    
    def add_tool_usage(self, tool_name: str, args: dict, result: dict):
        """Track tool usage and persist"""
        self._state["tools_used"].append({
            "tool": tool_name,
            "args": args,
            "result": result,
            "timestamp": datetime.now().isoformat()
        })
        self._save_state()
        print(f"🔧 Tool used: {tool_name}")
    
    def set_order_id(self, order_id: str):
        """Set order ID and persist"""
        old_id = self._state["order_id"]
        self._state["order_id"] = order_id
        self._save_state()
        if old_id != order_id:
            print(f"📝 Order ID {'updated' if old_id else 'set'}: {order_id}")
    
    def set_user_role(self, role: str):
        """Set user role and persist"""
        self._state["user_role"] = role
        self._save_state()
        print(f"👤 User role set: {role}")
    
    def add_evidence(self, evidence: dict):
        """Add evidence and persist"""
        self._state["evidence"] = evidence
        self._save_state()
        print(f"📸 Evidence added (confidence: {evidence.get('confidence', 0):.1%})")
    
    def add_answers(self, answers: str):
        """Add customer answers and persist"""
        if "customer_responses" not in self._state["answers"]:
            self._state["answers"]["customer_responses"] = answers
        else:
            # Append if there are already responses
            self._state["answers"]["customer_responses"] += "\n\n" + answers
        self._save_state()
        print(f"💬 Customer responses recorded")
    
    def get_summary(self) -> str:
        """Get formatted context summary"""
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
Session Count: {self._state.get('session_count', 1)}
============================="""
    
    def is_ready_for_verdict(self) -> bool:
        """Check if all required data is available for verdict"""
        has_order = bool(self._state["order_id"])
        has_evidence = bool(self._state["evidence"]) and self._state["evidence"].get("confidence", 0) > 0.7
        has_answers = bool(self._state["answers"].get("customer_responses"))
        return has_order and has_evidence and has_answers
    
    def get_state(self):
        """Get copy of current state"""
        return self._state.copy()
    
    def reset_context(self):
        """Reset context for new dispute (optional utility method)"""
        # Backup current state before reset
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO context_backups 
                (thread_id, state_json, backup_timestamp, reason) 
                VALUES (?, ?, ?, ?)
            """, (
                self.thread_id,
                json.dumps(self._state),
                datetime.now().isoformat(),
                "manual_reset"
            ))
            conn.commit()
        
        self._state = {
            "state": "initial",
            "user_role": None,
            "order_id": None,
            "evidence": {},
            "answers": {},
            "tools_used": [],
            "timestamp": datetime.now().isoformat(),
            "session_count": self._state.get('session_count', 0) + 1
        }
        self._save_state()
        print("🔄 Context reset - previous state backed up")
    
    def get_thread_history(self):
        """Get all available threads"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    SELECT thread_id, last_updated, 
                           json_extract(state_json, '$.state') as current_state,
                           json_extract(state_json, '$.order_id') as order_id
                    FROM dispute_contexts 
                    ORDER BY last_updated DESC
                """)
                return cursor.fetchall()
        except Exception as e:
            print(f"❌ Could not fetch thread history: {e}")
            return []
    
    def export_context(self, file_path: str = None):
        """Export context to JSON file for debugging"""
        if not file_path:
            file_path = f"context_export_{self.thread_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        try:
            export_data = {
                "thread_id": self.thread_id,
                "export_timestamp": datetime.now().isoformat(),
                "current_state": self._state,
                "thread_history": self.get_thread_history()
            }
            
            with open(file_path, 'w') as f:
                json.dump(export_data, f, indent=2)
            
            print(f"📤 Context exported to: {file_path}")
            return file_path
        except Exception as e:
            print(f"❌ Could not export context: {e}")
            return None
    
    def validate_state(self) -> bool:
        """Validate current state integrity"""
        required_keys = ["state", "user_role", "order_id", "evidence", "answers", "tools_used", "timestamp"]
        
        for key in required_keys:
            if key not in self._state:
                print(f"⚠️ Missing required key: {key}")
                return False
        
        # Validate state values - UPDATED with new state
        valid_states = ["initial", "initiated", "evidence_collected", "needs_more_photos", "questioning", "responses_collected", "completed"]
        if self._state["state"] not in valid_states:
            print(f"⚠️ Invalid state: {self._state['state']}")
            return False
        
        print("✅ State validation passed")
        return True

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
        search_dir = dir
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
        "5. Photo quality: can you see the full package clearly?\n"
        "6. Confidence score (0–1): Calculate confidence based on a weighted combination of factors — extent of visible damage, severity level, packaging condition, product exposure, and leak evidence. Adjust this score with a photo-quality modifier (lower confidence if the photo is unclear, higher if the photo is sharp and complete). The final score should reflect both how severe the damage is and how certain the assessment is, clamped between 0 and 1.\n\n"
        
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
        '  "confidence": [0.0-1.0],\n'
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

# ===== ENHANCED TOOLS WITH CONTEXT TRACKING =====
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
    needs_more_photos = bool(agg.get("needs_more_photos"))
    
    description = f"Analysis: {types} (severity: {severity}) on {colors} packaging. Confidence: {confidence:.1%}"
    
    result = {
        "status": "ok",
        "description": description,
        "structured": agg,
        "confidence": confidence,
        "needs_more_photos": needs_more_photos,
        "can_proceed_to_questions": confidence > 0.7 and not needs_more_photos,
        "analyzed_paths": [p.get("path") for p in valid],
    }
    
    # Add specific guidance for more photos if needed
    if needs_more_photos:
        result["photo_guidance"] = [
            "Take a closer shot of the damaged area",
            "Capture the package from different angles", 
            "Ensure good lighting to see damage details clearly",
            "Show the full package and the damaged portion",
            "Include any leaked substances or stains"
        ]
    
    if dispute_context:
        dispute_context.add_evidence(agg)
        
        # Update state based on photo sufficiency
        if needs_more_photos:
            dispute_context.update_state("needs_more_photos")
        else:
            dispute_context.update_state("evidence_collected")
            
        dispute_context.add_tool_usage("collect_evidence", {"paths": paths, "dir": dir}, result)
        print(dispute_context.get_summary())
    
    return result

@tool(parse_docstring=True)
def ask_follow_up_questions(target_role: str = "customer"):
    """Generate targeted questions based on evidence analysis.
    Only call this when evidence is sufficient (needs_more_photos = false).

    Args:
        target_role: 'customer' or 'driver' - who should answer
    """
    global dispute_context
    
    # Check if evidence is sufficient before proceeding
    if dispute_context:
        state = dispute_context.get_state()
        evidence = state.get("evidence", {})
        needs_more_photos = evidence.get("needs_more_photos", True)
        confidence = evidence.get("confidence", 0.0)
        
        if needs_more_photos:
            return {
                "status": "error",
                "reason": "insufficient_photos",
                "message": "More photos are needed before proceeding to questions",
                "next_action": "collect_additional_evidence"
            }
        
        if confidence <= 0.7:
            return {
                "status": "error", 
                "reason": "low_confidence",
                "message": f"Evidence confidence too low ({confidence:.1%}). More photos needed.",
                "next_action": "collect_additional_evidence"
            }
    
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
    
    # Check if we have order_id now
    needs_order_id = not (dispute_context and dispute_context.get_state().get("order_id"))
    
    result = {
        "status": "stored",
        "responses": responses,
        "order_id": order_id,
        "message": "Responses recorded.",
        "next_action": "Ask for order_id to proceed with finalization" if needs_order_id else "Ready for finalization with order_id",
        "needs_order_id": needs_order_id
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
    severity = damage.get("severity", "severe")  # Default to severe for safety
    is_exposed = damage.get("is_product_exposed", True)  # Default to exposed for safety
    
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
        "resolution_time": datetime.now().isoformat(),
        "evidence_summary": {
            "confidence": evidence.get("confidence", 0.0),
            "damage_severity": severity,
            "product_exposed": is_exposed
        }
    }
    
    if dispute_context:
        dispute_context.update_state("completed")
        dispute_context.add_tool_usage("finalize_verdict", {"order_id": order_id}, result)
        print(dispute_context.get_summary())
    
    return result

@tool(parse_docstring=True)
def get_context_help():
    """Get information about current context state and available commands."""
    global dispute_context
    
    if not dispute_context:
        return {"status": "error", "message": "No context available"}
    
    state = dispute_context.get_state()
    
    # Determine what can be done next
    next_actions = []
    current_state = state.get("state", "initial")
    
    if current_state == "initial":
        next_actions.append("Report damage to start dispute resolution")
    elif current_state == "initiated":
        next_actions.append("Provide photo paths or directory for evidence collection")
    elif current_state == "needs_more_photos":
        evidence = state.get("evidence", {})
        confidence = evidence.get("confidence", 0.0)
        next_actions.append(f"Provide additional photos (current confidence: {confidence:.1%})")
        next_actions.append("Take closer shots, different angles, or better lighting")
    elif current_state == "evidence_collected":
        next_actions.append("Answer follow-up questions about the damage")
    elif current_state == "questioning":
        next_actions.append("Provide answers to the safety and impact questions")
    elif current_state == "responses_collected":
        if not state.get("order_id"):
            next_actions.append("Provide order ID to finalize verdict")
        else:
            next_actions.append("Finalize verdict (all requirements met)")
    elif current_state == "completed":
        next_actions.append("Case completed - start new dispute or export context")
    
    result = {
        "status": "ok",
        "current_context": dispute_context.get_summary(),
        "thread_id": dispute_context.thread_id,
        "next_actions": next_actions,
        "ready_for_verdict": dispute_context.is_ready_for_verdict(),
        "available_commands": [
            "/reset - Reset current context",
            "/export - Export context to file", 
            "/threads - List all available threads",
            "/help - Show this help"
        ]
    }
    
    return result

# ===== AGENT SETUP =====
load_dotenv()

if not os.getenv("OPENAI_API_KEY"):
    print("Missing OPENAI_API_KEY. Add it to a .env file in the project root.")
    raise SystemExit(1)

ASSISTANT_MODEL = os.getenv("ASSISTANT_MODEL", "gpt-4o-mini")
VISION_MODEL = os.getenv("VISION_MODEL", "gpt-4o-mini")

SYSTEM_MESSAGE = """You are GrabFood's AI Dispute Resolution Agent, specialized in resolving packaging damage claims.

**CONTEXT AWARENESS**: You have persistent context across sessions. Check the current state and continue appropriately.

**WORKFLOW**:
1. When customer reports damage → call `initiate_mediation_flow`
2. When photo paths provided → call `collect_evidence` 
3. **IF `needs_more_photos` is true → ask for additional photos and call `collect_evidence` again**
4. **ONLY when `needs_more_photos` is false AND confidence >70% → call `ask_follow_up_questions`**
5. When customer answers questions → call `store_customer_responses`
6. When order_id available and all data collected → call `finalize_verdict`

**EVIDENCE COLLECTION LOGIC**:
- If evidence analysis returns `needs_more_photos: true`, politely ask for additional photos
- Explain what specific angles or details are needed (closer shots, different angles, better lighting)
- Continue collecting evidence until `needs_more_photos: false`
- Only proceed to questions when evidence is sufficient

**STATE-AWARE RESPONSES**:
- Always check current context state from the summary
- If state is "evidence_collected" but `needs_more_photos` is true → request more photos
- If state is "questioning" and user provides answers → store responses
- If state is "responses_collected" but no order_id → ask for order_id
- If ready for verdict → proceed with finalization
- If customer provides file paths, treat them as photo evidence

**DECISION FRAMEWORK**:
- SEVERE damage + product exposed = Full refund + replacement
- MODERATE damage = Replacement only  
- MINOR damage = Partial refund
- NO damage = Deny claim

**SPECIAL COMMANDS**:
- If user types "/help" → call `get_context_help`
- If user types "/reset" → suggest context reset
- If user types "/export" → suggest context export

Be proactive with tools, context-aware, and always maintain conversation flow based on current state.
"""

llm = ChatOpenAI(model=ASSISTANT_MODEL, temperature=0.3)
tools = [
    initiate_mediation_flow, 
    collect_evidence, 
    ask_follow_up_questions, 
    store_customer_responses, 
    finalize_verdict,
    get_context_help
]

# ===== INTERACTIVE MODE FUNCTIONS =====
def handle_special_commands(user_input: str) -> bool:
    """Handle special commands like /help, /reset, etc. Returns True if command was handled."""
    global dispute_context
    
    if not user_input.startswith('/'):
        return False
    
    command = user_input.lower().strip()
    
    if command == '/help':
        print("\n🆘 Available Commands:")
        print("  /help     - Show this help")
        print("  /reset    - Reset current dispute context")
        print("  /export   - Export context to JSON file")
        print("  /threads  - List all dispute threads")
        print("  /validate - Validate current context state")
        print("  /quit     - Exit the application")
        print("\n📋 Current Context:")
        if dispute_context:
            print(dispute_context.get_summary())
        return True
    
    elif command == '/reset':
        if dispute_context:
            confirm = input("⚠️  Are you sure you want to reset the context? (y/N): ").lower()
            if confirm == 'y':
                dispute_context.reset_context()
                print("✅ Context has been reset")
            else:
                print("❌ Reset cancelled")
        else:
            print("❌ No context to reset")
        return True
    
    elif command == '/export':
        if dispute_context:
            file_path = dispute_context.export_context()
            if file_path:
                print(f"✅ Context exported successfully")
            else:
                print("❌ Export failed")
        else:
            print("❌ No context to export")
        return True
    
    elif command == '/threads':
        if dispute_context:
            threads = dispute_context.get_thread_history()
            if threads:
                print("\n📋 Available Dispute Threads:")
                for i, (thread_id, last_updated, state, order_id) in enumerate(threads, 1):
                    order_info = f" (Order: {order_id})" if order_id else " (No Order ID)"
                    print(f"  {i}. {thread_id} - {state}{order_info}")
                    print(f"     Last Updated: {last_updated}")
            else:
                print("❌ No threads found")
        else:
            print("❌ No context available")
        return True
    
    elif command == '/validate':
        if dispute_context:
            is_valid = dispute_context.validate_state()
            if is_valid:
                print("✅ Context state is valid")
            else:
                print("❌ Context state validation failed")
        else:
            print("❌ No context to validate")
        return True
    
    elif command in ['/quit', '/exit', '/q']:
        print("👋 Goodbye!")
        return True
    
    else:
        print(f"❌ Unknown command: {command}")
        print("💡 Type /help for available commands")
        return True

def detect_file_paths(user_input: str) -> list[str]:
    """Detect if user input contains file paths"""
    import re
    
    # Pattern to match file paths (Unix/Linux style)
    path_pattern = r'(/[\w\-_\.\s\(\)]+)+\.(png|jpg|jpeg|gif|bmp|webp|heic)'
    matches = re.findall(path_pattern, user_input, re.IGNORECASE)
    
    if matches:
        # Extract just the full paths
        paths = []
        for match in re.finditer(path_pattern, user_input, re.IGNORECASE):
            paths.append(match.group(0))
        return paths
    
    return []

def preprocess_user_input(user_input: str) -> str:
    """Preprocess user input to handle file paths and context"""
    global dispute_context
    
    # Check for file paths
    file_paths = detect_file_paths(user_input)
    if file_paths:
        print(f"🔍 Detected {len(file_paths)} file path(s): {', '.join(file_paths)}")
        
        # If we're in initial state and files are provided, suggest evidence collection
        if dispute_context and dispute_context.get_state().get("state") == "initial":
            return f"I have photos of damaged packaging at these paths: {', '.join(file_paths)}"
        elif dispute_context and dispute_context.get_state().get("state") in ["initiated", "evidence_collected"]:
            return f"Here are photos of the damage: {', '.join(file_paths)}"
    
    return user_input

# ===== MAIN EXECUTION =====
if __name__ == "__main__":
    try:
        # Use separate DB for LangGraph checkpoints and our context
        langgraph_db = os.getenv("CHECKPOINT_DB", "langgraph_checkpoints.db")
        
        with SqliteSaver.from_conn_string(langgraph_db) as memory:
            # Initialize context for this thread  
            thread_id = os.getenv("THREAD_ID", "grabhack_demo168")
            dispute_context = DisputeContext(memory, thread_id)
            
            # Validate context on startup
            dispute_context.validate_state()
            
            agent_executor = create_react_agent(llm, tools, checkpointer=memory, prompt=SYSTEM_MESSAGE)
            config = {"configurable": {"thread_id": thread_id}}
            
            # Show initial context
            print("=== GrabFood Dispute Resolution Agent ===")
            print("Context Engineering Enhanced Version")
            print("==========================================")
            print(dispute_context.get_summary())
            print("\n💡 Type /help for commands or describe your packaging issue")
            print("📝 Example: 'Hi, I just received my GrabFood order and the packaging is damaged'")
            print("📁 You can also provide file paths directly: '/home/user/damage_photo.png'")
            print("="*60)
            
            # Interactive mode
            while True:
                try:
                    user_input = input("\n👤 User: ").strip()
                    
                    if not user_input:
                        continue
                    
                    # Handle special commands
                    if handle_special_commands(user_input):
                        if user_input.lower() in ['/quit', '/exit', '/q']:
                            break
                        continue
                    
                    # Preprocess input for file paths and context
                    processed_input = preprocess_user_input(user_input)
                    if processed_input != user_input:
                        print(f"🔄 Interpreted as: {processed_input}")
                    
                    print(f"\n🤖 Processing...")
                    
                    # Execute agent
                    messages_processed = False
                    for step in agent_executor.stream(
                        {"messages": [{"role": "user", "content": processed_input}]}, 
                        config, 
                        stream_mode="values"
                    ):
                        if step.get("messages"):
                            step["messages"][-1].pretty_print()
                            messages_processed = True
                    
                    if not messages_processed:
                        print("⚠️ No response generated. Please try again.")
                    
                    # Show updated context
                    print("\n" + "="*60)
                    print(dispute_context.get_summary())
                    
                    # Show helpful next steps
                    state = dispute_context.get_state()
                    current_state = state.get("state", "initial")
                    
                    if current_state == "initiated":
                        print("💡 Next: Provide photo paths or say 'I have photos at [path]'")
                    elif current_state == "needs_more_photos":
                        evidence = state.get("evidence", {})
                        confidence = evidence.get("confidence", 0.0)
                        print(f"📸 Next: More photos needed (current confidence: {confidence:.1%})")
                        print("💡 Try: Different angles, closer shots, better lighting")
                    elif current_state == "evidence_collected":
                        print("💡 Next: Answer the follow-up questions about safety and impact")
                    elif current_state == "responses_collected" and not state.get("order_id"):
                        print("💡 Next: Provide your order ID to complete the resolution")
                    elif dispute_context.is_ready_for_verdict():
                        print("💡 Ready: All information collected, proceeding with final verdict")
                    
                except KeyboardInterrupt:
                    print("\n\n🛑 Interrupted by user")
                    confirm = input("Do you want to exit? (y/N): ").lower()
                    if confirm == 'y':
                        break
                    else:
                        print("Continuing...")
                        continue
                        
                except Exception as e:
                    print(f"❌ Error processing request: {e}")
                    print("🔄 Please try again or type /help for assistance")
                    continue
            
            # Final export option
            if dispute_context and dispute_context.get_state().get("state") != "initial":
                export_option = input("\n💾 Would you like to export the context before exiting? (y/N): ").lower()
                if export_option == 'y':
                    dispute_context.export_context()
        
    except Exception as e:
        print(f"💥 Critical error: {e}")
        print("🔧 Please check your configuration and try again")
    
    finally:
        print("\n🏁 GrabFood Dispute Agent session ended")
        print("📊 Context has been automatically saved and will persist for next session")