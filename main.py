from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.sqlite import SqliteSaver  # <-- add this
import base64
import glob
from pathlib import Path
import os
from typing import Any

def pause_order_completion(order_id=None):
    """Pause the order completion in your system. Replace body with your implementation."""
    if order_id:
        print(f"Order {order_id} paused (placeholder).")
    else:
        print("Order paused (placeholder).")


def trigger_ui_update():
    """Trigger a UI update so the front-end reflects the mediation state."""
    print("UI update triggered (placeholder).")


def collect_photos(paths: list[str] | None = None, dir: str | None = None):
    """Read one or more photos, base64-encode them, and return metadata.

    Args:
        paths: Optional explicit list of image paths to load.
        dir: Optional directory to scan for images (jpg, jpeg, png, webp, heic, bmp).

    Returns:
        A list of dicts: [{"path": str, "b64": str}] or, on error, [{"path": str, "b64": None, "error": str}].

    Notes:
        - If both `paths` and `dir` are None, this falls back to env var `IMAGES_DIR` if present.
        - As a final fallback, it loads the single sample path currently in use.
    """
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
            # Fallback to the original single example
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


@tool(parse_docstring=True)
def initiate_mediation_flow(order_id: str | None = None):
    """The user has issues with his order regarding the packaging and wants to resolve it — this tool only initializes the mediation process.

    Args:
        order_id: Optional order identifier to act upon.
    """
    pause_order_completion(order_id)
    trigger_ui_update()
    return {"status": "initiated", "order_id": order_id}


def _extract_json(text: str) -> dict | None:
    """Extract the first JSON object from a string safely without extra deps."""
    import json, re
    match = re.search(r"\{[\s\S]*\}", text)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except Exception:
        return None


def _describe_with_vision(image_b64: str) -> dict:
    """Call a vision-capable model on the provided base64 image and return structured facts.

    Returns a dict with keys: status, summary, packaging_colors, labels_or_receipts,
    damage, confidence, needs_more_photos, recommended_shots, uncertainty_reasons, notes.
    Falls back to a plain-text summary if JSON parsing fails.
    """
    vision_llm = ChatOpenAI(model=VISION_MODEL, temperature=0)
    schema = {
        "type": "object",
        "properties": {
            "summary": {"type": "string"},
            "packaging_colors": {"type": "array", "items": {"type": "string"}},
            "labels_or_receipts": {
                "type": "object",
               "properties": {
                    "present": {"type": "boolean"},
                    "description": {"type": "string"},
                },
                "required": ["present", "description"],
            },
            "damage": {
                "type": "object",
                "properties": {
                    "types": {"type": "array", "items": {"type": "string"}},
                    "severity": {"type": "string", "enum": ["none", "minor", "moderate", "severe"]},
                    "areas": {"type": "array", "items": {"type": "string"}},
                    "substance_color": {"type": "array", "items": {"type": "string"}},
                    "is_product_exposed": {"type": "boolean"},
                },
                "required": ["types", "severity", "areas", "substance_color", "is_product_exposed"],
            },
            "notes": {"type": "string"},
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
            "needs_more_photos": {"type": "boolean"},
            "recommended_shots": {"type": "array", "items": {"type": "string"}},
            "uncertainty_reasons": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["summary", "packaging_colors", "labels_or_receipts", "damage", "confidence", "needs_more_photos"],
    }
    prompt_text = (
        "Analyze this delivery photo strictly from visible evidence. "
        "Identify: packaging color(s); presence of labels/receipts; damage types (e.g., leakage, spill, crushing, tearing, soaking), "
        "severity; affected areas; color(s) of any leaked substance; whether contents appear exposed. "
        "Use precise descriptors like 'front-left corner', 'top fold', 'seam', 'receipt area'. "
        "Also include a numeric 'confidence' (0–1), a boolean 'needs_more_photos' if one image is insufficient, "
        "and, when low confidence, provide 'recommended_shots' with specific angles (e.g., whole package, opposite corner, contents out of box, close-up of damage) "
        "and 'uncertainty_reasons'. Respond ONLY with JSON matching this schema and do not include extra text: "
        "The confidence parameter should be calculated as follows:\n"
        "1. It should be between 0 to 1.\n"
        "2. Is the full view of the package (all four corners) visible.\n"
        "3. The extent of the damage.\n"
        "4. The overall quality of the photo.\n"
        "only recommend those shots which are not present in this array [Full view of the whole outer packaging, Opposite corner from the damage (to assess extent),"
        "Close-up of the damaged area (2-3 inches away)"
        f"{schema}"
    )
    msg = HumanMessage(
        content=[
            {"type": "text", "text": prompt_text},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}},
        ]
    )
    res = vision_llm.invoke([msg])
    text = getattr(res, "content", str(res))
    data = _extract_json(text)
    if not data:
        # Fallback: return plain text in summary
        return {"status": "ok", "summary": text, "packaging_colors": [], "labels_or_receipts": {"present": False, "description": ""}, "damage": {"types": [], "severity": "unknown", "areas": [], "substance_color": [], "is_product_exposed": False}, "notes": "model returned non-JSON; used summary fallback", "confidence": 0.4, "needs_more_photos": True, "recommended_shots": ["whole outer packaging", "opposite corner", "close-up of damaged area", "product outside box"], "uncertainty_reasons": ["no parseable JSON"]}
    data["status"] = "ok"
    return data


def _severity_rank(label: str) -> int:
    order = {"none": 0, "minor": 1, "moderate": 2, "severe": 3}
    return order.get((label or "").lower(), 1)


def _merge_analyses(per_image: list[dict]) -> dict:
    """Aggregate multiple per-image analyses into a single verdict with guided recommendations."""
    colors = set()
    damage_types = set()
    areas = set()
    substance_colors = set()
    exposed = False
    notes: list[str] = []
    confidences: list[float] = []
    needs_more_any = False
    rec_shots: list[str] = []
    uncertainty: list[str] = []

    for d in per_image:
        colors.update([c for c in d.get("packaging_colors", []) if c])
        dmg = d.get("damage", {})
        damage_types.update([t for t in dmg.get("types", []) if t])
        areas.update([a for a in dmg.get("areas", []) if a])
        substance_colors.update([s for s in dmg.get("substance_color", []) if s])
        exposed = exposed or bool(dmg.get("is_product_exposed"))
        if d.get("notes"):
            notes.append(d["notes"])
        if isinstance(d.get("confidence"), (int, float)):
            confidences.append(float(d["confidence"]))
        needs_more_any = needs_more_any or bool(d.get("needs_more_photos"))
        rec_shots.extend(d.get("recommended_shots", []) or [])
        uncertainty.extend(d.get("uncertainty_reasons", []) or [])

    # Pick the highest severity seen across images
    severities = [d.get("damage", {}).get("severity", "unknown") for d in per_image]
    if severities:
        severities_sorted = sorted(severities, key=_severity_rank, reverse=True)
        agg_severity = severities_sorted[0]
    else:
        agg_severity = "unknown"

    # Confidence: average across images, capped into [0,1]
    avg_conf = sum(confidences) / len(confidences) if confidences else 0.5
    avg_conf = max(0.0, min(1.0, avg_conf))

    return {
        "summary": None,  # caller composes summary
        "packaging_colors": sorted(colors),
        "labels_or_receipts": None,  # not aggregated reliably here
        "damage": {
            "types": sorted(damage_types),
            "severity": agg_severity,
            "areas": sorted(areas),
            "substance_color": sorted(substance_colors),
            "is_product_exposed": exposed,
        },
        "confidence": avg_conf,
        "needs_more_photos": avg_conf <= 0.7,
        "recommended_shots": rec_shots,
        "uncertainty_reasons": list(dict.fromkeys(uncertainty)),
        "notes": "; ".join(notes[:3]) if notes else "",
        "per_image": per_image,
    }


@tool(parse_docstring=True)
def collect_evidence(paths: list[str] | None = None, dir: str | None = None):
    """Retrieve and analyze one or more photos of the damaged package.

    Args:
        paths: Optional explicit list of image paths to analyze.
        dir: Optional directory to scan for images if `paths` is not provided.

    Returns:
        - status: 'ok' or 'error'
        - description: concise natural-language summary of aggregated evidence
        - structured: aggregated JSON including per-image analyses
        - needs_more_photos: boolean indicating whether more angles are requested
        - requested_shots: suggested additional angles if more photos are needed
    """
    photos = collect_photos(paths=paths, dir=dir)
    if not photos:
        return {"status": "error", "reason": "no_photos_found"}

    # If any photo failed to load and no valid ones exist, error out with details
    valid = [p for p in photos if p.get("b64")]
    if not valid:
        return {"status": "error", "reason": ", ".join({p.get("error", "unknown") for p in photos}), "paths": [p.get("path") for p in photos]}

    # Analyze each valid image
    per_image: list[dict] = []
    for p in valid:
        analysis = _describe_with_vision(p["b64"])  # dict with damage/colors/etc.
        analysis["path"] = p.get("path")
        per_image.append(analysis)

    agg = _merge_analyses(per_image)

    # Compose concise description from aggregated data
    dmg = agg.get("damage", {})
    colors = ", ".join(agg.get("packaging_colors", [])) or "unknown"
    types = ", ".join(dmg.get("types", [])) or "unknown damage"
    areas = ", ".join(dmg.get("areas", [])) or "unspecified area"
    severity = dmg.get("severity", "unknown")
    substance = ", ".join(dmg.get("substance_color", [])) or "unspecified color"
    summary = agg.get("summary") or (
        f"Packaging color: {colors}. Visible damage: {types} (severity: {severity}) at {areas}. "
        f"Leaked substance color: {substance}."
    )

    return {
        "status": "ok",
        "description": summary,
        "structured": agg,
        "needs_more_photos": bool(agg.get("needs_more_photos")),
        "requested_shots": agg.get("recommended_shots", []),
        "uncertainty_reasons": agg.get("uncertainty_reasons", []),
        "analyzed_paths": [p.get("path") for p in valid],
    }


@tool(parse_docstring=True)
def request_more_photos(reason: str | None = None, recommended_shots: list[str] | None = None):
    """Ask the user to upload additional photos/angles when one image is insufficient for a reliable verdict.

    Args:
        reason: Optional short sentence explaining why more photos are needed.
        recommended_shots: Optional list of specific angles or frames to capture.

    Returns:
        A dict with a user-facing message and a structured list of requested shots.
    """
    default_shots = [
        "Full view of the whole outer packaging",
        "Opposite corner from the damage (to assess extent)",
        "Close-up of the damaged area (2-3 inches away)",
    ]
    shots = recommended_shots if recommended_shots else default_shots
    msg = (
        "To give a precise verdict, I need a few more angles. "
        + (f"Reason: {reason}. " if reason else "")
        + "Please upload the following shots: " + "; ".join(shots) + "."
    )
    trigger_ui_update()  # placeholder: could open an upload prompt in a real app
    return {"status": "requested", "message": msg, "requested_shots": shots}

@tool(parse_docstring=True)
def ask_dynamic_questions(
    analysis: dict,
    role: str = "user",
    extra_context: dict | None = None,
    max_questions: int = 5
):
    """Create targeted, role-specific follow-up questions to finalize a packaging-damage verdict.

    Args:
        analysis: Aggregated output produced by collect_evidence -> structured 'agg' dict.
        role: 'user' or 'driver' (controls phrasing and topics).
        extra_context: Optional dict with known metadata (e.g., order_id, product type, perishability).
        max_questions: Upper bound on number of questions to return.

    Returns:
        {
          "status": "ok",
          "questions": [
            {
              "to": "user" | "driver",
              "question": "string",
              "why": "string",
              "priority": 1..3,             # 1 = highest
              "blocking": bool,             # must be answered before verdict?
              "topic": "evidence|liability|resolution|safety"
            },
            ...
          ],
          "notes": "string"
        }
    """
    # ---- deterministic scaffolding (reliable, model-agnostic) ----
    dmg = (analysis or {}).get("damage", {}) or {}
    types = set((dmg.get("types") or []))
    severity = (dmg.get("severity") or "unknown").lower()
    areas = set((dmg.get("areas") or []))
    substance_colors = set((dmg.get("substance_color") or []))
    is_exposed = bool(dmg.get("is_product_exposed"))
    needs_more = bool((analysis or {}).get("needs_more_photos"))
    conf = float((analysis or {}).get("confidence") or 0.0)
    rec_shots = (analysis or {}).get("recommended_shots") or []
    uncertainty = (analysis or {}).get("uncertainty_reasons") or []
    packaging_colors = (analysis or {}).get("packaging_colors") or []

    role = (role or "user").lower()
    extra_context = extra_context or {}

    def q(item_to:str, question:str, why:str, priority:int=2, blocking:bool=False, topic:str="evidence"):
        return {
            "to": item_to,
            "question": question.strip(),
            "why": why.strip(),
            "priority": max(1, min(priority, 3)),
            "blocking": bool(blocking),
            "topic": topic
        }

    questions: list[dict[str, Any]] = []

    # --- Evidence gaps & angle coverage ---
    if needs_more or conf <= 0.7:
        if rec_shots:
            questions.append(q(
                "user",
                f"Can you share {', '.join(rec_shots[:3])}?",
                "Image coverage is incomplete; targeted photos will materially raise confidence.",
                priority=1, blocking=True, topic="evidence"
            ))
        else:
            questions.append(q(
                "user",
                "Could you share a full shot of the entire outer packaging and a close-up of the damaged area?",
                "We need standard angles to validate extent and exact location.",
                priority=1, blocking=True, topic="evidence"
            ))

    # --- Damage-specific questions (user) ---
    if "leakage" in {t.lower() for t in types} or "spill" in {t.lower() for t in types}:
        questions.append(q(
            "user",
            "Is the leaked substance sticky, oily, or watery, and does it have any noticeable odor?",
            "Characterizing the substance helps distinguish transit leakage vs. condensation or pre-existing spill.",
            priority=1, blocking=False, topic="evidence"
        ))
        if not is_exposed:
            questions.append(q(
                "user",
                "Is the inner product seal intact when you open the package?",
                "Determines contamination risk and severity classification.",
                priority=1, blocking=True, topic="safety"
            ))

    if "crushing" in {t.lower() for t in types}:
        questions.append(q(
            "user",
            "Are the contents inside deformed or only the outer box is crushed?",
            "Affects compensation band: cosmetic vs. functional damage.",
            priority=2, blocking=False, topic="evidence"
        ))

    if "tearing" in {t.lower() for t in types} or "soaking" in {t.lower() for t in types}:
        questions.append(q(
            "user",
            "Did you notice any moisture on the inner packaging or product?",
            "Confirms ingress reaching product vs. outer wrap only.",
            priority=2, blocking=False, topic="evidence"
        ))

    if substance_colors and "unknown" not in substance_colors:
        questions.append(q(
            "user",
            f"Does the leaked color match any product inside (we observed: {', '.join(sorted(substance_colors))})?",
            "Cross-check leak origin with product content.",
            priority=2, blocking=False, topic="evidence"
        ))

    # --- Logistics chain questions (driver) ---
    # Only add when severity ≥ moderate or confidence low (we need chain-of-custody clarity)
    if severity in ("moderate", "severe") or conf <= 0.7:
        questions.append(q(
            "driver",
            "At pickup, was the parcel visually intact and dry? Any pre-existing dents, tears, or damp patches?",
            "Establish baseline condition at handover.",
            priority=1, blocking=False, topic="liability"
        ))
        questions.append(q(
            "driver",
            "During transit, was the package stored upright and secured from tilting or compression by heavier parcels?",
            "Crushing/tilt can plausibly cause the observed damage types.",
            priority=2, blocking=False, topic="liability"
        ))
        if "leakage" in {t.lower() for t in types} or "spill" in {t.lower() for t in types}:
            questions.append(q(
                "driver",
                "Did you notice any leakage or dampness in the vehicle or on adjacent parcels after pickup?",
                "Helps time-bound when the leak occurred.",
                priority=2, blocking=False, topic="liability"
            ))

    # --- Documentation / traceability (both sides) ---
    questions.append(q(
        "user",
        "Do you have the order ID and any receipt or label photo you can share?",
        "Links evidence to a specific shipment and SKU; needed for claims processing.",
        priority=1, blocking=True, topic="resolution"
    ))

    questions.append(q(
        "driver",
        "Please confirm the pickup and delivery timestamps and route segment (hub handovers, if any).",
        "Narrows the window when damage likely occurred.",
        priority=3, blocking=False, topic="liability"
    ))

    # --- Safety checks if product looks exposed or is perishable / hazardous ---
    perishable = bool(extra_context.get("perishable"))
    hazardous = bool(extra_context.get("hazardous"))
    if is_exposed or perishable or hazardous:
        questions.append(q(
            "user",
            "Please avoid using the product until our check is done. Is the item perishable or hazardous in any way?",
            "Safety gating prior to verdict.",
            priority=1, blocking=True, topic="safety"
        ))

    # Cap, sort by priority (1 first)
    questions = sorted(questions, key=lambda x: x["priority"])[:max_questions]

    # ---- Optional LLM pass to polish phrasing (keeps content intact) ----
    # Keep this resilient: if model call fails, return as-is.
    try:
        llm = ChatOpenAI(model=ASSISTANT_MODEL, temperature=0.2)
        schema = {
            "type": "object",
            "properties": {
                "questions": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "to": {"type": "string", "enum": ["user","driver"]},
                            "question": {"type": "string"},
                            "why": {"type": "string"},
                            "priority": {"type": "integer"},
                            "blocking": {"type": "boolean"},
                            "topic": {"type": "string"}
                        },
                        "required": ["to","question","why","priority","blocking","topic"]
                    }
                }
            },
            "required": ["questions"]
        }
        prompt = (
            "Rewrite the following question list for clarity and brevity. Keep semantics and fields identical. "
            "Return ONLY JSON in the provided schema:\n"
            f"{schema}\n\n"
            f"Questions:\n{questions}"
        )
        resp = llm.invoke([HumanMessage(content=prompt)])
        polished = _extract_json(getattr(resp, "content", str(resp)))
        if polished and "questions" in polished and isinstance(polished["questions"], list):
            questions = polished["questions"]
    except Exception:
        pass

    return {"status": "ok", "questions": questions, "notes": "Auto-generated based on current evidence gaps"}


@tool(parse_docstring=True)
def finalize_verdict(
    decision: str,
    rationale: str,
    payout_band: str | None = None,
    next_steps: list[str] | None = None,
    analysis: dict | None = None,
    order_id: str | None = None,
):
    """Record the final decision for this case and trigger any follow-up actions.

    Args:
        decision: 'refund' | 'replace' | 'partial_refund' | 'deny'
        rationale: Short justification referencing the collected evidence.
        payout_band: Optional internal band/slug for compensation handling.
        next_steps: Optional user-facing steps (pickup/disposal, replacement ETA, etc.).
        analysis: Optional echo of the aggregated analysis used for the verdict.
        order_id: Optional order id to log/route.
    """
    # In your real system, persist to DB/ticketing here:
    trigger_ui_update()
    return {
        "status": "finalized",
        "order_id": order_id,
        "decision": decision,
        "rationale": rationale,
        "payout_band": payout_band,
        "next_steps": next_steps or [],
        "analysis": analysis,
    }

# --- Agent setup ---
load_dotenv()

# Validate OpenAI credentials early to provide a clear error message.
if not os.getenv("OPENAI_API_KEY"):
    print(
        "Missing OPENAI_API_KEY. Add it to a .env file in the project root "
        "(OPENAI_API_KEY=sk-...), or export it in your shell before running."
    )
    raise SystemExit(1)

# Use a valid, vision-capable default. Allow overrides via env.
ASSISTANT_MODEL = os.getenv("ASSISTANT_MODEL", "gpt-4o-mini")
VISION_MODEL = os.getenv("VISION_MODEL", "gpt-4o-mini")
llm = ChatOpenAI(model=ASSISTANT_MODEL, temperature=0.4)
tools = [initiate_mediation_flow, collect_evidence, request_more_photos, ask_dynamic_questions, finalize_verdict]
# memory = MemorySaver()
memory = SqliteSaver.from_conn_string(os.getenv("CHECKPOINT_DB", "state.db"))

# UPDATED: add tiny state policy to the system message
system_message = """You are a helpful AI assistant.
** Role & Purpose:
1. You are a chatbot designed to assist users with all types of queries.
2. You have access to specialized tools for handling packaging disputes and related problem resolution.

** Behavior Guidelines:
*** General Queries:
1. Provide clear, accurate, and concise answers.
2. Be polite, professional, and approachable.

*** Packaging Disputes & Related Issues:
1. Ask clarifying questions if the user’s concern is not fully clear.
2. Guide the user step by step through the resolution process.
3. Use the appropriate tools when necessary.
4. If the user reports packaging damage, always call the `collect_evidence` tool to fetch and analyze available photos before proceeding. The tool supports multiple photos.
5. If tools cannot resolve the issue, suggest alternative solutions or escalate appropriately.
6. Do not invent or reuse canned sample descriptions. Base answers strictly on evidence returned by tools.
7. If no valid image is available or parsing fails, state that clearly and request a new upload.
8. If `collect_evidence` returns `needs_more_photos=true` or shows `confidence < 0.7`, call `request_more_photos` with the provided reasons and recommended shots. Provide any preliminary observations, then wait for new photos before issuing a final verdict.
9. **If `collect_evidence` indicates `confidence > 0.7`, proceed to a decision by calling `finalize_verdict`, referencing the aggregated `structured` analysis (severity, exposed/not, types). Do not ask for photos again unless new uncertainty is introduced.**

** Tone & Style:
1. Maintain a professional, empathetic, and solution-oriented tone.
2. Be proactive in helping users reach a resolution.
3. Avoid unnecessary jargon—keep responses user-friendly.

** Limitations:
1. If a request is outside your scope, politely inform the user and redirect them to the best alternative option.
"""

agent_executor = create_react_agent(
    llm, tools, checkpointer=memory, prompt=system_message
)

thread_id = os.getenv("THREAD_ID", "abc456")
config = {"configurable": {"thread_id": thread_id}}

# Use only the real user request to avoid bias from sample few-shots.
real_user = {
    "role": "user",
    "content": (
        "please proceed"
    ),
}

messages = [real_user]


if __name__ == "__main__":
    try:
        with SqliteSaver.from_conn_string(os.getenv("CHECKPOINT_DB", "state.db")) as memory:
            agent_executor = create_react_agent(
                llm, tools, checkpointer=memory, prompt=system_message
            )
            thread_id = os.getenv("THREAD_ID", "abc456")
            config = {"configurable": {"thread_id": thread_id}}

            for step in agent_executor.stream(
                    {"messages": messages}, config, stream_mode="values"
            ):
                step["messages"][-1].pretty_print()
    except Exception as e:
        print("Streaming failed:", e)

    print("Agent run complete.")
