from langchain_core.tools import tool
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
import base64


def pause_order_completion(order_id=None):
    """Pause the order completion in your system. Replace body with your implementation."""
    if order_id:
        print(f"Order {order_id} paused (placeholder).")
    else:
        print("Order paused (placeholder).")


def trigger_ui_update():
    """Trigger a UI update so the front-end reflects the mediation state."""
    print("UI update triggered (placeholder).")


def collect_photos():
    """Read a single example photo from disk, encode to base64 and return metadata.
    Replace the path or logic to fetch from your storage/service as needed.
    """
    image_path = "/home/ss141309/Downloads/spilt-package2.jpg"
    try:
        with open(image_path, "rb") as f:
            image_bytes = f.read()
        image_b64 = base64.b64encode(image_bytes).decode("utf-8")
        return {"path": image_path, "b64": image_b64}
    except FileNotFoundError:
        return {"path": image_path, "b64": None, "error": "file_not_found"}


@tool(parse_docstring=True)
def initiate_mediation_flow(order_id: str | None = None):
    """The user has issues with his order regarding the packaging and wants to resolve it — this tool only initializes the mediation process.

    Args:
        order_id: Optional order identifier to act upon.
    """
    pause_order_completion(order_id)
    trigger_ui_update()
    return {"status": "initiated", "order_id": order_id}


@tool(parse_docstring=True)
def collect_evidence():
    """Retrieve photos of the damaged package. Always call this when the user reports damaged or disputed packaging."""
    return collect_photos()


# --- Agent setup ---
load_dotenv()

llm = ChatOpenAI(model="gpt-5-mini", temperature=0.4)
tools = [initiate_mediation_flow, collect_evidence]
memory = MemorySaver()

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
4. If the user reports packaging damage, always call the `collect_evidence` tool to fetch the evidence photos before proceeding.
5. If tools cannot resolve the issue, suggest alternative solutions or escalate appropriately.
6. Do not invent or reuse canned sample descriptions. Base answers strictly on evidence returned by tools.
7. If no valid image is available or parsing fails, state that clearly and request a new upload.

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

config = {"configurable": {"thread_id": "abc456"}}

# Use only the real user request to avoid bias from sample few-shots.
real_user = {
    "role": "user",
    "content": (
        "I want you to describe the damage in the product and the colour of the packaging. "
        "Call the collect_evidence tool to fetch the image before answering."
    ),
}

messages = [real_user]


if __name__ == "__main__":
    try:
        for step in agent_executor.stream(
            {"messages": messages}, config, stream_mode="values"
        ):
            last = step.get("messages", [])[-1] if step.get("messages") else None
            if last:
                if isinstance(last, dict) and last.get("content"):
                    print(last["content"])
                else:
                    print(last)
    except Exception as e:
        print("Streaming failed:", e)

    print("Agent run complete.")