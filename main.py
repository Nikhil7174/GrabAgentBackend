from langchain_core.tools import tool
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
import base64


def pause_order_completion():
    pass


def trigger_ui_update():
    pass


def collect_photos():
    image_path = "/home/ss141309/Downloads/spilt-package.jpg"
    with open(image_path, "rb") as f:
        image_bytes = f.read()
    image_b64 = base64.b64encode(image_bytes).decode("utf-8")
    return {"path": image_path, "b64": image_b64}


@tool(parse_docstring=True)
def initiate_mediation_flow():
    """The user has issues with his order regarding the packaging and wants to resolve it, this tools only initializes the mediation process."""
    pause_order_completion()
    trigger_ui_update()


@tool(parse_docstring=True)
def collect_evidence():
    """Retrieve photos of the damaged package. Always call this when the user reports damaged or disputed packaging."""
    return collect_photos()


load_dotenv()

llm = ChatOpenAI(model="gpt-5-mini", temperature=0.4)
tools = [initiate_mediation_flow, collect_evidence]
memory = MemorySaver()

system_message = """You are a helpful AI assistant.
** Role & Purpose:
1. You are a chatbot designed to assist users with all types of queries.
2. You have access to specialized tools for handling packaging disputes and related problem resolution.

** Behavior Guidelines:
* General Queries:
1. Provide clear, accurate, and concise answers.
2. Be polite, professional, and approachable.

* Packaging Disputes & Related Issues:
1. Ask clarifying questions if the user’s concern is not fully clear.
2. Guide the user step by step through the resolution process.
3. Use the appropriate tools when necessary.
4. If the user reports packaging damage, always call the collect_evidence tool to fetch the evidence photos before proceeding.
5. If tools cannot resolve the issue, suggest alternative solutions or escalate appropriately.

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

config = {"configurable": {"thread_id": "abc123"}}

input_message = {
    "role": "user",
    "content": "I want you to describe the damage in the product, I also want you to describe the colour of the packaging",
}

for step in agent_executor.stream(
    {"messages": [input_message]}, config, stream_mode="values"
):
    step["messages"][-1].pretty_print()