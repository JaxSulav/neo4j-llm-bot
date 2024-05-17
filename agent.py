from llm import llm
from langchain import hub
from langchain.tools import Tool
from langchain.agents import AgentExecutor, create_react_agent
from langchain.chains.conversation.memory import ConversationBufferWindowMemory


class Tools:
    def __init__(self) -> None:
        self.tools_set_1 = [
            Tool.from_function(
                name="General Chat",
                description="For general chat not covered by other tools",
                func=llm.invoke,
                return_direct=True,
            )
        ]


class Agents:
    def __init__(self, llm, tool, prompt) -> None:
        self.react_agent = create_react_agent(llm, tool, prompt)


memory = ConversationBufferWindowMemory(
    memory_key="chat_history",
    k=5,
    return_messages=True,
)

tool_set_1 = Tools().tools_set_1

agent_prompt = hub.pull("hwchase17/react-chat")

react_agent = Agents(llm, tool_set_1, agent_prompt).react_agent

agent_executor = AgentExecutor(
    agent=react_agent, tools=tool_set_1, memory=memory, verbose=True
)


# * Main Handler
def generate_response(prompt):
    """
    Create a handler that calls the Conversational agent
    and returns a response to be rendered in the UI
    """

    response = agent_executor.invoke({"input": prompt})

    return response["output"]
