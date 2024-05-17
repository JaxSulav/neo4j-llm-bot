from llm import llm
from langchain import hub
from langchain.tools import Tool
from langchain.prompts import PromptTemplate
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

# agent_prompt = hub.pull("hwchase17/react-chat")
agent_prompt = PromptTemplate.from_template(
    """
You are a movie expert providing information about movies.
Be as helpful as possible and return as much information as possible.
Do not answer any questions that do not relate to movies, actors or directors.

Do not answer any questions using your pre-trained knowledge, only use the information provided in the context. If the answer isn’t included in the provided context, refuse to answer the question and ask for more information.

TOOLS:
------

You have access to the following tools:

{tools}

To use a tool, please use the following format:

```
Thought: Do I need to use a tool? Yes
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
```
The Observation should not contain any sentences that gives away that you are having a series of converstion with youself. It should be coherent, concise and direct to the original question.

When you have a response to say to the Human, or if you do not need to use a tool, you MUST use the format:

```
Thought: Do I need to use a tool? No
Final Answer: [your response here]
```

Begin!

Previous conversation history:
{chat_history}

New input: {input}
{agent_scratchpad}
"""
)

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
