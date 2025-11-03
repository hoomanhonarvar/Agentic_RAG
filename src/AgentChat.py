import asyncio
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_agentchat.agents import AssistantAgent
from autogen_core import CancellationToken
import dotenv
import os
from autogen_core.models import ModelInfo, ModelFamily
from autogen_core.tools import FunctionTool
from autogen_agentchat.ui import Console
from tools import Retrieval_Tool
from pydantic import BaseModel



dotenv.load_dotenv()
API_KEY=os.getenv("API_KEY")
BASE_URL=os.getenv("BASE_URL")
MODEL_NAME=os.getenv("MODEL_NAME")


class AgentRagResponse(BaseModel):
    thoughts: str
    answer: str
    retrieve_document: str
    
cancellation_token = CancellationToken()
model_info=ModelInfo(
    vision=False,
    function_calling=True,
    json_output=True,
    family=ModelFamily.UNKNOWN,
    structured_output=True,
    )

model_client = OpenAIChatCompletionClient(
        model=MODEL_NAME,
        api_key=API_KEY,
        base_url=BASE_URL,
        model_info=model_info
    )

RAG_tool = FunctionTool(Retrieval_Tool, description="Retrieving information about places in the world to travel",strict=True)
agent = AssistantAgent(
        name="assistant",
        model_client=model_client,
        model_client_stream=True,
        tools=[RAG_tool],
        system_message="""You are an intelligent assistant that helps solve user tasks.
                            - If the user's task involves TRAVEL (e.g., destinations, flights, hotels, itineraries, travel advice, local attractions, visas, or transportation), you MUST use the RAG_tool to retrieve up-to-date and contextually relevant information before answering.
                            - For all other topics, solve the task using your internal reasoning and knowledge base.

                            When using the RAG_tool:
                            1. Formulate an appropriate retrieval query based on the user's request.
                            2. Integrate the retrieved information naturally into your response.
                            3. Always mention that the information was retrieved from the travel knowledge base.

                            When NOT using the RAG_tool:
                            - Use your own reasoning, models, or logic to solve the task directly.

                            Finally:
                            - Provide a clear, structured answer.
                            - Include reasoning steps in a natural way (no “chain-of-thought” style).
                            - Avoid hallucinating facts; if unsure, state your uncertainty.""",
        output_content_type=AgentRagResponse,
        )
async def Agentic_RAG(task:str) :

    result = await agent.run(task=task)
    # print(result.messages)
    # for message in result.messages:

    #     print(message.content)
    return result.messages[-1].content.retrieve_document,result.messages[-1].content.thoughts,result.messages[-1].content.answer
# retrive_document,thought=asyncio.run(Agentic_RAG(" What does Beijing in 2026 gives travelers to explore this most culture-rich of capitals."))
# print("result: ",retrive_document)
# print("thought :",thought)