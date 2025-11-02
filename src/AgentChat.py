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

RAG_tool = FunctionTool(Retrieval_Tool, description="Retrieves relevant documents from a qdrant",strict=True)
agent = AssistantAgent(
        name="assistant",
        model_client=model_client,
        model_client_stream=True,
        tools=[RAG_tool],
        system_message="Use RAG_tool to solve the tasks about travelling around the world",
        output_content_type=AgentRagResponse
    )
async def main() -> None:

    stream = agent.run_stream(task="what beautiful place do you offer in Québec?",cancellation_token=cancellation_token)
    await Console(stream)


asyncio.run(main())