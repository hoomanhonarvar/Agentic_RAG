import asyncio
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_agentchat.agents import AssistantAgent
import dotenv
import os
from typing_extensions import Any, AsyncGenerator, Required, TypedDict, Union, deprecated
from autogen_core.models import ModelInfo, ModelFamily
dotenv.load_dotenv()
API_KEY=os.getenv("API_KEY")
BASE_URL=os.getenv("BASE_URL")
MODEL_NAME=os.getenv("MODEL_NAME")


model_info=ModelInfo(
    vision=False,
    function_calling=True,
    json_output=True,
    family=ModelFamily.ANY,
    structured_output=True
    )
async def main() -> None:
    model_client = OpenAIChatCompletionClient(
        model=MODEL_NAME,
        api_key=API_KEY,
        base_url=BASE_URL,
        model_info=model_info
    )
    agent = AssistantAgent(
        name="assistant",
        model_client=model_client,
    )

    result = await agent.run(task="Name two cities in North America.")
    print(result)


asyncio.run(main())