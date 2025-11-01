import dotenv
import os
from autogen import AssistantAgent,ConversableAgent

dotenv.load_dotenv()
API_KEY=os.getenv("API_KEY")
BASE_URL=os.getenv("BASE_URL")
MODEL_NAME=os.getenv("MODEL_NAME")

print(API_KEY, BASE_URL, MODEL_NAME)



llm_config = {
    "config_list":[{"model":MODEL_NAME,"api_key":API_KEY, "base_url": BASE_URL}],
}

assistant=AssistantAgent(
    name="assistant",
    llm_config=llm_config
)

agent = ConversableAgent(
    "chatbot",
    llm_config=llm_config,
    code_execution_config=False,  # Turn off code execution, by default it is off.
    function_map=None,  # No registered functions, by default it is None.
    human_input_mode="NEVER",  # Never ask for human input.
)

reply=agent.generate_reply(messages=[{"content":"Tell me a joke","role":"user"}])
print("reply:",reply)