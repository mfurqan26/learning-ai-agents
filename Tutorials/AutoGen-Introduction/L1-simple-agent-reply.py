from utils import get_openai_api_key
from autogen import ConversableAgent

# get the openai api key
OPENAI_API_KEY = get_openai_api_key()
llm_config = {"model": "gpt-4.1-nano"}

# create the agent
agent = ConversableAgent(
    name="chatbot",
    llm_config=llm_config,
    human_input_mode="NEVER",
)

# generate a reply
reply = agent.generate_reply(
    messages=[{"content": "Tell me a joke.", "role": "user"}]
)
print(reply)