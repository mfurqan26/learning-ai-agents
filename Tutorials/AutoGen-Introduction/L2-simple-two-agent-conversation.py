from utils import get_openai_api_key
from autogen import ConversableAgent
import pprint

# get the openai api key
OPENAI_API_KEY = get_openai_api_key()
llm_config = {"model": "gpt-4.1-nano"}

# create the agents
cathy = ConversableAgent(
    name="cathy",
    system_message=
    "Your name is Cathy and you are a stand-up comedian.",
    llm_config=llm_config,
    human_input_mode="NEVER",
)

joe = ConversableAgent(
    name="joe",
    system_message=
    "Your name is Joe and you are a stand-up comedian. "
    "Start the next joke from the punchline of the previous joke.",
    llm_config=llm_config,
    human_input_mode="NEVER",
)

# initiate the conversation with better summary
chat_result = joe.initiate_chat(
    cathy, 
    message="I'm Joe. Cathy, let's keep the jokes rolling.", 
    max_turns=2, 
    summary_method="reflection_with_llm",
    summary_prompt="Summarize the conversation",
)

# print the summary
pprint.pprint(chat_result.summary)

# create the agents with termination message instead of 2 max_turns
alice = ConversableAgent(
    name="alice",
    system_message=
    "Your name is Alice and you are a stand-up comedian. "
    "When you're ready to end the conversation, say 'I gotta go'.",
    llm_config=llm_config,
    human_input_mode="NEVER",
    is_termination_msg=lambda msg: "I gotta go" in msg["content"],
)

bob = ConversableAgent(
    name="bob",
    system_message=
    "Your name is Bob and you are a stand-up comedian. "
    "When you're ready to end the conversation, say 'I gotta go' or 'Goodbye'.",
    llm_config=llm_config,
    human_input_mode="NEVER",
    is_termination_msg=lambda msg: "I gotta go" in msg["content"] or "Goodbye" in msg["content"],
)

# initiate the conversation and it will terminate itself when agents say gotta go or goodbye
chat_result = alice.initiate_chat(
    bob,
    message="I'm Alice. Bob, let's keep the jokes rolling."
)

# alice sends a message to bob to get the last thing by agent
alice.send(message="What's last joke we talked about?", recipient=bob)
