from utils import get_openai_api_key
from autogen import ConversableAgent, initiate_chats, AssistantAgent, GroupChat, GroupChatManager
from autogen.coding import LocalCommandLineCodeExecutor
import pprint
import datetime
import yfinance
import matplotlib.pyplot as plt
import os
from IPython.display import Image

# get the openai api key
OPENAI_API_KEY = get_openai_api_key()
llm_config = {"model": "gpt-4.1-nano"}

task = "Write a blogpost about the stock price performance of "\
"Nvidia in the past month. Today's date is 2025-05-26."

# Create agents for the group chat
user_proxy = ConversableAgent(
    name="Admin",
    system_message="Give the task, and send "
    "instructions to writer to refine the blog post.",
    code_execution_config=False,
    llm_config=llm_config,
    human_input_mode="ALWAYS",
)
planner = ConversableAgent(
    name="Planner",
    system_message="Given a task, please determine "
    "what information is needed to complete the task. "
    "Please note that the information will all be retrieved using"
    " Python code. Please only suggest information that can be "
    "retrieved using Python code. "
    "After each step is done by others, check the progress and "
    "instruct the remaining steps. If a step fails, try to "
    "workaround",
    description="Planner. Given a task, determine what "
    "information is needed to complete the task. "
    "After each step is done by others, check the progress and "
    "instruct the remaining steps",
    llm_config=llm_config,
)
engineer = AssistantAgent(
    name="Engineer",
    llm_config=llm_config,
    description="An engineer that writes code based on the plan "
    "provided by the planner.",
)
executor = ConversableAgent(
    name="Executor",
    system_message="Execute the code written by the "
    "engineer and report the result.",
    human_input_mode="NEVER",
    code_execution_config={
        "last_n_messages": 3,
        "work_dir": "coding",
        "use_docker": False,
    },
)
writer = ConversableAgent(
    name="Writer",
    llm_config=llm_config,
    system_message="Writer."
    "Please write blogs in markdown format (with relevant titles)"
    " and put the content in pseudo ```md``` code block. "
    "You take feedback from the admin and refine your blog.",
    description="Writer."
    "Write blogs based on the code execution results and take "
    "feedback from the admin to refine the blog."
)

# Define the group chat and manager
groupchat = GroupChat(
    agents=[user_proxy, engineer, writer, executor, planner],
    messages=[],
    max_round=10,
)

# Create the manager
manager = GroupChatManager(
    groupchat=groupchat, llm_config=llm_config
)

# Initiate the group chat
groupchat_result = user_proxy.initiate_chat(
    manager,
    message=task,
)

# create group chat with user policy on transition order
groupchat = GroupChat(
    agents=[user_proxy, engineer, writer, executor, planner],
    messages=[],
    max_round=10,
    allowed_or_disallowed_speaker_transitions={
        user_proxy: [engineer, writer, executor, planner],
        engineer: [user_proxy, executor],
        writer: [user_proxy, planner],
        executor: [user_proxy, engineer, planner],
        planner: [user_proxy, engineer, writer],
    },
    speaker_transitions_type="allowed",
)

# Create the manager and initiate the group chat with user policy on transition order
manager = GroupChatManager(
    groupchat=groupchat, llm_config=llm_config
)
groupchat_result = user_proxy.initiate_chat(
    manager,
    message=task,
)
