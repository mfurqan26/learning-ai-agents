from dotenv import load_dotenv
import anthropic

load_dotenv()
anthropic = anthropic.Anthropic()

# See /mcp_project/mcp_chatbot.py for the created mcp client code 
# Note how the tools are defined in the server and then passed to the client and here we just have chatbot logic
