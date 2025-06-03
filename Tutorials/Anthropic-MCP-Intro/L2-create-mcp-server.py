# start a new terminal
import os
from IPython.display import IFrame

IFrame(f"{os.environ.get('DLAI_LOCAL_URL').format(port=8888)}terminals/1", 
       width=600, height=768)

# Or for local server go to this path in terminal
# "C:\Users\furqa\Desktop\JobHunt2025\learning-ai-agents\Tutorials\Anthropic-MCP-Intro\mcp_project"
# and go here "C:\Users\furqa\Desktop\JobHunt2025\learning-ai-agents\Tutorials\Anthropic-MCP-Intro\mcp_project\.venv\Scripts\activate"
# then run "npx @modelcontextprotocol/inspector uv run research_server.py"
