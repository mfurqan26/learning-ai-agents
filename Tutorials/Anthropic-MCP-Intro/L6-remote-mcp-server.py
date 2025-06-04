# for remote server, Initialize FastMCP server like this
# mcp = FastMCP("research", port=8001)

# And specify main mcp run like this:
# if __name__ == "__main__":
"mcp.run(transport='sse')" 

# run the server like this:
"cd mcp_project"
"uv run research_server.py"
"npx @modelcontextprotocol/inspector"
"uv pip compile pyproject.toml > requirements.txt"
'echo "python-3.12.3" > runtime.txt'