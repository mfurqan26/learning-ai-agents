from dotenv import load_dotenv
import os
from tavily import TavilyClient
import requests
from bs4 import BeautifulSoup
from duckduckgo_search import DDGS
import re
import json
from pygments import highlight, lexers, formatters

# load environment variables from .env file
_ = load_dotenv()
client = TavilyClient(api_key=os.environ.get("TAVILY_API_KEY"))

result = client.search("What is in Nvidia's new Blackwell GPU?",
                       include_answer=True)
result["answer"]

# Regular Search
city = "Dammam, Saudi Arabia"
query = f"""
    what is the current weather in {city}?
    Should I travel there today?
    "weather.com"
"""

ddg = DDGS()

def search(query, max_results=6):
    try:
        results = ddg.text(query, max_results=max_results)
        return [i["href"] for i in results]
    except Exception as e:
        print(f"returning previous results due to exception reaching ddg.")
        results = [ # cover case where DDG rate limits due to high deeplearning.ai volume
            "https://weather.com/weather/today/l/USCA0987:1:US",
            "https://weather.com/weather/hourbyhour/l/54f9d8baac32496f6b5497b4bf7a277c3e2e6cc5625de69680e6169e7e38e9a8",
        ]
        return results  

def scrape_weather_info(url):
    """Scrape content from the given URL"""
    if not url:
        return "Weather information could not be found."
    
    # fetch data
    headers = {'User-Agent': 'Mozilla/5.0'}
    response = requests.get(url, headers=headers)
    if response.status_code != 200:
        return "Failed to retrieve the webpage."

    # parse result
    soup = BeautifulSoup(response.text, 'html.parser')
    return soup

for i in search(query):
    print(i)

# use DuckDuckGo to find websites and take the first result
url = search(query)[0]
# scrape first wesbsite
soup = scrape_weather_info(url)
print(f"Website: {url}\n\n")
print(str(soup.body)[:50000]) # limit long outputs

# extract text
weather_data = []
for tag in soup.find_all(['h1', 'h2', 'h3', 'p']):
    text = tag.get_text(" ", strip=True)
    weather_data.append(text)

# combine all elements into a single string and remove all spaces from the combined text
weather_data = "\n".join(weather_data)
weather_data = re.sub(r'\s+', ' ', weather_data)
print(f"Website: {url}\n\n")
print(weather_data)

# Agentic Search with Tavily
result = client.search(query, max_results=1)
data = result["results"][0]["content"]
print(data)

parsed_json = json.loads(data.replace("'", '"'))
formatted_json = json.dumps(parsed_json, indent=4)
colorful_json = highlight(formatted_json,
                          lexers.JsonLexer(),
                          formatters.TerminalFormatter())

print(colorful_json)