import os
import requests
import urllib.parse
from requests.exceptions import RequestException

def get_google_results(query, num_results=2):
    google_api_key = os.getenv("GOOGLE_API_KEY")
    search_engine_id = os.getenv("SEARCH_ENGINE_ID")
    
    encoded_query = urllib.parse.quote(query)
    url = f"https://www.googleapis.com/customsearch/v1?key={google_api_key}&cx={search_engine_id}&q={encoded_query}&num={num_results}"
    
    try:
        response = requests.get(url)
        response.raise_for_status()
        results = response.json()
        
        search_results = []
        if "items" in results:
            for item in results["items"]:
                search_results.append(f"Title: {item['title']}\nLink: {item['link']}\nSnippet: {item['snippet']}\n")
            return "\n".join(search_results)
        else:
            return "No items found."
    except RequestException:
        return "Network error. Please check your connection."
    except ValueError:
        return "Invalid query format."
    except Exception:
        return "An error occurred. Please contact support."
