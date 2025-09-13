import os
from dotenv import load_dotenv
import boto3
import json
import requests

# Load AWS creds
load_dotenv()
aws_access_key = os.getenv("AWS_ACCESS_KEY_ID")
aws_secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")
aws_region = "us-west-2"

# Bedrock clients
bedrock_agent_runtime = boto3.client(
    "bedrock-agent-runtime",
    region_name=aws_region,
    aws_access_key_id=aws_access_key,
    aws_secret_access_key=aws_secret_key,
)
bedrock_runtime = boto3.client(
    "bedrock-runtime",
    region_name=aws_region,
    aws_access_key_id=aws_access_key,
    aws_secret_access_key=aws_secret_key,
)

KNOWLEDGE_BASE_ID = "VDOXW9LBVP"
MODEL_ID = "anthropic.claude-3-7-sonnet-20250219-v1:0"

# ---- Fetch KB context ----
def fetch_kb_context(query: str, num_results=3):
    kb_response = bedrock_agent_runtime.retrieve(
        knowledgeBaseId=KNOWLEDGE_BASE_ID,
        retrievalQuery={"text": query},
        retrievalConfiguration={
            "vectorSearchConfiguration": {"numberOfResults": num_results}
        },
    )
    return [doc["content"]["text"] for doc in kb_response.get("retrievalResults", [])]

# ---- Fetch and extract full article text using Serper Extract API ----
def fetch_article_text(url: str, api_key: str) -> str:
    serper_extract_url = "https://google.serper.dev/extract"
    headers = {"X-API-KEY": api_key, "Content-Type": "application/json"}
    payload = {"url": url}
    resp = requests.post(serper_extract_url, headers=headers, json=payload)
    data = resp.json()
    return data.get("text", "")

# ---- Fetch external Reuters articles (via Serper.dev as example) ----
def fetch_reuters_articles(query: str, api_key: str, num_results=2):
    """Use Serper.dev (Google Search API) to get Reuters links + snippets."""
    url = "https://google.serper.dev/news"
    headers = {"X-API-KEY": api_key, "Content-Type": "application/json"}
    payload = {"q": f"site:reuters.com {query}", "num": num_results}
    resp = requests.post(url, headers=headers, json=payload)
    data = resp.json()
    articles = []
    for item in data.get("news", []):
        articles.append(f"{item.get('title')} - {item.get('snippet')} ({item.get('link')})")
    return articles

# ---- RAG evaluation ----
def evaluate_bias(article_url: str, ai_summary: str, system_prompt: str, google_api_key: str):
    original_article = fetch_article_text(article_url, google_api_key)
    # Step 1: fetch KB context
    kb_context = fetch_kb_context("company policy for news reporting and approved corpus")

    # Step 2: fetch Reuters references
    reuters_articles = fetch_reuters_articles(original_article[:150], google_api_key)

    # Step 3: Build prompt
    prompt = f"""{system_prompt}

Original Article:
{original_article}

AI-Generated Summary:
{ai_summary}

Company Policy & Approved Corpus (from KB):
{chr(10).join(kb_context)}

External Reuters References:
{chr(10).join(reuters_articles)}

Now evaluate the AI-generated summary strictly following the instructions."""
    
    # Step 4: Call Bedrock model
    model_response = bedrock_runtime.invoke_model(
        modelId=MODEL_ID,
        body=json.dumps({
            "inputText": prompt,
            "textGenerationConfig": {
                "maxTokenCount": 1000,
                "temperature": 0.3,
                "topP": 0.9
            }
        }),
    )
    response_body = json.loads(model_response["body"].read())
    return response_body.get("outputText", "No response.")

if __name__ == "__main__":
    system_prompt = """You are an AI assistant specialized in evaluating AI-generated news summaries for bias, factual accuracy, and alignment with company policy. ..."""
    
    # Example user input
    ARTICLE_URL = "https://www.aftonbladet.se/nyheter/a/Oo4ngk/pa-migrantkyrkogarden-pa-lampedusa-vilar-barnen-som-dog-till-havs"
    ai_summary = """    Lampedusa has become an important arrival point for migrants to Europe, and the local population is showing solidarity by helping them, despite the difficult circumstances of migration across the Mediterranean.
    The Red Cross is cleaning the beaches to hide the traces of migrants' journeys that were previously common.
    The exhibition "Migrated Objects" in Lampedusa aims to increase understanding of the complexity of migration by displaying everyday objects left behind by migrants. """
    
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")  # store in .env

    print("AWS_ACCESS_KEY_ID:", os.getenv("AWS_ACCESS_KEY_ID"))
    print("AWS_SECRET_ACCESS_KEY:", os.getenv("AWS_SECRET_ACCESS_KEY")[:4] + "****")

    
    result = evaluate_bias(ARTICLE_URL, ai_summary, system_prompt, GOOGLE_API_KEY)
    print("Evaluation Result:\n", result)
