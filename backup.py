import os
from dotenv import load_dotenv
import boto3
import json
import requests

# Load AWS creds
load_dotenv()
# aws_access_key = os.getenv("AWS_ACCESS_KEY_ID")
# aws_secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")
# aws_region = "us-west-2"

# Bedrock client
bedrock_runtime = boto3.client(
    "bedrock-runtime"
)

MODEL_ID = "amazon.nova-premier-v1:0"

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

# ---- Bias evaluation ----
def evaluate_bias(article_url: str, ai_summary: str, system_prompt: str, company_policy: str, google_api_key: str):
    original_article = fetch_article_text(article_url, google_api_key)

    # Step 1: fetch Reuters references
    reuters_articles = fetch_reuters_articles(original_article[:150], google_api_key)

    # Step 2: Build prompt
    prompt = f"""{system_prompt}

Original Article:
{original_article}

AI-Generated Summary:
{ai_summary}

Company Policy & Approved Corpus (embedded):
{company_policy}

External Reuters References:
{chr(10).join(reuters_articles)}

Now evaluate the AI-generated summary strictly following the instructions."""
    
    # Step 3: Call Bedrock model
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
    system_prompt = """You are an AI assistant specialized in evaluating AI-generated news summaries for bias, factual accuracy, and alignment with company policy. Your task is to help human news editors by assessing how well a summary reflects the original article, adheres to company-approved sources, and avoids internal AI bias or editorial slant.

Instructions:

1. Compare the AI-generated summary to the original article and the approved corpus.
2. Identify any factual inconsistencies, omissions, exaggerations, or distortions.
3. Detect any language or framing that could indicate internal AI bias or misalignment with company policy.
4. Evaluate how much the AI altered the words or meaning compared to the original article.
5. Provide a Bias/Accuracy Score between 0 and 100, where 100 means perfectly accurate, fully compliant with policy, and unbiased.
6. Provide a short explanation supporting your score, highlighting the main issues or areas where the summary deviates from the original article or policy.
7. If possible, suggest minimal edits to improve alignment with the original article and company policy.
"""

    # Example company policy text (replace with your actual corpus)
    company_policy = """Company-approved reporting must:
- Remain neutral in tone
- Avoid emotionally charged or speculative language
- Reference only verified, reputable sources
- Ensure factual accuracy without omission of key details"""

    ARTICLE_URL = "https://www.aftonbladet.se/nyheter/a/Oo4ngk/pa-migrantkyrkogarden-pa-lampedusa-vilar-barnen-som-dog-till-havs"
    ai_summary = """Lampedusa has become an important arrival point for migrants to Europe, and the local population is showing solidarity by helping them, despite the difficult circumstances of migration across the Mediterranean.
    The Red Cross is cleaning the beaches to hide the traces of migrants' journeys that were previously common.
    The exhibition "Migrated Objects" in Lampedusa aims to increase understanding of the complexity of migration by displaying everyday objects left behind by migrants."""

    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")  # store in .env
    
    result = evaluate_bias(ARTICLE_URL, ai_summary, system_prompt, company_policy, GOOGLE_API_KEY)
    print("Evaluation Result:\n", result)
