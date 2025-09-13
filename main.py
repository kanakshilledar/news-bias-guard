import os
import boto3
import requests
import json
from strands.tools.mcp.mcp_client import MCPClient
from mcp.client.stdio import stdio_client
from mcp.client.stdio import StdioServerParameters
from strands.models import BedrockModel
from strands import Agent
from strands_tools import retrieve

# get account and region
account_id = boto3.client('sts').get_caller_identity()['Account']
region = boto3.Session().region_name

model = BedrockModel(
    model_id="us.anthropic.claude-3-7-sonnet-20250219-v1:0",  # choose your model id. Here - Claude 3.7 Sonnet
    temperature=0.3,  # Lower temperature = more consistent, reliable responses for customer service
    region_name=region  # Primary AWS region with latest model availability
)


def download_files(): 
    bucket_name = f"{account_id}-{region}-kb-data-bucket"

    os.makedirs("knowledge_base_data", exist_ok=True)

    # download all files

    s3 = boto3.client('s3')
    objects = s3.list_objects_v2(Bucket=bucket_name)

    for obj in objects['Contents']:
        file_name = obj['Key']
        s3.download_file(bucket_name, file_name, f"knowledge_base_data/{file_name}")
        print(f"[+] Downloaded: {file_name}")

    print(f"[*] All files saved to: knowledge_base_data/")


def fetch_article_text(article_url: str) -> str:
    serper_extract_url = "https://scrape.serper.dev"
    headers = {
        "X-API-KEY": "YOURAPIKEYHERE",
        "Content-Type": "application/json"
    }
    payload = {"url": article_url}

    try:
        response = requests.post(serper_extract_url, headers=headers, json=payload)
        response.raise_for_status()
        data = response.json()

        # Return only the text field
        return data.get("text", "")

    except requests.exceptions.RequestException as e:
        print(f"[!] Error fetching article: {e}")
        return f"[!] Error fetching article: {e}"


def fetch_similar_articles(summary: str) -> str:
    url = "https://google.serper.dev/news"

    payload = json.dumps({
    "q": summary,
    "gl": "se",
    "hl": "sv"
    })
    headers = {
    'X-API-KEY': 'YOURAPIKEYHERE',
    'Content-Type': 'application/json'
    }

    response = requests.request("POST", url, headers=headers, data=payload)

    print(response.text)

# download_files()

# System prompt: This defines your agent's personality and capabilities
# Think of this as the "job description" for your AI agent

KB_CONTEXT = """NewsCorp Content & Editorial Policy (Terms of Service, Brand, and Conflict
of Interest)
1. Purpose and Scope
This policy governs the creation, review, and publication of all content produced or distributed
under the NewsCorp brand. It applies to journalists, editors, freelancers, and contractors who
contribute to NewsCorp platforms. The policy ensures that content aligns with our brand values,
legal obligations, and ethical standards.
2. Terms of Service Compliance
● All content must comply with applicable laws and regulations, including copyright,
privacy, and data protection rules.
● Users submitting content or interacting with NewsCorp platforms must agree to our
Terms of Service, which define rights, responsibilities, and permissible use of our
services.
● Any external content or third-party material incorporated into NewsCorp reporting must
have proper attribution, permissions, or licensing.
3. Brand Language and Tone
● NewsCorp communications must be neutral, professional, and accessible, reflecting
our commitment to factual reporting.
● Avoid sensationalism, inflammatory language, or personal attacks.
● Maintain clarity, conciseness, and grammatical accuracy in all content, whether digital,
print, or multimedia.
● Use inclusive language that respects diversity, avoids stereotypes, and reflects the
global community we serve.
● Headlines and summaries must accurately represent the article’s content; avoid
misleading phrasing to attract clicks.
4. Conflict of Interest
● Staff must disclose any potential conflicts of interest, including personal, financial, or
organizational relationships that could compromise editorial integrity.
● Editors and reporters should avoid assignments or coverage where conflicts of interest
exist. If unavoidable, a disclosure must be published alongside the content.
● Acceptance of gifts, payments, or incentives from sources, sponsors, or stakeholders
related to reported content is strictly prohibited.
5. Editorial Review and Accountability
● All articles, summaries, and multimedia content must undergo fact-checking and editorial
review before publication.
● Any content identified as violating this policy will be corrected or removed promptly.
● Editors are responsible for documenting corrections and notifying stakeholders when
errors or misrepresentations occur.
6. AI-Generated Content
● AI-generated content must be clearly labeled and reviewed for factual accuracy,
neutrality, and compliance with this policy.
● Automated tools must not replace human editorial judgment; the final responsibility rests
with the editorial team.
7. Transparency and Public Trust
● Maintain transparency with readers regarding sources, corrections, and potential
editorial decisions.
● Uphold NewsCorp’s reputation for integrity, accuracy, and reliability in all reporting.
8. Enforcement
● Violations of this policy may result in disciplinary action, including suspension or
termination, depending on the severity of the breach.
● Staff are encouraged to report any suspected violations to their editor-in-chief or
compliance officer.
9. Updates
● This policy will be reviewed annually and updated to reflect evolving legal standards,
technological developments, and best practices in journalism.
Effective Date: September 2025"""

SYSTEM_PROMPT = """You are an AI assistant specialized in evaluating AI-generated news summaries for bias, factual accuracy, and alignment with company policy. Your task is to help human news editors by assessing how well a summary reflects the original article, adheres to company-approved sources, and avoids internal AI bias or editorial slant.

Inputs:
-   `Original Article`: The full text of a news article.
-   `AI-Generated Summary`: The summary produced by an AI system.
-   `Company Policy`: Guidelines that define acceptable language, tone, and ethical standards for reporting.
-   `List of Approved Articles`: Reuters.

Instructions:
1.  Compare the AI-generated summary to the original article and the approved corpus.
2.  Identify any factual inconsistencies, omissions, exaggerations, or distortions.
3.  Detect any language or framing that could indicate internal AI bias or misalignment with company policy.
4.  Evaluate how much the AI altered the words or meaning compared to the original article.
5.  Provide a Bias/Accuracy Score between 0 and 100, where 100 means perfectly accurate, fully compliant with policy, and unbiased.
6.  Provide a short explanation supporting your score, highlighting the main issues or areas where the summary deviates from the original article or policy.
7.  If possible, suggest minimal edits to improve alignment with the original article and company policy"""

AI_SUMMARY = """    Lampedusa has become an important arrival point for migrants to Europe, and the local population is showing solidarity by helping them, despite the difficult circumstances of migration across the Mediterranean.
    The Red Cross is cleaning the beaches to hide the traces of migrants' journeys that were previously common.
    The exhibition "Migrated Objects" in Lampedusa aims to increase understanding of the complexity of migration by displaying everyday objects left behind by migrants. """

URL = "https://www.aftonbladet.se/nyheter/a/Oo4ngk/pa-migrantkyrkogarden-pa-lampedusa-vilar-barnen-som-dog-till-havs"

# Agent assembly: Combining model + tools + instructions = functional AI agent
agent = Agent(
    model=model,                    # The "brain" - our configured Claude 3.7 model
    system_prompt= SYSTEM_PROMPT    # behavioral guidelines
)

original_news = fetch_article_text(URL)
similar_articles = fetch_similar_articles(AI_SUMMARY)

query = f"""{SYSTEM_PROMPT}
Original Article: {original_news}
AI Summary: {AI_SUMMARY}
Knowledge Base Context: {KB_CONTEXT}
Similar Articles: {similar_articles}

Now evaluate the AI-generated summary strictly following the instructions."""

# # Interactive testing: See how your agent responds to real customer queries
# for query in test_queries:
#     print(f"\n Customer: {query}")
#     response = agent(query)
#     print(f" Agent: {response}")

response = agent(query)
print(response)
    


