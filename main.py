import os
import re
import json
import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from bs4 import BeautifulSoup
from langchain_groq import ChatGroq
from langchain_community.tools.tavily_search import TavilySearchResults

# 1. إعداد المفاتيح (تأكد من صحتها)

# الكود كدة هيدور على المفاتيح في إعدادات السيرفر مش جوه الكود
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

app = FastAPI(title="AI Fact-Checker API")

# تعريف الموديلات

llm = ChatGroq(
    temperature=0.1, 
    model_name="llama-3.3-70b-versatile", # التعديل هنا
    groq_api_key=GROQ_API_KEY
)

search_tool = TavilySearchResults(k=2, tavily_api_key=TAVILY_API_KEY)

class ClaimRequest(BaseModel):
    text: str

def scrape_url(url: str):
    try:
        response = requests.get(url, timeout=10)
        soup = BeautifulSoup(response.text, 'html.parser')
        paragraphs = soup.find_all(['h1', 'p'])
        content = " ".join([p.get_text() for p in paragraphs])
        return content[:2000]
    except Exception as e:
        return f"Error scraping URL: {str(e)}"

@app.post("/verify")
async def verify_claim(request: ClaimRequest):
    try:
        user_input = request.text.strip()
        
        if user_input.startswith("http"):
            content_to_verify = scrape_url(user_input)
        else:
            content_to_verify = user_input

        # التعديل هنا: تمرير النص مباشرة وتحويل النتيجة لنص
        print("Searching for evidence...")
        search_data = search_tool.invoke(content_to_verify)
        search_results_text = str(search_data) # تأمين البيانات كنص
        
        prompt = f"""
        You are a professional fact-checker. 
        Analyze the following claim based ONLY on the provided evidence.

        CRITICAL INSTRUCTIONS:
        1. Language: If the claim is in Arabic, write the 'explanation' in Arabic.
        2. Verdict: Must be "True", "False", or "Misleading".
        3. Format: Return ONLY a valid JSON.

        Claim: {content_to_verify}
        Evidence: {search_results_text}

        Response Format (JSON):
        {{
            "verdict": "Verdict here",
            "confidence_score": 0-100,
            "explanation": "Your explanation here",
            "top_sources": ["url1", "url2"]
        }}
        """

        response = llm.invoke(prompt)
        raw_content = response.content.strip()

        # تنظيف الـ JSON
        clean_json = re.sub(r'```json\s*|```', '', raw_content).strip()
        return json.loads(clean_json)

    except Exception as e:
        # هذا الجزء سيمنع الـ 500 ويريك سبب الخطأ الحقيقي
        raise HTTPException(status_code=500, detail=f"Internal Error: {str(e)}")

# uvicorn main:app --reload