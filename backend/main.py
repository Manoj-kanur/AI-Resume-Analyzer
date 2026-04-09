from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import google.generativeai as genai
import pdfplumber
import io
import json
import re
import os

app = FastAPI(title="AI Resume Analyzer")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

genai.configure(api_key=os.environ["GEMINI_API_KEY"])
model = genai.GenerativeModel("gemini-2.5-flash")


def extract_text_from_pdf(pdf_bytes: bytes) -> str:
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        text = ""
        for page in pdf.pages:
            text += (page.extract_text() or "") + "\n"
    return text.strip()


def analyze_with_gemini(resume_text: str, job_description: str) -> dict:
    prompt = f"""You are an expert technical recruiter and career coach.

Analyze the following resume against the job description and return ONLY a valid JSON object (no markdown, no extra text).

RESUME:
{resume_text}

JOB DESCRIPTION:
{job_description}

Return this exact JSON structure:
{{
  "match_score": <integer 0-100>,
  "summary": "<2-3 sentence overall assessment>",
  "strengths": ["<strength 1>", "<strength 2>", "<strength 3>"],
  "missing_keywords": ["<keyword1>", "<keyword2>", "<keyword3>", "<keyword4>", "<keyword5>"],
  "improvements": [
    {{"section": "<Resume section>", "suggestion": "<specific actionable improvement>"}},
    {{"section": "<Resume section>", "suggestion": "<specific actionable improvement>"}},
    {{"section": "<Resume section>", "suggestion": "<specific actionable improvement>"}}
  ],
  "ats_tips": ["<ATS tip 1>", "<ATS tip 2>", "<ATS tip 3>"],
  "verdict": "<one of: Strong Match | Good Match | Partial Match | Weak Match>"
}}"""

    response = model.generate_content(prompt)
    raw = response.text.strip()
    raw = re.sub(r"^```json\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)
    return json.loads(raw)


@app.post("/analyze")
async def analyze_resume(
    resume: UploadFile = File(...),
    job_description: str = Form(...)
):
    if not resume.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    if len(job_description.strip()) < 50:
        raise HTTPException(status_code=400, detail="Please provide a more detailed job description.")

    pdf_bytes = await resume.read()
    resume_text = extract_text_from_pdf(pdf_bytes)

    if len(resume_text) < 100:
        raise HTTPException(status_code=400, detail="Could not extract text from PDF. Make sure it's not a scanned image.")

    result = analyze_with_gemini(resume_text, job_description)
    return JSONResponse(content=result)


@app.get("/health")
def health():
    return {"status": "ok"}