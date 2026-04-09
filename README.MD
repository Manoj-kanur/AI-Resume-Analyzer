# AI Resume Analyzer

An AI-powered resume analysis platform that compares a candidate's resume against a job description and generates a match score, missing keywords, strengths, ATS tips, and improvement suggestions using the Gemini API.

---

## Features

* Upload resume files in PDF format
* Paste any job description
* AI-powered analysis using Gemini
* Resume-to-job match score
* Missing skills and keyword detection
* Strength identification
* Improvement suggestions
* ATS optimization tips
* FastAPI backend with REST endpoints
* Clean HTML/CSS/JavaScript frontend

---

## Tech Stack

### Frontend

* HTML5
* CSS3
* JavaScript

### Backend

* Python 3.10+
* FastAPI
* Uvicorn
* Google Gemini API
* python-dotenv
* PyPDF2 / pdfplumber

---

## Project Structure

```text
AI-resume-analyser/
│
├── backend/
│   ├── main.py
│   ├── requirements.txt
│   ├── .env
│   
│
├── frontend/
│   ├── index.html
│   ├── style.css
│ 
└── README.md
```

---

## Installation

### 1. Clone the Repository

```bash
git clone <repository-url>
cd AI-resume-analyser
```

### 2. Create a Virtual Environment

#### Windows

```powershell
python -m venv venv
venv\Scripts\activate
```



### 3. Install Backend Dependencies

```bash
cd backend
pip install -r requirements.txt
```

---

## Environment Variables

Create a `.env` file inside the `backend` folder.

```env
GEMINI_API_KEY=your_api_key_here
```

Example:

```env
GEMINI_API_KEY=AIzaSyxxxxxxxxxxxxxxxxxxxx
```

---

## Gemini Configuration

In your backend code, configure Gemini like this:

```python
import google.generativeai as genai

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

model = genai.GenerativeModel("gemini-2.0-flash")
```


---

## Running the Backend

```bash
cd backend
uvicorn main:app --reload
```



---

## Running the Frontend

Open the `frontend/index.html` file in your browser.

Or use a local server:

```bash
cd frontend
python -m http.server 5500
```

Then open:

```text
http://localhost:5500
```

---

## API Endpoint
## Sample Response

```json
{
  "match_score": 84,
  "verdict": "Strong Match",
  "summary": "Your resume aligns well with the required skills and experience.",
  "strengths": [
    "Strong Python knowledge",
    "Experience with FastAPI",
    "Relevant AI projects"
  ],
  "missing_keywords": [
    "Docker",
    "AWS",
    "CI/CD"
  ],
  "improvements": [
    {
      "section": "Projects",
      "suggestion": "Add measurable outcomes and technologies used."
    }
  ],
  "ats_tips": [
    "Use standard section headings",
    "Include keywords from the job description"
  ]
}
```
---


## Future Enhancements

* Support DOCX and TXT resumes
* Add user authentication
* Save analysis history in MongoDB
* Generate downloadable PDF reports
* Add dashboard and charts
* Compare multiple resumes
* Add LinkedIn profile analysis


---

## Author

Developed **K R Manoj** as an AI-based resume analysis project using FastAPI and Gemini API.
