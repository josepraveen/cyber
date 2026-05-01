import tempfile
import os
import asyncio

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from pydantic import BaseModel, Field
from typing import List
from dotenv import load_dotenv

from openai import AsyncOpenAI
from agents import Agent, Runner, trace

from context import (
    SECURITY_RESEARCHER_INSTRUCTIONS,
    get_analysis_prompt,
    enhance_summary,
)
from mcp_servers import create_semgrep_server

load_dotenv(override=True)



app = FastAPI(title="Cybersecurity Analyzer API")

# ----------------------------
# CORS
# ----------------------------
cors_origins = [
    "http://localhost:3000",
    "http://frontend:3000",
]

if os.getenv("ENVIRONMENT") == "production":
    cors_origins.append("*")

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------------------
# =% OPENROUTER CONFIG (IMPORTANT FIX)
# ----------------------------
os.environ["OPENAI_BASE_URL"] = "https://openrouter.ai/api/v1"
os.environ["OPENAI_API_KEY"] = os.getenv("OPENROUTER_API_KEY")

# Optional client (not passed to Runner anymore)
client = AsyncOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
)

# ----------------------------
# REQUEST / RESPONSE MODELS
# ----------------------------
class AnalyzeRequest(BaseModel):
    code: str


class SecurityIssue(BaseModel):
    title: str
    description: str
    code: str
    fix: str
    cvss_score: float
    severity: str


class SecurityReport(BaseModel):
    summary: str
    issues: List[SecurityIssue]


# ----------------------------
# HELPERS
# ----------------------------
def validate_request(request: AnalyzeRequest):
    if not request.code.strip():
        raise HTTPException(status_code=400, detail="No code provided")


def check_api_keys():
    if not os.getenv("OPENROUTER_API_KEY"):
        raise HTTPException(
            status_code=500,
            detail="OPENROUTER_API_KEY not configured"
        )


def create_security_agent(semgrep_server):
    return Agent(
        name="Security Researcher",
        instructions=SECURITY_RESEARCHER_INSTRUCTIONS,
        model=os.getenv("MODEL", "openai/gpt-4o-mini"),
        mcp_servers=[semgrep_server],
        output_type=SecurityReport,
    )


# ----------------------------
# CORE ANALYSIS PIPELINE
# ----------------------------
async def run_security_analysis(code: str) -> SecurityReport:
    with trace("Security Researcher"):
        async with create_semgrep_server() as semgrep:

            agent = create_security_agent(semgrep)

            try:
                with tempfile.NamedTemporaryFile(
                    mode="w",
                    suffix=".py",
                    delete=False
                ) as temp:
                    temp.write(code)
                    temp_path = temp.name

                try:
                    #  FIXED: NO openai_client HERE
                    result = await Runner.run(
                        agent,
                        input=get_analysis_prompt(code, temp_path)
                    )

                    report = result.final_output_as(SecurityReport)

                    # Sort by severity
                    report.issues.sort(
                        key=lambda x: x.cvss_score,
                        reverse=True
                    )

                    return report

                finally:
                    try:
                        os.unlink(temp_path)
                    except OSError:
                        pass

            except Exception as err:
                print(f"Error: {err}, type={type(err)}")
                raise


# ----------------------------
# RESPONSE ENHANCEMENT
# ----------------------------
def format_analysis_response(code: str, report: SecurityReport) -> SecurityReport:
    enhanced_summary = enhance_summary(len(code), report.summary)
    return SecurityReport(
        summary=enhanced_summary,
        issues=report.issues
    )


# ----------------------------
# API ROUTES
# ----------------------------
@app.post("/api/analyze", response_model=SecurityReport)
async def analyze_code(request: AnalyzeRequest):
    validate_request(request)
    check_api_keys()

    try:
        report = await run_security_analysis(request.code)
        return format_analysis_response(request.code, report)

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Analysis failed: {str(e)}"
        )


@app.get("/health")
async def health():
    return {"message": "Cybersecurity Analyzer API running"}


# ----------------------------
# STATIC FRONTEND
# ----------------------------
if os.path.exists("static"):
    app.mount("/", StaticFiles(directory="static", html=True), name="static")


# ----------------------------
# ENTRY POINT
# ----------------------------
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)




