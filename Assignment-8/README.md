# AI Resume Screening System with Tracing

## Overview
This project implements an AI Resume Screening System that evaluates resumes against a given job description.  
The pipeline performs skill extraction, matching, scoring, and explanation generation.  
LangSmith tracing is enabled to monitor and debug each stage of the workflow.

## Objective
The goal of this project is to:
- Build an AI-powered resume screening pipeline
- Compare resumes with job requirements
- Generate a fit score with explanation
- Use LangSmith tracing for debugging and monitoring

## Pipeline Flow
Resume → Extract → Match → Score → Explain

## Features
- Resume skill extraction
- Job-resume matching
- Candidate scoring (0–100)
- Explainable output
- LangSmith tracing support

## Tech Stack
- Python
- LangChain
- LangSmith
- Jupyter Notebook

## Files
- `AI Resume Screening System with Tracing.ipynb` – main notebook
- `screenshots/` – LangSmith tracing screenshots
- `README.md` – project documentation

## Output
The system evaluates:
- Strong candidate
- Average candidate
- Weak candidate

For each candidate, it returns:
- Fit score
- Matched skills
- Missing skills
- Explanation

## LangSmith Tracing
LangSmith tracing is enabled using:
- `LANGCHAIN_API_KEY`
- `LANGCHAIN_TRACING_V2=true`
- `LANGCHAIN_PROJECT=ai-resume-screening`

This helps inspect each step of the pipeline and debug incorrect outputs.

## Conclusion
This project demonstrates how to build a modular AI evaluation workflow for resume screening.  
It also shows how LangSmith can be used for observability and tracing in AI systems.
