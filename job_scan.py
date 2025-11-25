#!/usr/bin/env python3
import os, json, re, time
import requests
import yaml
import pandas as pd
from datetime import datetime
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np

# ---------------- LOAD PROFILE ----------------
with open("profile.yaml", "r") as f:
    PROFILE = yaml.safe_load(f)

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------- SEARCH CONFIG ----------------
# PAN-India search + remote jobs
WHERE = "India"

REMOTE_KEYWORDS = [
    "remote", "wfh", "work from home", "work-from-home", "anywhere"
]

# Titles/keywords expansion
SEARCH_TERMS = (
    "QA OR Tester OR Testing OR SDET OR "
    "\"Software Test\" OR \"Quality Assurance\" OR "
    "\"Automation\" OR \"Manual Tester\" OR "
    "\"QA Engineer\" OR \"Test Engineer\" OR "
    "\"Quality Engineer\""
)

# MAX pages to fetch (1 page = 50 results)
MAX_PAGES = 3

# ---------------- UTILS ----------------
def clean(s):
    s = (s or "")
    s = s.lower()
    s = re.sub(r"[^a-z0-9\s/+-]", " ", s)
    return re.sub(r"\s+", " ", s).strip()

def is_remote(t):
    t = clean(t)
    return any(k in t for k in REMOTE_KEYWORDS)

# ---------------- ADZUNA FETCHER ----------------
def fetch_adzuna_page(page=1, query=SEARCH_TERMS, where=WHERE):
    app_id = os.getenv("ADZUNA_APP_ID")
    app_key = os.getenv("ADZUNA_APP_KEY")
    if not app_id or not app_key:
        raise RuntimeError("Missing ADZUNA_APP_ID or ADZUNA_APP_KEY")

    url = f"https://api.adzuna.com/v1/api/jobs/in/search/{page}"
    params = {
        "app_id": app_id,
        "app_key": app_key,
        "results_per_page": 50,
        "what": query,
        "where": where,
        "content-type": "application/json"
    }

    try:
        resp = requests.get(url, params=params, timeout=15)
        resp.raise_for_status()
        return resp.json().get("results", [])
    except Exception as e:
        print("Adzuna error:", e)
        return []

def fetch_pan_india_jobs():
    all_jobs = []
    for p in range(1, MAX_PAGES + 1):
        results = fetch_adzuna_page(page=p)
        if not results:
            break
        for it in results:
            all_jobs.append({
                "source": "adzuna",
                "title": it.get("title", ""),
                "company": (it.get("company") or {}).get("display_name", ""),
                "location": it.get("location", {}).get("display_name", ""),
                "desc": it.get("description", ""),
                "link": it.get("redirect_url", "")
            })
        time.sleep(1)
    return all_jobs

# ---------------- SCORING ----------------
def score(job):
    text = clean(job["title"] + " " + job["desc"])

    score = 0

    # Title relevance
    if any(k in text for k in ["qa", "quality", "test", "sdet", "automation", "tester"]):
        score += 10
