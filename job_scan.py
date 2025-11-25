#!/usr/bin/env python3
import os, json, re, time
import requests
import yaml
import pandas as pd
from datetime import datetime
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np

# ---------------- CONFIG ----------------
with open("profile.yaml", "r") as f:
    PROFILE = yaml.safe_load(f)

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# target cities to keep (case-insensitive matching)
TARGET_CITIES = [
    "gandhinagar", "nadiad", "anand", "surat", "pune", "nashik"
]
REMOTE_KEYWORDS = ["remote", "work from home", "wfh", "work-from-home", "anywhere"]

# Adzuna: India (country code 'in') endpoint
ADZUNA_BASE = "https://api.adzuna.com/v1/api/jobs/in/search/{}"

# polite pause between requests
PAUSE_SEC = 1.0

# ---------------- utilities ----------------
def clean_text(s):
    s = (s or "")
    s = s.lower()
    s = re.sub(r"[^a-z0-9\s/+-]", " ", s)
    return re.sub(r"\s+", " ", s).strip()

def is_remote_text(text):
    t = clean_text(text)
    return any(k in t for k in REMOTE_KEYWORDS)

def city_matches(text):
    t = clean_text(text)
    matches = [c for c in TARGET_CITIES if c in t]
    return matches  # list possibly empty

# ---------------- Adzuna fetcher ----------------
def fetch_adzuna_page(page=1, what="QA Tester OR SDET OR Software Test Engineer", where="India", results_per_page=50):
    app_id = os.getenv("ADZUNA_APP_ID")
    app_key = os.getenv("ADZUNA_APP_KEY")
    if not app_id or not app_key:
        raise RuntimeError("ADZUNA_APP_ID and ADZUNA_APP_KEY must be set in environment")

    url = ADZUNA_BASE.format(page)
    params = {
        "app_id": app_id,
        "app_key": app_key,
        "results_per_page": results_per_page,
        "what": what,
        "where": where,
        "content-type": "application/json"
    }
    try:
        r = requests.get(url, params=params, timeout=20)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        print("Adzuna request error:", e)
        return {}

def fetch_all_adzuna_jobs(max_pages=2):
    jobs = []
    for p in range(1, max_pages + 1):
        data = fetch_adzuna_page(page=p)
        if not data:
            break
        results = data.get("results", [])
        if not results:
            break
        for it in results:
            jobs.append({
                "source": "adzuna",
                "title": it.get("title", "") or "",
                "company": (it.get("company") or {}).get("display_name", "") or "",
                "location": it.get("location", {}).get("display_name", "") or it.get("location", ""),
                "desc": it.get("description", "") or "",
                "link": it.get("redirect_url", "") or it.get("link", "")
            })
        time.sleep(PAUSE_SEC)
    return jobs

# ---------------- scoring & filtering ----------------
def score_job(job):
    text = clean_text(job.get("title", "") + " " + job.get("desc", "") + " " + job.get("company", ""))
    score = 0
    # title relevance
    if any(k in text for k in ["qa", "quality", "test", "tester", "sdet", "automation"]):
        score += 10
    # skill tokens
    skill_score = 0
    cores = PROFILE.get("skills", {}).get("core", [])
    secs = PROFILE.get("skills", {}).get("secondary", [])
    for s in cores:
        if s.lower() in text:
            skill_score += 6
    for s in secs:
        if s.lower() in text:
            skill_score += 3
    score += min(skill_score, 40)
    # domain match
    for d in PROFILE.get("domains", []):
        if d.lower() in text:
            score += 5
    job["score"] = min(score, 100)
    job["label"] = ("Excellent" if job["score"] >= 75 else
                    "Good" if job["score"] >= 60 else
                    "Potential" if job["score"] >= 40 else "Discard")
    return job

def filter_for_targets(jobs):
    kept = []
    for j in jobs:
        loc_text = j.get("location", "") or ""
        desc_text = j.get("desc", "") or ""
        # remote match
        if is_remote_text(loc_text) or is_remote_text(desc_text):
            j["_matched_locations"] = ["remote"]
            kept.append(j)
            continue
        # city match
        matched = city_matches(loc_text) or city_matches(desc_text)
        if matched:
            j["_matched_locations"] = matched
            kept.append(j)
    return kept

# ---------------- trending ----------------
def extract_trends(jobs, top_k=25):
    docs = [clean_text((j.get("title","") + " " + j.get("desc",""))) for j in jobs if (j.get("desc","") or "").strip()]
    if not docs:
        return [], []
    vectorizer = CountVectorizer(stop_words="english", max_features=2000)
    X = vectorizer.fit_transform(docs)
    freqs = np.asarray(X.sum(axis=0)).ravel()
    terms = vectorizer.get_feature_names_out()
    idx = np.argsort(freqs)[::-1][:top_k]
    trends = [{"term": terms[i], "count": int(freqs[i])} for i in idx]
    high = [t for t in trends if t["count"] >= len(docs) * 0.2]
    return trends, high

# ---------------- main ----------------
def main():
    print("Starting Adzuna India-wide fetch -> filter for target cities & remote")
    # fetch India-wide QA jobs (pages can be increased if needed)
    raw_jobs = fetch_all_adzuna_jobs(max_pages=3)  # ~ up to 150 results
    print("Fetched total Adzuna results:", len(raw_jobs))

    # filter locally for target cities + remote
    matched_jobs = filter_for_targets(raw_jobs)
    print("After filtering for target cities & remote:", len(matched_jobs))

    # score
    scored = [score_job(j) for j in matched_jobs]

    # dedupe by link
    seen = set()
    deduped = []
    for j in scored:
        key = (j.get("link") or j.get("title","") + j.get("company",""))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(j)

    # curated (Excellent + Good)
    curated = [j for j in deduped if j.get("label") in ("Excellent","Good")]

    # trends across all fetched jobs (not only filtered) to surface what market demands
    trends, high = extract_trends(raw_jobs)

    now = datetime.now().strftime("%Y%m%d_%H%M")
    jobs_out = os.path.join(OUTPUT_DIR, f"jobs_{now}.csv")
    trends_out = os.path.join(OUTPUT_DIR, f"trends_{now}.json")

    pd.DataFrame(curated).to_csv(jobs_out, index=False)

    with open(trends_out, "w", encoding="utf-8") as f:
        json.dump({"trends": trends, "high_demand": high}, f, indent=2)

    print("Saved:", jobs_out, trends_out)
    print("Curated count:", len(curated))
    print("Trends count:", len(trends))

if __name__ == "__main__":
    main()
