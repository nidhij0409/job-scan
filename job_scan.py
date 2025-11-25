#!/usr/bin/env python3
import requests, json, re, os, yaml, time
from bs4 import BeautifulSoup
import pandas as pd
from datetime import datetime
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np

# Load profile
with open("profile.yaml", "r") as f:
    PROFILE = yaml.safe_load(f)

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

LOCATIONS = [
    "Gandhinagar", "Nadiad", "Anand", "Surat",
    "Pune", "Nashik", "Remote"
]

def clean(s):
    s = (s or "").lower()
    s = re.sub(r"[^a-z0-9\s/-]", "", s)
    return s

# Fetch jobs from a simple public Naukri search proxy
def fetch_naukri(location):
    url = f"https://www.naukri.com/{location.lower()}-jobs"
    headers = {"User-Agent": "Mozilla/5.0 JobScanBot"}
    try:
        r = requests.get(url, headers=headers, timeout=15)
        soup = BeautifulSoup(r.text, "html.parser")

        jobs = []
        for card in soup.select("article.jobTuple"):
            title = card.select_one("a.title")
            comp = card.select_one("a.subTitle")
            desc = card.select_one("div.job-description")
            link = title["href"] if title else ""

            jobs.append({
                "source": "naukri",
                "location": location,
                "title": title.get_text(strip=True) if title else "",
                "company": comp.get_text(strip=True) if comp else "",
                "desc": desc.get_text(strip=True) if desc else "",
                "link": link
            })
        return jobs
    except:
        return []

def score(job):
    text = clean(job["title"] + " " + job["desc"])

    score = 0
    # Title relevance
    for k in ["qa", "quality", "test", "sdet", "tester"]:
        if k in text:
            score += 10
            break

    # Skills match
    skill_score = 0
    for s in PROFILE["skills"]["core"]:
        if s in text:
            skill_score += 6

    for s in PROFILE["skills"]["secondary"]:
        if s in text:
            skill_score += 3

    score += min(skill_score, 40)

    # Domain match
    for d in PROFILE["domains"]:
        if d in text:
            score += 5

    job["score"] = min(score, 100)
    job["label"] = (
        "Excellent" if score >= 75
        else "Good" if score >= 60
        else "Potential" if score >= 40
        else "Discard"
    )
    return job

def trend_terms(jobs):
    docs = [clean(j["title"] + " " + j["desc"]) for j in jobs]
    if not docs:
        return [], []

    vectorizer = CountVectorizer(stop_words="english", max_features=5000)
    X = vectorizer.fit_transform(docs)
    freqs = np.asarray(X.sum(axis=0)).ravel()
    terms = vectorizer.get_feature_names_out()

    idx = np.argsort(freqs)[::-1][:25]
    trends = [{"term": terms[i], "count": int(freqs[i])} for i in idx]

    high_demand = [t for t in trends if t["count"] >= len(jobs) * 0.2]

    return trends, high_demand

def main():
    all_jobs = []
    for loc in LOCATIONS:
        all_jobs += fetch_naukri(loc)
        time.sleep(1)

    scored = [score(j) for j in all_jobs]
    curated = [j for j in scored if j["label"] in ("Excellent", "Good")]

    trends, high = trend_terms(all_jobs)

    now = datetime.now().strftime("%Y%m%d_%H%M")

    pd.DataFrame(curated).to_csv(f"{OUTPUT_DIR}/jobs_{now}.csv", index=False)

    with open(f"{OUTPUT_DIR}/trends_{now}.json", "w") as f:
        json.dump({"trends": trends, "high_demand": high}, f, indent=2)

    print("Finished run:", now)

if __name__ == "__main__":
    main()
