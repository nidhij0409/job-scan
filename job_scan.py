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
    "Gandhinagar",
    "Nadiad",
    "Anand",
    "Surat",
    "Pune",
    "Nashik",
    "Remote"
]

# Updated realistic browser headers (avoids blocking)
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8"
}

def clean(s):
    s = (s or "").lower()
    s = re.sub(r"[^a-z0-9\s/-]", " ", s)
    return s.strip()

# ---------- INDEED FETCHER ----------
def fetch_indeed(location):
    query_loc = location if location.lower() != "remote" else "Remote"
    url = (
        "https://in.indeed.com/jobs?"
        "q=QA+Tester+OR+Quality+Engineer+OR+Software+Test+Engineer+OR+SDET"
        f"&l={query_loc}&radius=50"
    )

    jobs = []

    try:
        resp = requests.get(url, headers=HEADERS, timeout=20)
        soup = BeautifulSoup(resp.text, "html.parser")

        cards = soup.select("a.tapItem")
        if not cards:
            # New Indeed fallback layout
            cards = soup.select("div.job_seen_beacon a")

        for c in cards:
            title_el = c.select_one("h2.jobTitle span")
            comp_el = c.select_one("span.companyName")
            loc_el = c.select_one("div.companyLocation")
            desc_el = c.select_one("div.job-snippet")

            jobs.append({
                "source": "indeed",
                "location": location,
                "title": title_el.get_text(strip=True) if title_el else "",
                "company": comp_el.get_text(strip=True) if comp_el else "",
                "desc": desc_el.get_text(" ", strip=True) if desc_el else "",
                "link": "https://in.indeed.com" + c.get("href", "")
            })

    except Exception as e:
        print(f"Error fetching {location}: {e}")

    print(f"[{location}] Jobs fetched: {len(jobs)}")   # Debug line
    return jobs

# ---------- SCORING ----------
def score(job):
    text = clean(job["title"] + " " + job["desc"])

    score = 0
    # Title relevance
    if any(k in text for k in ["qa", "quality", "test", "sdet", "tester"]):
        score += 10

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

    # Label
    score = min(score, 100)
    job["score"] = score
    job["label"] = (
        "Excellent" if score >= 75
        else "Good" if score >= 60
        else "Potential" if score >= 40
        else "Discard"
    )
    return job

# ---------- TRENDS ----------
def trend_terms(jobs):
    docs = [clean(j["title"] + " " + j["desc"]) for j in jobs if j["desc"]]

    if not docs:
        return [], []

    vectorizer = CountVectorizer(stop_words="english", max_features=1500)
    X = vectorizer.fit_transform(docs)
    freqs = np.asarray(X.sum(axis=0)).ravel()
    terms = vectorizer.get_feature_names_out()

    idx = np.argsort(freqs)[::-1][:25]
    trends = [{"term": terms[i], "count": int(freqs[i])} for i in idx]

    high_demand = [t for t in trends if t["count"] >= len(docs) * 0.2]

    return trends, high_demand

# ---------- MAIN ----------
def main():
    all_jobs = []
    for loc in LOCATIONS:
        all_jobs += fetch_indeed(loc)
        time.sleep(1)  # polite pause

    scored = [score(j) for j in all_jobs]
    curated = [j for j in scored if j["label"] in ("Excellent", "Good")]

    trends, high = trend_terms(all_jobs)

    now = datetime.now().strftime("%Y%m%d_%H%M")

    pd.DataFrame(curated).to_csv(f"{OUTPUT_DIR}/jobs_{now}.csv", index=False)

    with open(f"{OUTPUT_DIR}/trends_{now}.json", "w") as f:
        json.dump({"trends": trends, "high_demand": high}, f, indent=2)

    print("---- SUMMARY ----")
    print("Total jobs scraped:", len(all_jobs))
    print("Curated matches:", len(curated))
    print("-----------------")

if __name__ == "__main__":
    main()
