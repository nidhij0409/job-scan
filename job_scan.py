#!/usr/bin/env python3
import requests, os, json, yaml, time, re
import pandas as pd
from datetime import datetime
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np

# Load profile
with open("profile.yaml", "r") as f:
    PROFILE = yaml.safe_load(f)

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Target locations
LOCATIONS = [
    "Gandhinagar",
    "Nadiad",
    "Anand",
    "Surat",
    "Pune",
    "Nashik",
]

# include remote
REMOTE_QUERY = "remote"

# Clean text
def clean(s):
    s = (s or "").lower()
    s = re.sub(r"[^a-z0-9\s/-]", " ", s)
    return s.strip()

# ------------------------------
# ADZUNA FETCHER
# ------------------------------
def fetch_adzuna_jobs(query, location):
    app_id = os.getenv("ADZUNA_APP_ID")
    app_key = os.getenv("ADZUNA_APP_KEY")

    url = (
        f"https://api.adzuna.com/v1/api/jobs/in/search/1"
        f"?app_id={app_id}&app_key={app_key}"
        f"&results_per_page=50"
        f"&what={query}"
        f"&where={location}"
        f"&content-type=application/json"
    )

    try:
        resp = requests.get(url, timeout=20)
        data = resp.json()

        results = data.get("results", [])
        jobs = []

        for item in results:
            jobs.append({
                "source": "adzuna",
                "location": location,
                "title": item.get("title", ""),
                "company": item.get("company", {}).get("display_name", ""),
                "desc": item.get("description", ""),
                "link": item.get("redirect_url", "")
            })

        print(f"[{location}] Jobs fetched: {len(jobs)}")
        return jobs

    except Exception as e:
        print("Error fetching:", location, e)
        return []

# ------------------------------
# SCORING
# ------------------------------
def score(job):
    text = clean(job["title"] + " " + job["desc"])
    score = 0

    # Title relevance
    if any(k in text for k in ["qa", "quality", "test", "tester", "sdet", "automation"]):
        score += 10

    # Skills
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

    # Cap & label
    score = min(score, 100)
    job["score"] = score
    job["label"] = (
        "Excellent" if score >= 75 else
        "Good" if score >= 60 else
        "Potential" if score >= 40 else
        "Discard"
    )
    return job

# ------------------------------
# TRENDS
# ------------------------------
def trending(jobs):
    docs = [clean(j["title"] + " " + j["desc"]) for j in jobs if j["desc"]]

    if not docs:
        return [], []

    vectorizer = CountVectorizer(stop_words="english", max_features=1500)
    X = vectorizer.fit_transform(docs)
    freqs = np.asarray(X.sum(axis=0)).ravel()
    terms = vectorizer.get_feature_names_out()

    idx = np.argsort(freqs)[::-1][:25]
    trend_list = [{"term": terms[i], "count": int(freqs[i])} for i in idx]

    high = [t for t in trend_list if t["count"] >= len(docs) * 0.2]
    return trend_list, high

# ------------------------------
# MAIN
# ------------------------------
def main():
    all_jobs = []

    # City searches
    for loc in LOCATIONS:
        all_jobs += fetch_adzuna_jobs("QA Tester OR SDET OR Software Test Engineer", loc)
        time.sleep(1)

    # Remote search
    all_jobs += fetch_adzuna_jobs("QA Tester OR SDET", REMOTE_QUERY)

    scored = [score(j) for j in all_jobs]
    curated = [j for j in scored if j["label"] in ("Excellent", "Good")]

    trends, high = trending(all_jobs)

    now = datetime.now().strftime("%Y%m%d_%H%M")

    # Save outputs
    pd.DataFrame(curated).to_csv(f"{OUTPUT_DIR}/jobs_{now}.csv", index=False)

    with open(f"{OUTPUT_DIR}/trends_{now}.json", "w") as f:
        json.dump({"trends": trends, "high_demand": high}, f, indent=2)

    print("---- SUMMARY ----")
    print("Total jobs scraped:", len(all_jobs))
    print("Curated matches:", len(curated))
    print("Trend terms:", len(trends))
    print("-----------------")

if __name__ == "__main__":
    main()
