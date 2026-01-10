import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import csv

# Start URL for HealthFeedback
BASE_DOMAIN = "https://science.feedback.org"

#ChatGpt: "Remove all analysis text, and extract only the misinformation claim from each article off the site: https://science.feedback.org."

#Crawl the site to get article URLs
def crawl_site(start_url, max_pages=200):
    visited = set()
    to_visit = [start_url]
    articles = set()

    while to_visit and len(visited) < max_pages:
        url = to_visit.pop()
        if url in visited:
            continue

        visited.add(url)

        try:
            r = requests.get(url, timeout=10)
            soup = BeautifulSoup(r.text, "html.parser")
        except:
            continue

        for a in soup.find_all("a", href=True):
            href = urljoin(BASE_DOMAIN, a["href"])

            if "/review/" in href:
                articles.add(href)

            if href.startswith(BASE_DOMAIN) and href not in visited:
                to_visit.append(href)

    return list(articles)

article_urls = crawl_site("https://science.feedback.org/health-feedback/")
print(article_urls[:10])
print(f"Found {len(article_urls)} articles")



#Extract title, claim, and verdict from each article
def extract_claim_only(url):
    """
    Extracts title, claim, and verdict from a HealthFeedback article
    - Tries to get only the misinformation claim, not the full analysis.
    """
    try:
        r = requests.get(url, timeout=20)
        soup = BeautifulSoup(r.text, "html.parser")

        # 1️⃣ Title
        title_tag = soup.find("h1")
        title = title_tag.get_text(strip=True) if title_tag else "No Title Found"

        # 2️⃣ Claim detection
        claim = None

        # First, look for bolded claim (often the actual claim)
        for p in soup.find_all("p"):
            strong_text = p.find("strong")
            if strong_text:
                claim_text = strong_text.get_text(strip=True)
                # Avoid including paragraphs that are clearly part of analysis
                if len(claim_text.split()) < 80:  # heuristic: short text is likely the claim
                    claim = claim_text
                    break

        # If no bolded claim found, try blockquotes
        if not claim:
            for bq in soup.find_all("blockquote"):
                text = bq.get_text(strip=True)
                if len(text.split()) < 80:
                    claim = text
                    break

        # If still not found, fallback: first paragraph that is short enough
        if not claim:
            for p in soup.find_all("p"):
                text = p.get_text(strip=True)
                # Skip if it contains verdict keywords (analysis)
                if any(word in text.lower() for word in ["false", "inaccurate", "incorrect", "unsupported", "no evidence", "misleading"]):
                    continue
                if len(text.split()) < 80:  # heuristic for short claim
                    claim = text
                    break

        if not claim:
            claim = "No Claim Found"

        # 3️⃣ Verdict detection
        verdict = "Unknown"
        verdict_keywords = ["false", "inaccurate", "incorrect", "unsupported", "no evidence", "misleading"]
        for p in soup.find_all("p"):
            text_lower = p.get_text(strip=True).lower()
            if any(word in text_lower for word in verdict_keywords):
                verdict = "false"
                break

        return title, claim, verdict

    except Exception as e:
        print(f"Failed to scrape {url}: {e}")
        return "Error", "Error", "Error"

csv_file_path = "/home/tacticrabbit/Datasets/ScrapedData.csv"

with open(csv_file_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["title", "claim", "verdict", "url"])
    writer.writeheader()

    for url in article_urls:  # your recursive crawl URLs
        title, claim, verdict = extract_claim_only(url)
        writer.writerow({
            "title": title,
            "claim": claim,
            "verdict": verdict,
            "url": url
        })

print(f"CSV saved successfully at {csv_file_path}")