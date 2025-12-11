# c-over-checker.py
# Author: Kyra Menai Hamilton

# script to check for overlap between a new proposal and existing topics online for FMARS. https://www.frontiersin.org/journals/marine-science/research-topics?submission=1&sort=3

import argparse
import requests
from bs4 import BeautifulSoup
import difflib 
import re
import sys
import os
import json
import time
from datetime import datetime
from urllib.parse import quote_plus, urljoin
import logging
from typing import List, Tuple

# lightweight stop word list for keyword extraction
STOP_WORDS = set([
    'the','and','is','in','to','of','a','for','on','with','that','this','by','an','as','are','be','or','from',
    'at','it','we','our','can','will','which','these','their','has','have','but','not','they','used','use'
])
logging.basicConfig(level=logging.INFO)

def fetch_topic_page(url: str) -> Tuple[str, str, str]:
    """Fetch title, description, and full text from an individual topic page."""
    try:
        resp = requests.get(url, timeout=10)
    except Exception as e:
        logging.debug(f"Error fetching topic page {url}: {e}")
        return ("", "")

    if resp.status_code != 200:
        logging.debug(f"Non-200 for {url}: {resp.status_code}")
        return ("", "")

    soup = BeautifulSoup(resp.content, 'html.parser')

    title = ""
    if soup.find('h1'):
        title = soup.find('h1').get_text(strip=True)
    elif soup.title:
        title = soup.title.get_text(strip=True)

    # try meta description first
    description = ""
    meta = soup.find('meta', attrs={'name': 'description'}) or soup.find('meta', attrs={'property': 'og:description'})
    if meta and meta.get('content'):
        description = meta.get('content').strip()
    else:
        # fallback: first paragraph in article/main
        article = soup.find('article') or soup.find('main') or soup
        p = article.find('p') if article else None
        if p:
            description = p.get_text(strip=True)

    # full page text fallback
    full_text = soup.get_text(separator=' ', strip=True)
    return (title, description, full_text)


def fetch_existing_topics(url: str) -> List[Tuple[str, str, str]]:
    """Fetch existing research topics from the given "main" URL.

    This will attempt to find links to individual topic pages on the main page,
    follow each link and extract the title + description. If no useful links
    are found, it falls back to scraping topic blocks on the main page.
    """
    try:
        response = requests.get(url, timeout=10)
    except Exception as e:
        logging.error(f"Failed to fetch topics main page: {e}")
        return []

    if response.status_code != 200:
        logging.error(f"Failed to fetch topics: {response.status_code}")
        return []

    soup = BeautifulSoup(response.content, 'html.parser')
    topics: List[Tuple[str, str, str]] = []

    # find candidate links
    links = []
    for a in soup.find_all('a', href=True):
        href = a['href']
        if href.startswith('mailto:') or href.startswith('#'):
            continue
        full = urljoin(url, href)
        links.append(full)

    # prefer links that look like topic pages (heuristic)
    candidate_links = []
    for l in links:
        if any(k in l for k in ('research-topics', '/articles/', '/topic/', '/topics/')):
            candidate_links.append(l)

    # if we found no likely candidates, try using all links (small limit)
    if not candidate_links:
        candidate_links = links[:50]

    seen = set()
    for link in candidate_links:
        if link in seen:
            continue
        seen.add(link)
        title, desc, full = fetch_topic_page(link)
        if title or desc or full:
            topics.append((title or link, desc, full))
            logging.debug(f"Added topic from {link}: {title}")
        time.sleep(0.5)

    # fallback: if no topics collected, try older page structure
    if not topics:
        for topic in soup.find_all('div', class_='research-topic'):
            h = topic.find('h3')
            p = topic.find('p')
            if h and p:
                title = h.get_text(strip=True)
                description = p.get_text(strip=True)
                topics.append((title, description, description))

    logging.info(f"Fetched {len(topics)} existing topics.")
    return topics
def clean_text(text: str) -> str:
    """Clean and normalize text for comparison."""
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s]', '', text)
    return text.strip()


def extract_keywords(text: str, top_n: int = 20) -> List[str]:
    """Return top_n keywords from text (simple frequency-based, stopword filtered)."""
    text = clean_text(text)
    tokens = [t for t in text.split() if t and not t.isdigit() and t not in STOP_WORDS and len(t) > 2]
    freq = {}
    for t in tokens:
        freq[t] = freq.get(t, 0) + 1
    # sort by frequency then alphabetically
    items = sorted(freq.items(), key=lambda x: (-x[1], x[0]))
    return [w for w, _ in items[:top_n]]


def keyword_overlap_ratio(kws1: List[str], kws2: List[str]) -> float:
    """Compute keyword overlap ratio between two keyword lists.

    Ratio = intersection_size / min(len(kws1), len(kws2)) if both non-empty else 0.
    """
    s1 = set(kws1)
    s2 = set(kws2)
    if not s1 or not s2:
        return 0.0
    inter = s1.intersection(s2)
    return len(inter) / float(min(len(s1), len(s2)))
def calculate_similarity(text1: str, text2: str) -> float:
    """Calculate similarity between two texts."""
    return difflib.SequenceMatcher(None, text1, text2).ratio()
def check_overlap(new_proposal: str, existing_topics: List[Tuple[str, str, str]], threshold: float = 0.6,
                  kw_threshold: float = 0.35, kw_top: int = 20) -> List[Tuple[str, str, str, float, float]]:
    """Check for overlap between the new proposal and existing topics.

    For each topic we compare against title, description, and the full page text
    and use the maximum similarity value.
    """
    overlaps = []
    cleaned_new_proposal = clean_text(new_proposal)

    # prepare proposal keywords
    proposal_keywords = extract_keywords(new_proposal, top_n=kw_top)

    for title, description, full_text in existing_topics:
        sims = []
        if title:
            sims.append(calculate_similarity(cleaned_new_proposal, clean_text(title)))
        if description:
            sims.append(calculate_similarity(cleaned_new_proposal, clean_text(description)))
        if full_text:
            sims.append(calculate_similarity(cleaned_new_proposal, clean_text(full_text)))

        similarity = max(sims) if sims else 0.0

        # compute keyword overlap using best available text (description or full_text)
        target_text = description or full_text or ''
        target_keywords = extract_keywords(target_text, top_n=kw_top)
        kw_ratio = keyword_overlap_ratio(proposal_keywords, target_keywords)

        # report if either similarity or keyword overlap passes thresholds
        if similarity >= threshold or kw_ratio >= kw_threshold:
            overlaps.append((title, description or full_text, full_text, similarity, kw_ratio))

    logging.info(f"Found {len(overlaps)} overlapping topics.")
    return overlaps
def main():
    parser = argparse.ArgumentParser(description='Check new proposal text against existing research topics')
    parser.add_argument('proposal', help='Path to new proposal text file')
    parser.add_argument('--topics-url', help='URL of the main research topics page to fetch (optional)',
                        default='https://www.frontiersin.org/journals/marine-science/research-topics?submission=1&sort=3')
    parser.add_argument('--similarity-threshold', type=float, default=0.6,
                        help='Text similarity threshold (0-1) to flag overlaps')
    parser.add_argument('--kw-threshold', type=float, default=0.35,
                        help='Keyword-overlap ratio threshold (0-1) to flag overlaps')
    parser.add_argument('--kw-top', type=int, default=20,
                        help='Number of top keywords to extract for overlap checks')
    parser.add_argument('--show-full', action='store_true', help='Print full matched topic page text')
    args = parser.parse_args()

    def fetch_proposal(source: str) -> str:
        """Accepts a local file path, an http(s) URL, or '-' for stdin and returns text."""
        if source == '-':
            return sys.stdin.read()

        if source.startswith('http://') or source.startswith('https://'):
            try:
                r = requests.get(source, timeout=10)
                if r.status_code == 200:
                    soup = BeautifulSoup(r.content, 'html.parser')
                    # prefer article/main text if present
                    main = soup.find('article') or soup.find('main')
                    text = ''
                    if main:
                        text = main.get_text(separator=' ', strip=True)
                    else:
                        text = soup.get_text(separator=' ', strip=True)
                    return text
                else:
                    logging.error(f"Failed to fetch proposal URL: {r.status_code}")
                    return ''
            except Exception as e:
                logging.error(f"Error fetching proposal URL: {e}")
                return ''

        # otherwise treat as local file
        if os.path.isfile(source):
            with open(source, 'r', encoding='utf-8') as f:
                return f.read()

        logging.error(f"Proposal source not found or unsupported: {source}")
        return ''

    proposal_source = args.proposal
    new_proposal = fetch_proposal(proposal_source)
    if not new_proposal:
        logging.error('No proposal text available; exiting.')
        sys.exit(1)

    existing_topics = fetch_existing_topics(args.topics_url)

    overlaps = check_overlap(new_proposal, existing_topics,
                             threshold=args.similarity_threshold,
                             kw_threshold=args.kw_threshold,
                             kw_top=args.kw_top)

    if overlaps:
        logging.info("Overlapping topics found:")
        for title, description, full_text, similarity, kw_ratio in overlaps:
            if args.show_full:
                logging.info(f"Title: {title}\nFull text:\n{full_text}\nSimilarity: {similarity:.2f}\nKeyword overlap: {kw_ratio:.2f}\n")
            else:
                logging.info(f"Title: {title}\nDescription: {description}\nSimilarity: {similarity:.2f}\nKeyword overlap: {kw_ratio:.2f}\n")
    else:
        logging.info("No overlapping topics found.")    
if __name__ == "__main__":
    main()  

# imputting the new proposal text file path as a command line argument
# Now all parts have been defined and the script is complete, need to run it in an appropriate environment with internet access to fetch existing topics.

# c-over-checker.py ends here