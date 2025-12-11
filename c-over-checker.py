# c-over-checker.py
# Author: Kyra Menai Hamilton

# script to check for overlap between a new proposal and existing topics online for FMARS. https://www.frontiersin.org/journals/marine-science/research-topics?submission=1&sort=3

import requests
from bs4 import BeautifulSoup
import difflib 
import re
import sys
import os
import json
import time
from datetime import datetime
from urllib.parse import quote_plus
import logging
from typing import List, Tuple
logging.basicConfig(level=logging.INFO)

def fetch_existing_topics(url: str) -> List[Tuple[str, str]]:
    """Fetch existing research topics from the given URL."""
    response = requests.get(url)
    if response.status_code != 200:
        logging.error(f"Failed to fetch topics: {response.status_code}")
        return []
    
    soup = BeautifulSoup(response.content, 'html.parser')
    topics = []
    
    for topic in soup.find_all('div', class_='research-topic'):
        title = topic.find('h3').get_text(strip=True)
        description = topic.find('p').get_text(strip=True)
        topics.append((title, description))
    
    logging.info(f"Fetched {len(topics)} existing topics.")
    return topics
def clean_text(text: str) -> str:
    """Clean and normalize text for comparison."""
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s]', '', text)
    return text.strip()
def calculate_similarity(text1: str, text2: str) -> float:
    """Calculate similarity between two texts."""
    return difflib.SequenceMatcher(None, text1, text2).ratio()
def check_overlap(new_proposal: str, existing_topics: List[Tuple[str, str]], threshold: float = 0.6) -> List[Tuple[str, str, float]]:
    """Check for overlap between the new proposal and existing topics."""
    overlaps = []
    cleaned_new_proposal = clean_text(new_proposal)
    
    for title, description in existing_topics:
        cleaned_description = clean_text(description)
        similarity = calculate_similarity(cleaned_new_proposal, cleaned_description)
        
        if similarity >= threshold:
            overlaps.append((title, description, similarity))
    
    logging.info(f"Found {len(overlaps)} overlapping topics.")
    return overlaps
def main():
    if len(sys.argv) != 2:
        logging.error("Usage: python c-over-checker.py <path_to_new_proposal.txt>")
        sys.exit(1)
    
    proposal_path = sys.argv[1]
    
    if not os.path.isfile(proposal_path):
        logging.error(f"File not found: {proposal_path}")
        sys.exit(1)
    
    with open(proposal_path, 'r', encoding='utf-8') as file:
        new_proposal = file.read()
    
    existing_topics_url = "https://www.frontiersin.org/journals/marine-science/research-topics?submission=1&sort=3"
    existing_topics = fetch_existing_topics(existing_topics_url)
    
    overlaps = check_overlap(new_proposal, existing_topics)
    
    if overlaps:
        logging.info("Overlapping topics found:")
        for title, description, similarity in overlaps:
            logging.info(f"Title: {title}\nDescription: {description}\nSimilarity: {similarity:.2f}\n")
    else:
        logging.info("No overlapping topics found.")    
if __name__ == "__main__":
    main()  

# imputting the new proposal text file path as a command line argument

# c-over-checker.py ends here
