# Imports
import requests
import logging
import csv
import time
import re

#logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

#constants
SEMANTIC_SCHOLAR_API_KEY = "put API key here"
SEMANTIC_SCHOLAR_URL = "https://api.semanticscholar.org/graph/v1/paper/search/bulk"
RATE_LIMIT_DELAY = 1  # seconds between request

# Keywords
AI_KEYWORDS = [
    "artificial intelligence", "ai", "generative ai", "machine learning", "deep learning", 
    "artificial neural networks", "convolutional neural networks", "predictive modelling",  "decision tree", "random forest",
    "large language models", "llm", "natural language processing", "nlp", "computer vision", 
    "support vector machine", "bayesian networks"]

DEMENTIA_KEYWORDS = ["dementia", "alzheimer", "vascular dementia", "lewy body dementia",
                     "frontotemporal dementia", "late-onset dementia", "late onset dementia",
                     "early onset dementia", "early-onset dementia", "senile dementia",
                     "subcortical dementia", "cortical dementia"]

def contains_keywords(text: str, ai_terms: list, dementia_terms: list) -> tuple[bool, str, str, str]:
    """
    Check if text contains at least one full term from each keyword list.
    """
    text = text.lower()
    found_ai = None
    found_dementia = None
    
    for term in ai_terms:
        pattern = r'\b' + re.escape(term.lower()) + r'\b'
        if re.search(pattern, text):
            found_ai = term
            break
            
    for term in dementia_terms:
        pattern = r'\b' + re.escape(term.lower()) + r'\b'
        if re.search(pattern, text):
            found_dementia = term
            break
    
    matches = bool(found_ai and found_dementia)
    details = f"AI term: {found_ai if found_ai else 'None'}, Dementia term: {found_dementia if found_dementia else 'None'}"
    
    return matches, found_ai, found_dementia, details

def fetch_semantic_scholar(query: str, year: int, start_idx: int) -> list:
    global response
    headers = {"x-api-key": SEMANTIC_SCHOLAR_API_KEY}
    all_data = []
    token = None
    max_retries = 3
    # Retry logic
    while True:
        retry_count = 0
        while retry_count < max_retries:
            try:
                params = {
                    "query": query,
                    "year": year,
                    "fields": "title,abstract,authors,year"
                }
                
                if token:
                    params["token"] = token

                logger.info(f"Making request for year {year} with params: {params}")
                response = requests.get(
                    SEMANTIC_SCHOLAR_URL,
                    headers=headers,
                    params=params,
                    timeout=30
                )

                logger.info(f"Request status: {response.status_code}")

                if response.status_code == 429:
                    wait_time = (retry_count + 1) * 10
                    logger.error(f'Rate limit exceeded. Waiting for {wait_time} seconds...')
                    time.sleep(wait_time)
                    retry_count += 1
                    continue

                if response.status_code in [500, 502, 503, 504]:
                    wait_time = (retry_count + 1) * 5
                    logger.error(f'Server error {response.status_code}. Waiting for {wait_time} seconds...')
                    time.sleep(wait_time)
                    retry_count += 1
                    continue

                if response.status_code == 400:
                    logger.error(f'400 Error: {response.text}')
                    return []

                response.raise_for_status()
                break
                
            except requests.exceptions.RequestException as e:
                wait_time = (retry_count + 1) * 5
                logger.error(f"Request failed: {str(e)}. Waiting {wait_time} seconds...")
                time.sleep(wait_time)
                retry_count += 1

        if retry_count == max_retries:
            logger.error("Max retries reached. Skipping this request.")
            break

        try:
            response_json = response.json()
            data = response_json.get("data", [])
            total_results = response_json.get("total", 0)
            
            if data:
                all_data.extend(data)
                logger.info(f"Retrieved {len(all_data)} records out of {total_results} total records")
            
            token = response_json.get("token")
            
            if not token or not data:
                break
                
            time.sleep(RATE_LIMIT_DELAY)
            
        except ValueError as e:
            logger.error(f"Failed to parse JSON response: {str(e)}")
            break
    
    papers = []
    skipped_count = 0
    current_idx = start_idx
    for item in all_data:
        try:
            # Check for required fields
            title = str(item.get("title", "") or "")
            abstract = str(item.get("abstract", "") or "")
            authors = item.get("authors", [])
            paper_year = item.get("year")
            
            # Skip papers missing required fields
            if not all([title, abstract, authors, paper_year]):
                skipped_count += 1
                logger.debug(f"Skipped paper (missing required fields): '{title[:100]}...'")
                continue
            
            combined_text = f"{title} {abstract}"
            matches, ai_term, dementia_term, match_details = contains_keywords(combined_text, AI_KEYWORDS, DEMENTIA_KEYWORDS)
            
            if matches:
                # Get author names
                author_names = [author.get('name', '') for author in authors]
                
                paper = {
                    "id": f"ss_{year}_{current_idx}",  # Use the global counter
                    "title": title,
                    "authors": ", ".join(filter(None, author_names)),
                    "abstract": abstract,
                    "year": str(paper_year),
                    "ai_term_found": ai_term,
                    "dementia_term_found": dementia_term
                }
                papers.append(paper)
                current_idx += 1  # Increment the counter for each accepted paper
                logger.info(f"Accepted paper: '{title[:100]}...' - {match_details}")
            else:
                skipped_count += 1
                logger.debug(f"Skipped paper: '{title[:100]}...' - {match_details}")
            
        except Exception as e:
            logger.error(f"Error processing paper {item}: {str(e)}")
            continue

    logger.info(f"Year {year}: Accepted {len(papers)} papers, Skipped {skipped_count} papers")
    return papers

def save_to_csv(papers: list) -> str | None:
    if not papers:
        logger.warning("No papers to save!")
        return None
        
    filename = f"semantic_scholar_results.csv"
    with open(filename, "w", newline="", encoding="utf-8") as csvfile:
        fieldnames = ["id", "title", "authors", "abstract", "year", "ai_term_found", "dementia_term_found"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for paper in papers:
            writer.writerow(paper)
    
    logger.info(f"Saved {len(papers)} papers to {filename}")
    return filename

def build_query():
    all_pairs = []
    for ai_term in AI_KEYWORDS:
        for dementia_term in DEMENTIA_KEYWORDS:
            query = f'({ai_term}) AND ({dementia_term})'
            all_pairs.append(query)
    return all_pairs


if __name__ == "__main__":
    try:
        queries = build_query()
        all_papers = []
        total_queries = len(queries)

        logger.info(f"Total number of queries to process: {total_queries}")
        logger.info(f"Number of AI keywords: {len(AI_KEYWORDS)}")
        logger.info(f"Number of Dementia keywords: {len(DEMENTIA_KEYWORDS)}")

        for year in range(2015, 2026):
            logger.info(f"\nProcessing year: {year}")
            year_papers = []
            paper_counter = 1  # Initialize counter for each year

            for i, query in enumerate(queries, 1):
                logger.info(f"Processing query {i}/{total_queries} for year {year}: {query}")
                try:
                    papers = fetch_semantic_scholar(query, year, paper_counter)
                    paper_counter += len(papers)  # Update counter based on number of papers found
                    year_papers.extend(papers)
                    logger.info(f"Retrieved {len(papers)} valid papers for this query")
                    time.sleep(RATE_LIMIT_DELAY)
                except Exception as e:
                    logger.error(f"Error processing query: {str(e)}")
                    continue

            # Remove duplicates where multiple keywords have matched and returned a paper more than once based on title
            seen_titles = set()
            unique_papers = []
            for paper in year_papers:
                if paper['title'] not in seen_titles:
                    seen_titles.add(paper['title'])
                    unique_papers.append(paper)

            logger.info(f"Year {year}: Found {len(year_papers)} papers, {len(unique_papers)} unique papers")
            all_papers.extend(unique_papers)

        # Save results
        csv_file = save_to_csv(all_papers)
        logger.info(f"Total unique papers saved to CSV: {len(all_papers)}")

    except Exception as e:
        logger.error(f"An error occurred in main execution: {str(e)}")