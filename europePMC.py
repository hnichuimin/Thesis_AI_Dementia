# Imports
import requests
import xml.etree.ElementTree as ET
import pandas as pd
import time
import logging
import re

# Set up logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Keywords
AI_KEYWORDS = [
    "artificial intelligence", "ai", "generative ai", "machine learning", "deep learning",
    "artificial neural networks", "convolutional neural networks", "predictive modelling",  "decision tree", "random forest",
    "large language models", "llm", "natural language processing", "nlp", "computer vision",
    "support vector machine", "bayesian networks"
]

DEMENTIA_KEYWORDS = [
    "dementia", "alzheimer", "vascular dementia", "lewy body dementia",
    "frontotemporal dementia", "late-onset dementia", "late onset dementia",
    "early onset dementia", "early-onset dementia", "senile dementia",
    "subcortical dementia", "cortical dementia"
]


MAX_EMPTY_RESULTS = 10
DELAY_BETWEEN_REQUESTS = 2

def build_query(ai_keywords, dementia_keywords, start_date, end_date):
    # Format each term set
    ai_terms = ' OR '.join(f'"{kw}"' for kw in ai_keywords)
    dementia_terms = ' OR '.join(f'"{kw}"' for kw in dementia_keywords)
    
    # Build query with proper field specification and nesting
    ai_part = f'(TITLE:({ai_terms}) OR ABSTRACT:({ai_terms}))'
    dementia_part = f'(TITLE:({dementia_terms}) OR ABSTRACT:({dementia_terms}))'
    date_range = f'(FIRST_PDATE:[{start_date} TO {end_date}])'
    
    # Combine everything
    query = f'{ai_part} AND {dementia_part} AND {date_range}'
    
    # Log the full query for debugging
    logger.info("Generated full query:")
    logger.info(query)
    
    return query

# Main functions
def fetch_europe_pmc(query, page_size=100, cursor="*", max_retries=5):
    base_url = 'https://www.ebi.ac.uk/europepmc/webservices/rest/search'
    payload = {
        'query': query,
        'format': 'xml',
        'resultType': 'core',
        'cursorMark': cursor,
        'pageSize': page_size
    }
    #Retry logic
    for attempt in range(max_retries):
        try:
            response = requests.get(base_url, params=payload)
            logger.info(f"Request status: {response.status_code}")

            if response.status_code == 200:
                return response.content

            elif response.status_code in [429, 502, 504]:
                wait_time = 10 * (attempt + 1)
                logger.error(f'{response.status_code} Error. Waiting for {wait_time} seconds before retry {attempt + 1}/{max_retries}...')
                time.sleep(wait_time)
                continue

            elif response.status_code == 400:
                logger.error(f'400 Error: {response.text}')
                return None

            else:
                logger.error(f'Unexpected status code {response.status_code}: {response.text}')
                return None

        except requests.exceptions.RequestException as e:
            logger.error(f"Request failed: {str(e)}")
            if attempt < max_retries - 1:
                time.sleep(5)
                continue
            return None

    logger.error("Max retries reached. Giving up.")
    return None
# Parsing the results
def parse_results(xml_content, ai_keywords, dementia_keywords):
    root = ET.fromstring(xml_content)
    papers = []
    batch_stats = {
        'total_in_batch': 0,
        'missing_fields': 0,
        'no_ai_matches': 0,
        'no_dementia_matches': 0,
        'passed_filters': 0
    }
    
    for result in root.findall('.//result'):
        batch_stats['total_in_batch'] += 1
        
        # Check for required fields
        title_elem = result.find('title')
        abstract_elem = result.find('abstractText')
        year_elem = result.find('pubYear')
        authors_elem = result.find('authorString')
        
        # Check if elements exist and have text content
        if (title_elem is None or abstract_elem is None or 
            year_elem is None or authors_elem is None):
            batch_stats['missing_fields'] += 1
            continue
            
        # Get text content of required fields
        title = title_elem.text
        abstract = abstract_elem.text
        year = year_elem.text
        authors = authors_elem.text
        
        # Skip if any required text is None or empty
        if not title or not abstract or not year or not authors:
            batch_stats['missing_fields'] += 1
            continue
        
        title_abstract_text = title + ' ' + abstract
        
        # Find AI terms that match using word boundaries
        ai_matches = find_keyword_matches(title_abstract_text, ai_keywords)
        if not ai_matches:
            batch_stats['no_ai_matches'] += 1
            continue
            
        dementia_matches = find_keyword_matches(title_abstract_text, dementia_keywords)
        if not dementia_matches:
            batch_stats['no_dementia_matches'] += 1
            continue
        
        ai_term_found = '; '.join(ai_matches)
        dementia_term_found = '; '.join(dementia_matches)
        
        # Create paper entry
        paper = {
            'title': title,
            'authors': authors,
            'abstract': abstract,
            'year': year,
            'ai_term_found': ai_term_found,
            'dementia_term_found': dementia_term_found
        }
        papers.append(paper)
        batch_stats['passed_filters'] += 1
    
    next_cursor_elem = root.find('.//nextCursorMark')
    next_cursor = next_cursor_elem.text if next_cursor_elem is not None else None
    
    hit_count_elem = root.find('.//hitCount')
    total_results = int(hit_count_elem.text) if hit_count_elem is not None else 0
    
    # More detailed logging
    logger.info(f"\nBatch Processing Summary:")
    logger.info(f"├── Papers in batch: {batch_stats['total_in_batch']}")
    logger.info(f"├── Filter Results:")
    logger.info(f"│   ├── Missing required fields: {batch_stats['missing_fields']}")
    logger.info(f"│   ├── No AI keyword matches: {batch_stats['no_ai_matches']}")
    logger.info(f"│   ├── No dementia keyword matches: {batch_stats['no_dementia_matches']}")
    logger.info(f"│   └── Passed all filters: {batch_stats['passed_filters']}")
    
    return papers, next_cursor, total_results

def find_keyword_matches(text, keywords):
    text = text.lower()
    matches = []
    for keyword in keywords:
        # Create word boundary pattern with proper escaping
        pattern = r'\b' + re.escape(keyword.lower()) + r'\b'
        if re.search(pattern, text):
            matches.append(keyword)
    return matches

# Main script section
if __name__ == "__main__":
    try:
        # Build query with date range
        query = build_query(AI_KEYWORDS, DEMENTIA_KEYWORDS, "2015-01-01", "2025-12-31")
        logger.info(f"Executing query: {query}\n")
        
        all_papers = []
        cursor = "*"
        total_results = None
        page_size = 100
        empty_result_count = 0  # Add counter for empty results
        MAX_EMPTY_RESULTS = 10  # Maximum number of empty results before stopping
        
        processed_total = 0
        
        while cursor is not None:
            logger.info(f"\n{'='*80}")
            logger.info(f"Fetching next batch with cursor: {cursor}")
            xml_content = fetch_europe_pmc(query, page_size=page_size, cursor=cursor)
            
            if xml_content is None:
                logger.error("Failed to fetch page - stopping")
                break
                
            papers, next_cursor, current_total = parse_results(xml_content, AI_KEYWORDS, DEMENTIA_KEYWORDS)
            processed_total += page_size
            
            if total_results is None:
                total_results = current_total
                logger.info(f"Total papers matching initial query: {total_results}")
            
            # Add check for completion before empty results check
            if processed_total >= total_results:
                logger.info("Retrieved all available results!")
                break
                
            if not papers:
                empty_result_count += 1
                logger.warning(f"No papers passed filters in this batch ({empty_result_count}/{MAX_EMPTY_RESULTS} empty batches)")
                remaining = total_results - processed_total
                logger.info(f"Still {remaining} papers to process ({(remaining/total_results)*100:.2f}% remaining)")
                
                if empty_result_count >= MAX_EMPTY_RESULTS and remaining > 500:  # Only stop if significant papers remain
                    logger.warning("Many consecutive empty batches, but significant papers remain. Continuing...")
                    empty_result_count = 0  # Reset counter to continue
                elif empty_result_count >= MAX_EMPTY_RESULTS and remaining <= 500:
                    logger.info("Nearly complete and hitting many empty batches - finishing up")
                    break
                    
                if next_cursor == cursor:
                    logger.warning("Cursor hasn't changed - stopping")
                    break
                cursor = next_cursor
                time.sleep(DELAY_BETWEEN_REQUESTS)
                continue
            
            empty_result_count = 0
            all_papers.extend(papers)
            
            logger.info(f"\nOverall Progress:")
            logger.info(f"├── Papers processed: {processed_total} of {total_results} ({(processed_total/total_results)*100:.2f}%)")
            logger.info(f"└── Valid papers collected: {len(all_papers)} (meeting all strict criteria)")
            
            if next_cursor == cursor:
                logger.warning("Cursor hasn't changed - stopping")
                break
                
            cursor = next_cursor
            
            # Add longer rate limiting
            time.sleep(DELAY_BETWEEN_REQUESTS)
            
            if len(all_papers) >= total_results:
                logger.info("Retrieved all available results")
                break
        
        logger.info(f"\nFinal count: Retrieved {len(all_papers)} out of {total_results} papers")
        
        # Creating dataframe
        if all_papers:
            df = pd.DataFrame(all_papers)
            
            # Add custom id column using year and index
            df['id'] = df.apply(lambda row: f"epmc_{row['year']}_{row.name + 1}", axis=1)
            
            # Specify the correct column order
            column_order = [
                'id',
                'title',
                'authors',
                'abstract',
                'year',
                'ai_term_found',
                'dementia_term_found'
            ]
            
            # Reorder columns and save to CSV
            df = df[column_order]
            filename = f'europepmc_results.csv'
            df.to_csv(filename, index=False, encoding='utf-8')
            logger.info(f"Results saved to {filename}")
        else:
            logger.error("No papers were retrieved successfully")
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")