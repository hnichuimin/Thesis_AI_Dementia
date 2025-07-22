#imports
from Bio import Entrez
import time
from urllib.error import URLError
from datetime import datetime
import pandas as pd
import re
from http.client import IncompleteRead

#constants
Entrez.email = 'put email here'
Entrez.api_key = 'put API key here'
#ratelimits
REQUESTS_PER_SECOND = 10
REQUEST_DELAY = 1.0 / REQUESTS_PER_SECOND

#keywords
AI_KEYWORDS = [
    "artificial intelligence", "ai", "generative ai", "machine learning", "deep learning",
    "artificial neural networks", "convolutional neural networks", "predictive modelling",  "decision tree", "random forest",
    "large language models", "llm", "natural language processing", "nlp", "computer vision",
    "support vector machine", "bayesian networks"]

DEMENTIA_KEYWORDS = ["dementia", "alzheimer", "vascular dementia", "lewy body dementia",
                     "frontotemporal dementia", "late-onset dementia", "late onset dementia",
                     "early onset dementia", "early-onset dementia", "senile dementia",
                     "subcortical dementia", "cortical dementia"]

def build_query(start_date="2015/01/01", end_date="2025/12/31", date_field="Create"):

    #Build a PubMed query with date range and keywords

    try:
        datetime.strptime(start_date, "%Y/%m/%d")
        datetime.strptime(end_date, "%Y/%m/%d")
    except ValueError as e:
        raise ValueError("Dates must be in YYYY/MM/DD format") from e
        
    valid_date_fields = ['Create', 'Publication', 'Modification']
    if date_field not in valid_date_fields:
        raise ValueError(f"date_field must be one of {valid_date_fields}")

    publication_filter = 'AND ("journal article"[Publication Type] NOT "conference"[Publication Type] OR "poster"[Publication Type]))'
    
    date_range = f'("{start_date}"[Date - {date_field}] : "{end_date}"[Date - {date_field}])'
    
    # Require abstract to be present
    has_abstract = 'AND hasabstract[text]'
    
    #keyword search to ensure more precise matching
    dementia_queries = [f'("{topic}"[Title/Abstract])' for topic in DEMENTIA_KEYWORDS]
    ai_queries = [f'("{topic}"[Title/Abstract])' for topic in AI_KEYWORDS]
    
    return f"({' OR '.join(dementia_queries)}) AND ({' OR '.join(ai_queries)}) AND {date_range} {has_abstract} {publication_filter}"

def get_all_pmids(query):
    try:
        max_retries = 3
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                # First get the total count of matches from pubmed
                search_handle = Entrez.esearch(db='pubmed', term=query, retmax=0, api_key=Entrez.api_key)
                search_results = Entrez.read(search_handle)
                search_handle.close()
                break
            except (URLError, IncompleteRead) as e:
                retry_count += 1
                if retry_count == max_retries:
                    print(f"Failed to get initial count after {max_retries} attempts. Error: {str(e)}")
                    return []
                print(f"Attempt {retry_count} failed. Retrying after {retry_count * 5} seconds...")
                time.sleep(retry_count * 5)
        
        total_count = int(search_results['Count'])
        print(f"Total matching articles found: {total_count}")
        
        # Now fetch all IDs in batches
        all_ids = []
        batch_size = 5000  # NCBI recommends not more than 5000 per request
        
        for start in range(0, total_count, batch_size):
            retry_count = 0
            while retry_count < max_retries:
                try:
                    print(f"Fetching IDs {start+1} to {min(start+batch_size, total_count)}...")
                    search_handle = Entrez.esearch(
                        db='pubmed',
                        term=query,
                        retstart=start,
                        retmax=batch_size,
                        api_key=Entrez.api_key
                    )
                    search_results = Entrez.read(search_handle)
                    search_handle.close()
                    all_ids.extend(search_results['IdList'])
                    time.sleep(REQUEST_DELAY)
                    break
                except (URLError, IncompleteRead) as e:
                    retry_count += 1
                    if retry_count == max_retries:
                        print(f"Failed to fetch batch after {max_retries} attempts. Error: {str(e)}")
                        continue
                    print(f"Attempt {retry_count} failed. Retrying after {retry_count * 5} seconds...")
                    time.sleep(retry_count * 5)
        
        return all_ids
    
    except Exception as e:
        print(f"An error occurred while fetching PMIDs: {str(e)}")
        return []

def fetch_pubmed_data(query, batch_size=100):
    try:
        id_list = get_all_pmids(query)
        
        if not id_list:
            print("No results found for the given query.")
            return pd.DataFrame()

        print(f"Starting download of {len(id_list)} articles...")
        
        records_data = []
        idx = 0
        
        for i in range(0, len(id_list), batch_size):
            batch_ids = id_list[i:i + batch_size]
            max_retries = 3
            retry_count = 0
            
            while retry_count < max_retries:
                try:
                    fetch_handle = Entrez.efetch(db='pubmed', id=batch_ids, retmode='xml', api_key=Entrez.api_key)
                    records = Entrez.read(fetch_handle)
                    fetch_handle.close()
                    break  # If successful, exit the retry loop
                except (URLError, IncompleteRead) as e:
                    retry_count += 1
                    if retry_count == max_retries:
                        print(f"Failed to fetch batch after {max_retries} attempts. Error: {str(e)}")
                        continue  # Skip to next batch
                    print(f"Attempt {retry_count} failed. Retrying after {retry_count * 5} seconds...")
                    time.sleep(retry_count * 5)
            
            for record in records['PubmedArticle']:
                try:
                    article = record['MedlineCitation']['Article']
                    
                    # Validate required fields
                    if not all(key in article for key in ['ArticleTitle', 'Abstract']):
                        continue
                        
                    title = article['ArticleTitle']
                    abstract = ' '.join(article['Abstract']['AbstractText'])
                    
                    # Get publication year
                    pub_date = article['Journal']['JournalIssue']['PubDate']
                    pub_year = pub_date.get('Year', '')
                    if not pub_year:  # Skip if no year available
                        continue
                        
                    # Get authors
                    authors = []
                    if 'AuthorList' in article:
                        for author in article['AuthorList']:
                            author_name = f"{author.get('LastName', '')} {author.get('ForeName', '')}".strip()
                            if author_name:
                                authors.append(author_name)
                    if not authors:  # Skip if no authors available
                        continue
                    
                    # Search for keywords with word boundaries
                    text_to_search = (title + " " + abstract).lower()
                    
                    # Find matching AI keywords
                    ai_matches = []
                    for keyword in AI_KEYWORDS:
                        pattern = r'\b' + re.escape(keyword.lower()) + r'\b'
                        if re.search(pattern, text_to_search):
                            ai_matches.append(keyword)
                    
                    # Find matching dementia keywords
                    dementia_matches = []
                    for keyword in DEMENTIA_KEYWORDS:
                        pattern = r'\b' + re.escape(keyword.lower()) + r'\b'
                        if re.search(pattern, text_to_search):
                            dementia_matches.append(keyword)
                    
                    # Skip if no keywords found in either category
                    if not (ai_matches and dementia_matches):
                        continue
                        
                    record_data = {
                        'id': f"pm_{pub_year}_{idx + 1}",
                        'title': title,
                        'authors': ', '.join(authors),
                        'abstract': abstract,
                        'year': pub_year,
                        'ai_term_found': '; '.join(ai_matches),
                        'dementia_term_found': '; '.join(dementia_matches)
                    }
                    records_data.append(record_data)
                    idx += 1
                    
                except KeyError as e:
                    print(f"Skipping record due to missing data: {str(e)}")
                    continue
            
            print(f"Processed {i + len(batch_ids)} of {len(id_list)} articles...")
            time.sleep(REQUEST_DELAY)
                
        return pd.DataFrame(records_data)
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return pd.DataFrame()

def main():
    try:
        # Break down the search into yearly chunks
        all_results = []

        for year in range(2015, 2026):
            start_date = f"{year}/01/01"
            end_date = f"{year}/12/31"

            print(f"\nProcessing year {year}...")
            query = build_query(
                start_date=start_date,
                end_date=end_date,
                date_field="Create"
            )

            print(f"Starting search for year {year}...")
            df_year = fetch_pubmed_data(query)

            if not df_year.empty:
                print(f"Found {len(df_year)} articles for year {year}")
                all_results.append(df_year)
            else:
                print(f"No results found for year {year}")

        # Combine all results
        if all_results:
            final_df = pd.concat(all_results, ignore_index=True)
            final_df.to_csv('PubMed_results.csv', index=False, encoding='utf-8')
            print(f"\nTotal results saved to PubMed_results.csv: {len(final_df)} articles")
        else:
            print("No results found for any year")

        print("Process completed!")

    except ValueError as e:
        print(f"Error with date parameters: {e}")
        return

if __name__ == "__main__":
    main()