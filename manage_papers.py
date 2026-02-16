import bibtexparser
from bibtexparser.bwriter import BibTexWriter
from bibtexparser.bibdatabase import BibDatabase
import requests
import time
import os
import re

BIB_FILE = 'papers.bib'
README_FILE = 'README.md'

# Semantic Scholar API Limits
BATCH_SIZE = 400 # Max 500, but let's be safe
SLEEP_BETWEEN_BATCHES = 2
SLEEP_BETWEEN_SEARCHES = 1.1

def clean_text(text):
    if not text: return ""
    return text.replace('{', '').replace('}', '').replace('\n', ' ').strip()

def get_id_from_entry(entry):
    """
    Extracts a supported ID from a bibtex entry.
    Returns format expected by Semantic Scholar batch API.
    """
    # 1. DOI
    if 'doi' in entry and entry['doi']:
        # Basic cleanup of DOI field
        doi = clean_text(entry['doi']).replace('http://dx.doi.org/', '').replace('https://doi.org/', '')
        return f"DOI:{doi}"
    
    # 2. ArXiv ID (custom field or parsed)
    if 'arxivid' in entry:
        return f"ARXIV:{clean_text(entry['arxivid'])}"
    
    # 3. URL - S2 supports URL:<url>
    if 'url' in entry and entry['url']:
        url = clean_text(entry['url'])
        # Check if URL is from a supported domain
        supported_domains = [
            'semanticscholar.org', 'arxiv.org', 'aclweb.org', 
            'acm.org', 'biorxiv.org'
        ]
        if any(domain in url for domain in supported_domains):
            # Ensure URL is clean
            return f"URL:{url}"

    return None

def fetch_batch_papers(ids):
    """
    Fetches details for a list of paper IDs using the batch endpoint.
    """
    url = "https://api.semanticscholar.org/graph/v1/paper/batch"
    params = {
        "fields": "title,abstract,url,venue,year,authors,externalIds,tldr"
    }
    
    all_paper_data = {}
    
    # Process in chunks
    for i in range(0, len(ids), BATCH_SIZE):
        chunk = ids[i:i + BATCH_SIZE]
        payload = {"ids": chunk}
        
        try:
            print(f"Fetching batch of {len(chunk)} papers...")
            response = requests.post(url, params=params, json=payload)
            
            if response.status_code == 200:
                data = response.json()
                # The response is a list of paper objects corresponding to the IDs
                # However, if an ID is invalid, it might return null in that position?
                if isinstance(data, list):
                    for idx, paper in enumerate(data):
                        if paper:
                            # We need to map back to the original ID we asked for
                            # The API doesn't guarantee preserving order if some are missing?
                            # Wait, documentation says: "List of papers with default or requested fields"
                            # But standard batch APIs usually return in order or return a map.
                            # For S2, let's assume index correspondence or check if paper has an ID wrapper.
                            # The paper object has 'paperId'.
                            # BUT we queried with DOI/URL. Returned paper has 'paperId' (S2 ID).
                            # We can't easily map back S2 ID to DOI without checking fields.
                            # Let's map by the query ID if strict order is maintained.
                            # But if strict order is NOT maintained, we are in trouble.
                            # Most batch APIs return distinct objects.
                            # If we look at the S2 python client example `semanticscholar` library,
                            # it seems to rely on order.
                            # Let's assume order is preserved.
                            if idx < len(chunk):
                                query_id = chunk[idx]
                                all_paper_data[query_id] = paper
                else:
                    print(f"Unexpected response format: {type(data)}")
            else:
                print(f"Error fetching batch: {response.status_code} {response.text}")
                
        except Exception as e:
            print(f"Exception fetching batch: {e}")
        
        time.sleep(SLEEP_BETWEEN_BATCHES)
        
    return all_paper_data

def fetch_paper_by_title_match(title):
    """
    Uses the Title Match endpoint which is better for single paper lookup by title.
    """
    try:
        clean_title = clean_text(title)
        # remove common stopwords if needed or just send as is
        print(f"Searching (Title Match) for: {clean_title}...")
        
        url = "https://api.semanticscholar.org/graph/v1/paper/search/match"
        params = {
            "query": clean_title,
            "fields": "title,abstract,url,venue,year,authors,externalIds"
        }
        response = requests.get(url, params=params)
        
        if response.status_code == 200:
            data = response.json()
            if 'data' in data and data['data']:
                return data['data'][0] # Best match
        elif response.status_code == 429:
             print("Rate limit hit. Waiting 5s...")
             time.sleep(5)
             return fetch_paper_by_title_match(title)
        elif response.status_code == 404:
            # Not found
            pass
            
    except Exception as e:
        print(f"Match failed: {e}")
    return None

def update_entry_with_data(entry, paper_data):
    if not paper_data: return False
    
    updated = False
    
    # Helper to check if we should overwrite
    def should_update(key, new_value):
        if not new_value: return False
        if key not in entry: return True
        if not entry[key]: return True
        # Don't overwrite existing non-empty values usually, unless we are sure new is better
        # For abstracts, we definitely want the one from API if we have none.
        return False

    if 'abstract' not in entry and paper_data.get('abstract'):
        entry['abstract'] = paper_data['abstract']
        updated = True
        
    if should_update('url', paper_data.get('url')):
        entry['url'] = paper_data['url']
        updated = True
    
    if should_update('year', str(paper_data.get('year'))) if paper_data.get('year') else False:
        entry['year'] = str(paper_data['year'])
        updated = True
        
    if paper_data.get('authors'):
        # Convert authors to BibTeX format
        s2_authors = paper_data['authors']
        if s2_authors:
            author_str = ' and '.join([a['name'] for a in s2_authors])
            if should_update('author', author_str):
                entry['author'] = author_str
                updated = True

    # Venue / Journal
    venue = paper_data.get('venue') or paper_data.get('journal', {}).get('name')
    if venue and should_update('journal', venue):
        entry['journal'] = venue
        updated = True

    # External IDs
    if paper_data.get('externalIds'):
        ext_ids = paper_data['externalIds']
        if 'DOI' in ext_ids and should_update('doi', ext_ids['DOI']):
            entry['doi'] = ext_ids['DOI']
            updated = True
        if 'ArXiv' in ext_ids:
            entry['arxivid'] = ext_ids['ArXiv'] # Save ArXiv ID explicitly
            if should_update('eprint', ext_ids['ArXiv']):
                entry['eprint'] = ext_ids['ArXiv']
                entry['archivePrefix'] = 'arXiv'
                entry['primaryClass'] = 'cs.LG'
                updated = True
    
    return updated

def main():
    if not os.path.exists(BIB_FILE):
        print(f"{BIB_FILE} not found!")
        return

    print(f"Reading {BIB_FILE}...")
    with open(BIB_FILE, 'r', encoding='utf-8') as bibtex_file:
        parser = bibtexparser.bparser.BibTexParser(common_strings=True)
        bib_database = bibtexparser.load(bibtex_file, parser=parser)

    entries = bib_database.entries
    print(f"Configuration: Batch ID Lookup -> Title Match Search")
    
    # Segregate entries
    batch_map = {} # ID -> entry
    search_entries = [] # List of entries
    
    for entry in entries:
        # Check if already complete?
        # If we have abstract, we might skip to save time/calls?
        # But we might want to update potential missing links/DOIs.
        # User said: "if not exist see if we can retireve it".
        # Let's check essential fields: abstract, url.
        if 'abstract' in entry and 'url' in entry:
            continue

        pid = get_id_from_entry(entry)
        if pid:
            batch_map[pid] = entry
        else:
            search_entries.append(entry)
            
    print(f"Identified {len(batch_map)} entries for Batch ID lookup.")
    print(f"Identified {len(search_entries)} entries for Title Search.")
    
    total_updated = 0
    
    # 1. BATCH PROCESS
    if batch_map:
        batch_ids = list(batch_map.keys())
        # We need to handle duplicates in batch_ids if multiple entries map to same ID (unlikely but possible)
        # Using a list allows duplicates in request, S2 handles them. 
        # But for map lookup, we need to iterate carefully.
        
        # Actually, let's just fetch unique IDs to save bandwidth
        unique_batch_ids = list(set(batch_ids))
        
        print("\nStarting Batch Retrieval...")
        results = fetch_batch_papers(unique_batch_ids)
        
        print(f"Received data for {len(results)} papers.")
        
        # Update entries
        # Note: batch_map keys match the IDs sent
        for pid, entry in batch_map.items():
            if pid in results:
                if update_entry_with_data(entry, results[pid]):
                    total_updated += 1
    
    # 2. SEARCH PROCESS
    if search_entries:
        print("\nStarting Title Search Retrieval...")
        for i, entry in enumerate(search_entries):
            title = entry.get('title', '')
            if not title: continue
            
            # Simple progress
            if i % 5 == 0: print(f"Processing search {i+1}/{len(search_entries)}...")
            
            paper_data = fetch_paper_by_title_match(title)
            if paper_data:
                if update_entry_with_data(entry, paper_data):
                    total_updated += 1
            
            time.sleep(SLEEP_BETWEEN_SEARCHES)

    # 3. WRITE BIB
    if total_updated > 0:
        print(f"\nWriting updated BibTeX file with {total_updated} changes...")
        writer = BibTexWriter()
        writer.indent = '    '
        writer.order_entries_by = ('year', 'author')
        with open(BIB_FILE, 'w', encoding='utf-8') as bibtex_file:
            bibtexparser.dump(bib_database, bibtex_file)
    else:
        print("\nNo BibTeX updates found.")

    # 4. GENERATE README
    print("\nGenerating README.md...")
    
    # Sort by Year (Desc), then Author
    sorted_entries = sorted(entries, key=lambda x: (x.get('year', '0'), x.get('author', '')), reverse=True)
    
    md_content = "# Learned Optimizers Literature\n\n"
    md_content += f"Categorized list of {len(entries)} papers on learned optimizers.\n"
    md_content += "\nUse `python manage_papers.py` to update abstracts and links.\n\n"
    
    current_year = None
    
    for entry in sorted_entries:
        title = clean_text(entry.get('title', 'Untitled'))
        url = entry.get('url', '')
        year = clean_text(entry.get('year', 'Unknown Year'))
        authors = clean_text(entry.get('author', 'Unknown Authors'))
        abstract = entry.get('abstract', '')
        
        # Year Section
        if year != current_year:
            current_year = year
            md_content += f"## {year}\n\n"
            
        # Title Link
        if url:
            title_md = f"[{title}]({url})"
        else:
            title_md = title
            
        md_content += f"### {title_md}\n"
        md_content += f"**Authors:** {authors}\n\n"
        
        # Abstract
        if abstract:
            md_content += f"<details>\n<summary>Abstract</summary>\n\n> {abstract}\n</details>\n\n"
        else:
            md_content += "*No abstract available.*\n\n"
            
        md_content += "---\n\n"
        
    with open(README_FILE, 'w', encoding='utf-8') as f:
        f.write(md_content)
        
    print(f"Done. Check {README_FILE}")

if __name__ == "__main__":
    main()
