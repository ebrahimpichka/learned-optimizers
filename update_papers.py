import bibtexparser
from bibtexparser.bwriter import BibTexWriter
from bibtexparser.bibdatabase import BibDatabase
import requests
import time
import os

# Configuration
BIB_FILE = 'papers.bib'
README_FILE = 'README.md'

def fetch_paper_details(title):
    """
    Fetches paper details from Semantic Scholar API.
    """
    try:
        # Search for the paper by title
        print(f"Searching for: {title}")
        search_url = "https://api.semanticscholar.org/graph/v1/paper/search"
        params = {
            "query": title,
            "limit": 1,
            "fields": "title,abstract,url,venue,year,authors,externalIds"
        }
        response = requests.get(search_url, params=params)
        
        if response.status_code == 200:
            data = response.json()
            if data['data']:
                paper = data['data'][0]
                return paper
        elif response.status_code == 429:
             print("Rate limit hit. Waiting...")
             time.sleep(5)
             return fetch_paper_details(title) # Retry once
        else:
            print(f"Error searching for paper: {response.status_code}")
            
    except Exception as e:
        print(f"Exception fetching details: {e}")
    return None

def update_bib_and_readme():
    if not os.path.exists(BIB_FILE):
        print(f"{BIB_FILE} not found!")
        return

    with open(BIB_FILE, 'r', encoding='utf-8') as bibtex_file:
        bib_database = bibtexparser.load(bibtex_file)

    entries = bib_database.entries
    readme_content = "# Learned Optimizers Literature\n\nThis is a categorized list of papers on learned optimizers.\n\n"

    # Group by year for the README? Or just a list? The user said "categorize and hold".
    # For now, let's just list them sorted by year (descending).
    
    # Sort entries by year, handling missing years
    entries.sort(key=lambda x: x.get('year', '0'), reverse=True)

    updated_count = 0

    for entry in entries:
        title = entry.get('title', '').replace('{', '').replace('}', '').replace('\n', ' ')
        
        # Check if abstract or url is missing
        needs_update = 'abstract' not in entry or 'url' not in entry
        
        paper_data = None
        if needs_update and title:
            # Add a small delay to be nice to the API
            time.sleep(1) 
            paper_data = fetch_paper_details(title)
            
            if paper_data:
                updated_count += 1
                if 'abstract' not in entry and paper_data.get('abstract'):
                    entry['abstract'] = paper_data['abstract']
                
                if 'url' not in entry and paper_data.get('url'):
                    entry['url'] = paper_data['url']
                
                # Try to get ArXiv ID if available for a better link
                if 'externalIds' in paper_data and 'ArXiv' in paper_data['externalIds']:
                     arxiv_id = paper_data['externalIds']['ArXiv']
                     entry['eprint'] = arxiv_id
                     entry['archivePrefix'] = 'arXiv'
                     entry['primaryClass'] = 'cs.LG' # Default assumption, or could fetch
                     if 'url' not in entry:
                         entry['url'] = f"https://arxiv.org/abs/{arxiv_id}"

        # Generate README entry
        paper_title = entry.get('title', 'Untitled').replace('{', '').replace('}', '')
        paper_url = entry.get('url', '#')
        paper_authors = entry.get('author', 'Unknown Authors').replace('\n', ' ')
        paper_year = entry.get('year', 'Unknown Year')
        paper_abstract = entry.get('abstract', 'No abstract available.')

        readme_content += f"### [{paper_title}]({paper_url})\n"
        readme_content += f"**Authors:** {paper_authors} ({paper_year})\n\n"
        readme_content += "<details>\n<summary>Abstract</summary>\n\n"
        readme_content += f"{paper_abstract}\n"
        readme_content += "\n</details>\n\n---\n\n"

    # Save updated BibTeX
    if updated_count > 0:
        print(f"Updating {BIB_FILE} with {updated_count} new details...")
        writer = BibTexWriter()
        writer.indent = '    ' # 4 spaces indentation
        with open(BIB_FILE, 'w', encoding='utf-8') as bibtex_file:
            bibtexparser.dump(bib_database, bibtex_file)
    else:
        print("No new details found preventing write to bib file.")

    # Save README
    with open(README_FILE, 'w', encoding='utf-8') as f:
        f.write(readme_content)
    print(f"Generated {README_FILE}")

if __name__ == "__main__":
    update_bib_and_readme()
