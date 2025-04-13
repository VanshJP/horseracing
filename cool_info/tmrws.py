import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
from datetime import datetime

def scrape_keeneland_entries(url):
    # Add headers to mimic a browser request
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5',
        'Connection': 'keep-alive',
        'Upgrade-Insecure-Requests': '1',
        'Cache-Control': 'max-age=0'
    }
    
    # Make the request with headers
    response = requests.get(url, headers=headers)
    
    # Check if the request was successful
    if response.status_code != 200:
        print(f"Failed to retrieve page: Status code {response.status_code}")
        return None
    
    # Parse the HTML content
    soup = BeautifulSoup(response.content, 'html.parser')
    
    # First, let's find all race tables - the actual class may have changed
    # Look for any tables that could contain race entries
    all_tables = soup.find_all('table')
    
    # Debug info
    print(f"Found {len(all_tables)} tables on the page")
    
    # First attempt to find race entries tables with common attributes
    races = soup.find_all(['table', 'div'], class_=lambda c: c and ('race' in c.lower() or 'entries' in c.lower()))
    
    # If no tables found with those classes, try a more general approach
    if not races:
        # Look for divs or sections that might contain race information
        races = soup.find_all(['div', 'section'], id=lambda i: i and ('race' in i.lower() or 'entries' in i.lower()))
    
    # Debug info
    print(f"Found {len(races)} potential race entry containers")
    
    # If still no tables found, print the page structure for analysis
    if not races and len(all_tables) < 3:
        print("No race entry tables found. Page structure may have changed.")
        # Save the HTML for inspection
        with open('keeneland_page.html', 'w', encoding='utf-8') as f:
            f.write(response.text)
        print("HTML saved to 'keeneland_page.html' for inspection")
        return None
    
    all_entries = []
    
    # If we found potential race containers, try to parse them
    for race_num, race in enumerate(races, start=1):
        print(f"Processing race {race_num}")
        
        # Try to find rows in this race container
        rows = race.find_all('tr')
        
        # Skip the header rows if they exist
        if len(rows) > 2:
            rows = rows[2:]
        
        for row in rows:
            cols = row.find_all('td')
            
            # Debug
            if cols:
                print(f"Row has {len(cols)} columns")
            
            # Skip rows with insufficient columns
            if len(cols) < 5:
                continue
            
            try:
                # Adjust column indices based on the actual HTML structure
                # These may need to be updated after examining the HTML
                post_position = cols[0].text.strip() if len(cols) > 0 else "Unknown"
                horse_name = cols[1].text.strip() if len(cols) > 1 else "Unknown"
                
                # Collect all available data (adjust indices as needed)
                entry_data = {
                    'race_number': race_num,
                    'post_position': post_position,
                    'horse_name': horse_name
                }
                
                # Add other columns if they exist
                if len(cols) > 2:
                    entry_data['jockey'] = cols[2].text.strip()
                if len(cols) > 3:
                    entry_data['trainer'] = cols[3].text.strip()
                if len(cols) > 4:
                    entry_data['weight'] = cols[4].text.strip()
                if len(cols) > 5:
                    entry_data['medication'] = cols[5].text.strip()
                if len(cols) > 6:
                    entry_data['morning_line_odds'] = cols[6].text.strip()
                
                all_entries.append(entry_data)
                print(f"Added entry: {post_position} - {horse_name}")
                
            except Exception as e:
                print(f"Error processing row: {e}")
                continue
    
    # If we found any entries, create a DataFrame
    if all_entries:
        df_entries = pd.DataFrame(all_entries)
        return df_entries
    else:
        print("No entries found in the tables")
        return None

# Try with current date instead of future date
today = datetime.now().strftime("%m/%d/%Y")
url = f'https://keeneland.equibase.com/eqbRaceEntriesDisplay.cfm?TRK=KEE&CY=USA&DATE={today}&STYLE=KEE'
print(f"Trying URL: {url}")

entries_df = scrape_keeneland_entries(url)

# If that fails, try alternative URLs
if entries_df is None:
    alternative_url = 'https://www.equibase.com/static/entry/KEE-entries.html'
    print(f"First attempt failed. Trying alternative URL: {alternative_url}")
    entries_df = scrape_keeneland_entries(alternative_url)

# Show the scraped data
if entries_df is not None:
    print(entries_df)
    # Save to CSV
    entries_df.to_csv('apr13/keeneland_entries.csv', index=False)
    print("Data saved to 'keeneland_entries.csv'")
else:
    print("Failed to retrieve race entries.")