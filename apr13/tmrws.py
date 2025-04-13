import requests
from bs4 import BeautifulSoup
import pandas as pd
import os
import re
import time
from datetime import datetime

def clean_text(text):
    """Clean up text by removing extra whitespace and special characters."""
    if not text:
        return ""
    return re.sub(r'\s+', ' ', text).strip()

def extract_race_info(race_text):
    """
    Extract race type, purse, distance, surface, etc. from race text.
    """
    race_info = {
        'race_type': None,
        'purse': None,
        'distance': None,
        'distance_unit': None,
        'course': None,
        'surface': 'Dirt',  # Default to dirt unless specified otherwise
        'track_condition': None,
        'post_time': None
    }
    
    # Extract post time
    post_time_match = re.search(r'POST\s+Time\s*-\s*(\d+:\d+\s*[AP]M)', race_text, re.IGNORECASE)
    if post_time_match:
        race_info['post_time'] = post_time_match.group(1)
    
    # Extract race type (MAIDEN SPECIAL WEIGHT, ALLOWANCE, STAKES, etc.)
    race_types = [
        'MAIDEN SPECIAL WEIGHT', 
        'MAIDEN CLAIMING', 
        'ALLOWANCE OPTIONAL CLAIMING',
        'ALLOWANCE', 
        'CLAIMING', 
        'STAKES'
    ]
    
    for race_type in race_types:
        if race_type in race_text:
            race_info['race_type'] = race_type
            break
    
    # Extract purse
    purse_match = re.search(r'Purse\s*\$?([\d,]+)', race_text)
    if purse_match:
        race_info['purse'] = purse_match.group(1).replace(',', '')
    
    # Extract distance
    distance_patterns = [
        r'(\d+(?:\s+\d+/\d+)?)\s+(Furlong)', 
        r'(\d+(?:\s+\d+/\d+)?)\s+(Mile)',
        r'(\d+(?:\s+\d+/\d+)?)\s+(Miles)',
        r'(\d+(?:\s+\d+/\d+)?)\s+(Yards)'
    ]
    
    for pattern in distance_patterns:
        match = re.search(pattern, race_text, re.IGNORECASE)
        if match:
            race_info['distance'] = match.group(1)
            race_info['distance_unit'] = match.group(2)
            break
    
    # Extract surface
    if '(Turf)' in race_text:
        race_info['surface'] = 'Turf'
        race_info['course'] = 'Turf'
    elif 'Dirt' in race_text:
        race_info['surface'] = 'Dirt'
        race_info['course'] = 'Dirt'
    
    return race_info

def parse_equibase_page(html_content):
    """
    Parse the Equibase HTML content to extract race and horse information.
    Returns a list of entries.
    """
    soup = BeautifulSoup(html_content, 'html.parser')
    all_entries = []
    
    # Track info
    track_code = 'KEE'
    track_name = 'Keeneland'
    race_date = '04/13/2025'
    
    # Find all race headers or sections
    race_blocks = []
    
    # Method 1: Look for race headers with "Race X"
    race_headers = soup.find_all(['h1', 'h2', 'h3', 'h4', 'div'], 
                               text=lambda t: t and re.search(r'Race\s+\d+', t, re.IGNORECASE))
    
    if race_headers:
        print(f"Found {len(race_headers)} race headers")
        
        for header in race_headers:
            race_num_match = re.search(r'Race\s+(\d+)', header.get_text(), re.IGNORECASE)
            if race_num_match:
                race_num = int(race_num_match.group(1))
                if race_num <= 9:  # Only process races 1-9
                    # Find the table or content section for this race
                    race_section = header.find_next(['table', 'div'])
                    if race_section:
                        race_blocks.append({
                            'race_num': race_num,
                            'header': header,
                            'section': race_section
                        })
    
    # Method 2: If race headers not found or incomplete, look for race sections by class
    if len(race_blocks) < 9:
        race_sections = soup.find_all(['div', 'table'], class_=lambda c: c and any(x in str(c).lower() for x in ['race-', 'race_', 'entry']))
        
        if race_sections:
            print(f"Found {len(race_sections)} race sections by class")
            
            for section in race_sections:
                # Try to find race number
                race_num = None
                header = section.find_previous(['h1', 'h2', 'h3', 'h4', 'div'], 
                                             text=lambda t: t and re.search(r'Race\s+\d+', t, re.IGNORECASE))
                
                if header:
                    race_num_match = re.search(r'Race\s+(\d+)', header.get_text(), re.IGNORECASE)
                    if race_num_match:
                        race_num = int(race_num_match.group(1))
                
                # If race number not found, try to find it in section content
                if not race_num:
                    race_num_match = re.search(r'Race\s+(\d+)', section.get_text(), re.IGNORECASE)
                    if race_num_match:
                        race_num = int(race_num_match.group(1))
                
                # If still no race number, check if section has race number in ID or class
                if not race_num:
                    section_id = section.get('id', '')
                    section_class = ' '.join(section.get('class', []))
                    
                    for attr in [section_id, section_class]:
                        match = re.search(r'race[_-]?(\d+)', attr, re.IGNORECASE)
                        if match:
                            race_num = int(match.group(1))
                            break
                
                if race_num and race_num <= 9:
                    # Check if we already have this race
                    if not any(block['race_num'] == race_num for block in race_blocks):
                        race_blocks.append({
                            'race_num': race_num,
                            'header': header,
                            'section': section
                        })
    
    # Sort race blocks by race number
    race_blocks = sorted(race_blocks, key=lambda x: x['race_num'])
    
    # Process each race block
    for block in race_blocks:
        race_num = block['race_num']
        header = block['header']
        section = block['section']
        
        print(f"Processing Race {race_num}")
        
        # Extract race information
        race_text = ""
        if header:
            race_text += header.get_text() + " "
        
        # Look for race details in nearby elements
        race_details = None
        if header:
            # Look at siblings after the header
            next_elem = header.next_sibling
            while next_elem and not (hasattr(next_elem, 'name') and next_elem.name in ['table', 'h1', 'h2', 'h3', 'h4']):
                if hasattr(next_elem, 'get_text'):
                    race_text += next_elem.get_text() + " "
                next_elem = next_elem.next_sibling
        
        race_info = extract_race_info(race_text)
        
        # Extract horses from table if present
        table = None
        if section.name == 'table':
            table = section
        else:
            table = section.find('table')
        
        if table:
            # Find header row
            header_row = None
            rows = table.find_all('tr')
            
            for row in rows:
                # Check if this is a header row
                if row.find('th') or 'Horse' in row.get_text():
                    header_row = row
                    break
            
            # Map column indices
            col_indices = {}
            if header_row:
                cols = header_row.find_all(['th', 'td'])
                for i, col in enumerate(cols):
                    col_text = col.get_text().strip().lower()
                    
                    # Map various possible header names to our column names
                    if any(word in col_text for word in ['p#', 'no.', 'program']):
                        col_indices['program_num'] = i
                    elif any(word in col_text for word in ['post', 'pp', '#']):
                        col_indices['post_position'] = i
                    elif any(word in col_text for word in ['horse', 'runner']):
                        col_indices['horse_name'] = i
                    elif 'vs' in col_text:
                        col_indices['vs'] = i
                    elif any(word in col_text for word in ['a/s', 'age', 'sex']):
                        col_indices['age_sex'] = i
                    elif any(word in col_text for word in ['med', 'medication']):
                        col_indices['medication'] = i
                    elif any(word in col_text for word in ['jockey', 'rider']):
                        col_indices['jockey'] = i
                    elif any(word in col_text for word in ['wgt', 'weight']):
                        col_indices['weight'] = i
                    elif any(word in col_text for word in ['trainer']):
                        col_indices['trainer'] = i
                    elif any(word in col_text for word in ['m/l', 'odds', 'morning']):
                        col_indices['morning_line_odds'] = i
                    elif any(word in col_text for word in ['owner']):
                        col_indices['owner'] = i
            
            # Process data rows
            for row in rows:
                # Skip header row
                if header_row and row == header_row:
                    continue
                
                cols = row.find_all(['td', 'th'])
                if len(cols) < 2:  # Need at least a couple of columns for meaningful data
                    continue
                
                # Skip rows that don't contain horse data
                row_text = row.get_text().strip().lower()
                if any(word in row_text for word in ['owners:', 'breeders:', 'pedigrees', 'equipment']):
                    continue
                
                try:
                    # Create entry with default race information
                    entry = {
                        'race_number': race_num,
                        'track_code': track_code,
                        'track_name': track_name,
                        'race_date': race_date
                    }
                    
                    # Add race information
                    entry.update(race_info)
                    
                    # Extract data based on mapped columns
                    if col_indices:
                        # Program number (P#)
                        if 'program_num' in col_indices and col_indices['program_num'] < len(cols):
                            entry['program_num'] = clean_text(cols[col_indices['program_num']].get_text())
                        
                        # Post position (PP)
                        if 'post_position' in col_indices and col_indices['post_position'] < len(cols):
                            entry['post_position'] = clean_text(cols[col_indices['post_position']].get_text())
                        
                        # Horse name
                        if 'horse_name' in col_indices and col_indices['horse_name'] < len(cols):
                            horse_text = clean_text(cols[col_indices['horse_name']].get_text())
                            entry['horse_name'] = horse_text
                            
                            # Extract breed from horse name if present
                            breed_match = re.search(r'\(([A-Z]{2,3})\)', horse_text)
                            if breed_match:
                                entry['breed'] = breed_match.group(1)
                                # Clean the horse name by removing the state code
                                entry['horse_name'] = re.sub(r'\s*\([A-Z]{2,3}\)\s*$', '', horse_text)
                        
                        # Age/Sex
                        if 'age_sex' in col_indices and col_indices['age_sex'] < len(cols):
                            age_sex_text = clean_text(cols[col_indices['age_sex']].get_text())
                            age_sex_match = re.search(r'(\d+)/([CFGMH])', age_sex_text)
                            if age_sex_match:
                                entry['age'] = age_sex_match.group(1)
                                sex_code = age_sex_match.group(2)
                                sex_map = {
                                    'C': 'Colt',
                                    'F': 'Filly',
                                    'G': 'Gelding',
                                    'M': 'Mare',
                                    'H': 'Horse'
                                }
                                entry['sex'] = sex_map.get(sex_code, sex_code)
                        
                        # Medication
                        if 'medication' in col_indices and col_indices['medication'] < len(cols):
                            entry['medication'] = clean_text(cols[col_indices['medication']].get_text())
                        
                        # Jockey
                        if 'jockey' in col_indices and col_indices['jockey'] < len(cols):
                            entry['jockey'] = clean_text(cols[col_indices['jockey']].get_text())
                        
                        # Weight
                        if 'weight' in col_indices and col_indices['weight'] < len(cols):
                            entry['weight'] = clean_text(cols[col_indices['weight']].get_text())
                        
                        # Trainer
                        if 'trainer' in col_indices and col_indices['trainer'] < len(cols):
                            entry['trainer'] = clean_text(cols[col_indices['trainer']].get_text())
                        
                        # Morning line odds
                        if 'morning_line_odds' in col_indices and col_indices['morning_line_odds'] < len(cols):
                            entry['morning_line_odds'] = clean_text(cols[col_indices['morning_line_odds']].get_text())
                        
                        # Owner (if available)
                        if 'owner' in col_indices and col_indices['owner'] < len(cols):
                            entry['owner'] = clean_text(cols[col_indices['owner']].get_text())
                    else:
                        # Default column mapping if headers not identified
                        program_num = clean_text(cols[0].get_text()) if len(cols) > 0 else ""
                        post_position = clean_text(cols[1].get_text()) if len(cols) > 1 else ""
                        horse_name = clean_text(cols[2].get_text()) if len(cols) > 2 else ""
                        
                        if program_num and post_position and horse_name:
                            entry['program_num'] = program_num
                            entry['post_position'] = post_position
                            entry['horse_name'] = horse_name
                            
                            # Try to extract remaining fields
                            if len(cols) > 3:
                                entry['jockey'] = clean_text(cols[3].get_text())
                            if len(cols) > 4:
                                entry['trainer'] = clean_text(cols[4].get_text())
                    
                    # If no post_position but we have program_num, use that
                    if ('post_position' not in entry or not entry['post_position']) and 'program_num' in entry:
                        entry['post_position'] = entry['program_num']
                    
                    # Check if this is a scratch
                    is_scratched = False
                    row_text = row.get_text().lower()
                    if 'scratched' in row_text or 'scr' in row_text:
                        is_scratched = True
                        entry['post_position'] = 'SCR'
                        entry['finish'] = 'SCR'
                    
                    # Only add entries with valid horse names
                    if ('horse_name' in entry and entry['horse_name'] and 
                        not entry['horse_name'].lower() in ['horse', 'program', 'horse vs a/s']):
                        
                        # Set null values for missing fields
                        required_fields = [
                            'race_type', 'purse', 'distance', 'distance_unit', 'course',
                            'surface', 'track_condition', 'weather', 'post_time', 'win_time',
                            'breed', 'age', 'sex', 'medication', 'program_num', 'finish',
                            'comment', 'jockey', 'trainer', 'owner', 'last_race_track',
                            'last_race_date', 'last_race_number', 'last_race_finish',
                            'dollar_odds', 'num_past_starts', 'num_past_wins',
                            'num_past_seconds', 'num_past_thirds'
                        ]
                        
                        for field in required_fields:
                            if field not in entry or not entry[field]:
                                entry[field] = None
                        
                        all_entries.append(entry)
                        print(f"Added entry: Race {race_num}, PP {entry.get('post_position', 'N/A')} - {entry.get('horse_name', 'N/A')}")
                
                except Exception as e:
                    print(f"Error processing row: {e}")
        
        # If no horses found, try alternate methods
        if not any(entry['race_number'] == race_num for entry in all_entries):
            print(f"No horses found in table for Race {race_num}, trying alternate methods...")
            
            # Look for horse information in non-table format
            horse_divs = section.find_all(['div', 'li', 'p'], class_=lambda c: c and any(word in str(c).lower() for word in ['horse', 'entry']))
            
            if horse_divs:
                for div in horse_divs:
                    try:
                        text = div.get_text()
                        
                        # Skip non-horse divs
                        if not text or len(text) < 5:
                            continue
                        
                        # Try to extract horse info from text
                        match = re.search(r'([0-9]+)\s*([A-Za-z\s\'\-\.]+)(?:\(([A-Z]{2,3})\))?\s*([0-9]/[CFGMH])?', text)
                        
                        if match:
                            pp = match.group(1)
                            horse_name = match.group(2).strip()
                            breed = match.group(3) if match.group(3) else None
                            age_sex = match.group(4) if match.group(4) else None
                            
                            entry = {
                                'race_number': race_num,
                                'post_position': pp,
                                'horse_name': horse_name,
                                'breed': breed,
                                'track_code': track_code,
                                'track_name': track_name,
                                'race_date': race_date
                            }
                            
                            # Extract age/sex if available
                            if age_sex:
                                age_sex_match = re.search(r'(\d+)/([CFGMH])', age_sex)
                                if age_sex_match:
                                    entry['age'] = age_sex_match.group(1)
                                    sex_code = age_sex_match.group(2)
                                    sex_map = {
                                        'C': 'Colt',
                                        'F': 'Filly',
                                        'G': 'Gelding',
                                        'M': 'Mare',
                                        'H': 'Horse'
                                    }
                                    entry['sex'] = sex_map.get(sex_code, sex_code)
                            
                            # Add race information
                            entry.update(race_info)
                            
                            # Set null values for missing fields
                            required_fields = [
                                'race_type', 'purse', 'distance', 'distance_unit', 'course',
                                'surface', 'track_condition', 'weather', 'post_time', 'win_time',
                                'weight', 'medication', 'program_num', 'finish', 'comment',
                                'jockey', 'trainer', 'owner', 'last_race_track', 'last_race_date',
                                'last_race_number', 'last_race_finish', 'dollar_odds',
                                'num_past_starts', 'num_past_wins', 'num_past_seconds',
                                'num_past_thirds'
                            ]
                            
                            for field in required_fields:
                                if field not in entry or not entry[field]:
                                    entry[field] = None
                            
                            all_entries.append(entry)
                            print(f"Added entry from div: Race {race_num}, PP {entry.get('post_position', 'N/A')} - {entry.get('horse_name', 'N/A')}")
                    
                    except Exception as e:
                        print(f"Error processing horse div: {e}")
    
    # Special fallback handling if we still don't have all 9 races
    races_found = set(entry['race_number'] for entry in all_entries)
    missing_races = [i for i in range(1, 10) if i not in races_found]
    
    if missing_races:
        print(f"Missing races: {missing_races}. Trying more generic extraction...")
        
        # Try to find any tables that might contain race data
        tables = soup.find_all('table')
        
        for table_idx, table in enumerate(tables):
            table_text = table.get_text()
            
            # Skip tables we've already processed
            if any(str(race_num) in table_text and re.search(rf'Race\s+{race_num}', table_text) for race_num in races_found):
                continue
            
            # Check if this table might be for a missing race
            for race_num in missing_races:
                if str(race_num) in table_text and re.search(rf'Race\s+{race_num}', table_text):
                    print(f"Found potential table for Race {race_num}")
                    
                    # Find race info
                    race_header = table.find_previous(['h1', 'h2', 'h3', 'h4', 'div'], 
                                                    text=lambda t: t and re.search(rf'Race\s+{race_num}', t, re.IGNORECASE))
                    
                    race_text = ""
                    if race_header:
                        race_text = race_header.get_text()
                    else:
                        # Look for race info in the table itself
                        race_text = table_text
                    
                    race_info = extract_race_info(race_text)
                    
                    # Process horses
                    rows = table.find_all('tr')
                    
                    for row in rows:
                        try:
                            cols = row.find_all(['td', 'th'])
                            
                            if len(cols) >= 3:  # Need at least a few columns for horse data
                                # Skip header rows
                                if cols[0].name == 'th' or 'horse' in cols[0].get_text().lower():
                                    continue
                                
                                # Try to extract horse data
                                pp = clean_text(cols[0].get_text()) if len(cols) > 0 else ""
                                horse = clean_text(cols[1].get_text()) if len(cols) > 1 else ""
                                
                                # Skip non-horse rows
                                if not horse or horse.lower() in ['horse', 'program', 'horse vs a/s']:
                                    continue
                                
                                # Extract breed if present
                                breed_match = re.search(r'\(([A-Z]{2,3})\)', horse)
                                breed = breed_match.group(1) if breed_match else None
                                
                                # Clean horse name
                                horse_name = re.sub(r'\s*\([A-Z]{2,3}\)\s*$', '', horse)
                                
                                # Create entry
                                entry = {
                                    'race_number': race_num,
                                    'post_position': pp,
                                    'horse_name': horse_name,
                                    'breed': breed,
                                    'track_code': track_code,
                                    'track_name': track_name,
                                    'race_date': race_date
                                }
                                
                                # Add additional fields if available
                                if len(cols) > 2:
                                    entry['jockey'] = clean_text(cols[2].get_text())
                                if len(cols) > 3:
                                    entry['trainer'] = clean_text(cols[3].get_text())
                                if len(cols) > 4:
                                    entry['weight'] = clean_text(cols[4].get_text())
                                if len(cols) > 5:
                                    entry['medication'] = clean_text(cols[5].get_text())
                                if len(cols) > 6:
                                    entry['morning_line_odds'] = clean_text(cols[6].get_text())
                                
                                # Add race information
                                entry.update(race_info)
                                
                                # Set null values for missing fields
                                required_fields = [
                                    'race_type', 'purse', 'distance', 'distance_unit', 'course',
                                    'surface', 'track_condition', 'weather', 'post_time', 'win_time',
                                    'age', 'sex', 'medication', 'program_num', 'finish', 'comment',
                                    'jockey', 'trainer', 'owner', 'last_race_track', 'last_race_date',
                                    'last_race_number', 'last_race_finish', 'dollar_odds',
                                    'num_past_starts', 'num_past_wins', 'num_past_seconds',
                                    'num_past_thirds'
                                ]
                                
                                for field in required_fields:
                                    if field not in entry or not entry[field]:
                                        entry[field] = None
                                
                                all_entries.append(entry)
                                print(f"Added entry from generic table: Race {race_num}, PP {pp} - {horse_name}")
                        
                        except Exception as e:
                            print(f"Error processing row in generic table: {e}")
    
    return all_entries

def scrape_equibase_entries(date_str=None, max_races=9):
    """
    Scrapes horse racing entries from Equibase for Keeneland.
    
    Args:
        date_str: Date string in format MMDDYY. If None, uses 041325 for April 13, 2025.
        max_races: Maximum number of races to scrape (default 9)
    
    Returns:
        DataFrame of race entries if successful, None otherwise.
    """
    # If no date provided, use April 13, 2025
    if not date_str:
        date_str = "041325"
    
    # URLs to try
    urls = [
        f'https://www.equibase.com/static/entry/KEE{date_str}USA-EQB.html',
        f'https://www.equibase.com/static/entry/KEE{date_str}-entries.html',
        'https://www.equibase.com/static/entry/KEE-entries.html'
    ]
    
    # Add headers to mimic a browser request
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.0.0 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.9',
        'Referer': 'https://www.keeneland.com/',
    }
    
    all_entries = []
    
    # Try each URL in order
    for url in urls:
        print(f"Trying URL: {url}")
        try:
            # Make the request with headers
            response = requests.get(url, headers=headers, timeout=30)
            
            # Check if the request was successful
            if response.status_code != 200:
                print(f"Failed to retrieve page: Status code {response.status_code}")
                continue
            
            # Save HTML for inspection
            with open('equibase_response.html', 'w', encoding='utf-8') as f:
                f.write(response.text)
                print(f"Saved HTML content to 'equibase_response.html' for inspection")
            
            # Parse the page
            entries = parse_equibase_page(response.text)
            
            if entries:
                all_entries = entries
                print(f"Successfully scraped {len(entries)} entries from {url}")
                break
            
        except Exception as e:
            print(f"Error trying URL {url}: {e}")
            continue
    
    # If we found any entries, create a DataFrame with all required columns
    if all_entries:
        # Define all required columns in specified order
        all_columns = [
            'race_number', 'race_type', 'purse', 'distance', 'distance_unit', 
            'course', 'surface', 'track_condition', 'weather', 'post_time', 
            'win_time', 'horse_name', 'breed', 'weight', 'age', 'sex', 
            'medication', 'program_num', 'post_position', 'finish', 'comment', 
            'jockey', 'trainer', 'owner', 'last_race_track', 'last_race_date', 
            'last_race_number', 'last_race_finish', 'track_code', 'track_name', 
            'race_date', 'dollar_odds', 'num_past_starts', 'num_past_wins', 
            'num_past_seconds','num_past_thirds'
        ]
        
        # Create DataFrame
        df_entries = pd.DataFrame(all_entries)
        
        # Ensure all required columns exist
        for col in all_columns:
            if col not in df_entries.columns:
                df_entries[col] = None
        
        # Reorder columns to match required format
        df_entries = df_entries[all_columns]
        
        # Filter to only include races 1-9
        df_entries = df_entries[df_entries['race_number'] <= max_races]
        
        # Sort by race number and post position
        df_entries['sort_pp'] = df_entries['post_position'].apply(
            lambda x: int(x) if isinstance(x, str) and x.isdigit() else float('inf')
        )
        df_entries = df_entries.sort_values(['race_number', 'sort_pp'])
        df_entries = df_entries.drop('sort_pp', axis=1)
        
        return df_entries
    else:
        print("No entries found in any of the tried URLs")
        return None

def main():
    """Main function to execute the script."""
    # Scrape entries for April 13, 2025, limiting to first 9 races
    date_str = "041325"  # Format: MMDDYY
    entries_df = scrape_equibase_entries(date_str, max_races=9)
    
    # Show and save the scraped data
    if entries_df is not None and len(entries_df) > 0:
        print("\nScraped Data Summary:")
        print(f"Total entries: {len(entries_df)}")
        print(f"Races: {entries_df['race_number'].nunique()}")
        print("\nSample of data:")
        print(entries_df.head())
        
        # Check which races we have
        races_found = sorted(entries_df['race_number'].unique())
        print(f"Races found: {races_found}")
        
        missing_races = [i for i in range(1, 10) if i not in races_found]
        if missing_races:
            print(f"Missing races: {missing_races}")
        
        # Save to CSV
        output_file = 'keeneland_entries.csv'
        entries_df.to_csv(output_file, index=False)
        print(f"\nData saved to '{output_file}'")
    else:
        print("\nFailed to retrieve any race entries.")
        print("\nTry accessing the Equibase site directly in your browser:")
        print("https://www.equibase.com/static/entry/KEE041325USA-EQB.html")

if __name__ == "__main__":
    main()