import requests
from bs4 import BeautifulSoup
import os
import ssl
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Create a custom SSL context
ssl_context = ssl.create_default_context()
ssl_context.set_ciphers("DEFAULT:@SECLEVEL=1")

http = urllib3.PoolManager(ssl_context=ssl_context)

# Base URL of the Shakespeare MIT site
base_url = "https://shakespeare.mit.edu/"

# Output folder for saving works
output_folder = "shakespeare_works"
os.makedirs(output_folder, exist_ok=True)

# Create category folders (if they don't exist already)
categories = ['comedy', 'history', 'tragedy', 'poetry']
for category in categories:
    os.makedirs(os.path.join(output_folder, category), exist_ok=True)

# Function to extract links to the subpages of works
def get_work_links(main_page_url):
    response = http.request("GET", main_page_url, headers={"User-Agent": "Mozilla/5.0"})
    soup = BeautifulSoup(response.data.decode('utf-8'), 'html.parser')
    
    work_links = {
        'comedy': [],
        'history': [],
        'tragedy': [],
        'poetry': []
    }

    # Locate the table containing categories
    table = soup.find("table", {"border": "", "cellpadding": "5", "align": "center"})
    
    if not table:
        print("Table not found on the page!")
        return work_links  # Return empty dict if table is not found

    # Debug: Print the raw HTML of the table to understand its structure
    print("Table HTML structure:", table.prettify()[:500])  # Print first 500 chars for debugging

    # Try to find the rows
    rows = table.find_all("tr")
    
    if len(rows) < 2:  # Ensure we have at least 2 rows (the header and one data row)
        print(f"Unexpected table structure, rows found: {len(rows)}")
        return work_links  # Return empty dict if the table structure is not as expected

    # Debugging: Check the number of rows
    print(f"Found {len(rows)} rows in the table.")

    columns = rows[1].find_all("td")
    
    # Ensure we have 4 columns (one for each category)
    if len(columns) < 4:
        print(f"Unexpected number of columns in row: {len(columns)}")
        return work_links  # Return empty dict if column count is unexpected

    # Assigning links to respective categories
    comedy_col = columns[0].find_all("a")
    history_col = columns[1].find_all("a")
    tragedy_col = columns[2].find_all("a")
    poetry_col = columns[3].find_all("a")

    for link in comedy_col:
        work_links['comedy'].append(base_url + link['href'])
    for link in history_col:
        work_links['history'].append(base_url + link['href'])
    for link in tragedy_col:
        work_links['tragedy'].append(base_url + link['href'])
    for link in poetry_col:
        work_links['poetry'].append(base_url + link['href'])
    
    return work_links

# Function to get the "full.html" link from a subpage
def get_full_page_link(subpage_url):
    response = http.request("GET", subpage_url, headers={"User-Agent": "Mozilla/5.0"})
    soup = BeautifulSoup(response.data.decode('utf-8'), 'html.parser')
    full_page_link = soup.find("a", href=True, text="Entire play")
    if full_page_link:
        return subpage_url.rsplit("/", 1)[0] + "/" + full_page_link['href']
    return None

# Function to scrape and save the full text of a play or poem
def scrape_and_save_work(work_url, title, category):
    response = http.request("GET", work_url, headers={"User-Agent": "Mozilla/5.0"})
    soup = BeautifulSoup(response.data.decode('utf-8'), 'html.parser')

    # Extract content from h3, blockquote, and a tags
    content = []

    for tag in soup.find_all(["h3", "blockquote", "a"]):
        if tag.name == "h3":
            content.append(f"### {tag.get_text(strip=True)}")  # Format h3 with ###
        #elif tag.name == "blockquote":
            #content.append(tag.get_text(separator="\n").strip())  # Keep blockquote as-is
        elif tag.name == "a" and tag.has_attr("name"):  # Filter <a> tags with 'name' attribute
            if tag.find("b"):  # If the <a> tag has a <b> child, it's likely a speaker name
                content.append(f"**{tag.b.get_text(strip=True)}**")
            else:
                content.append(tag.get_text(strip=True))  # General <a> tag text

    # Join content into a single string with double newlines for separation
    text_content = "\n\n".join(content)

    # Save to the appropriate category folder
    filename = os.path.join(output_folder, category, f"{title}.txt")
    with open(filename, "w", encoding="utf-8") as file:
        file.write(text_content)
    print(f"Saved: {filename}")

# Function to scrape and save a standalone poem
def scrape_and_save_standalone_poem(poem_url, title):
    response = http.request("GET", poem_url, headers={"User-Agent": "Mozilla/5.0"})
    soup = BeautifulSoup(response.data.decode('utf-8'), 'html.parser')

    # Extract content from blockquote tags
    stanzas = [blockquote.get_text(separator="").strip() for blockquote in soup.find_all("blockquote")]

    # Join stanzas with double newlines
    poem_text = "\n\n".join(stanzas)

    # Save the poem as a standalone file
    filename = f"{title}.txt"
    with open(filename, "w", encoding="utf-8") as file:
        file.write(poem_text)
    print(f"Saved standalone poem: {filename}")

# Function to scrape and save the elegy
def scrape_and_save_elegy(poem_url, title):
    response = http.request("GET", poem_url, headers={"User-Agent": "Mozilla/5.0"})
    soup = BeautifulSoup(response.data.decode('utf-8'), 'html.parser')

    content = []

    print(soup.find_all("td"))

    # Find all <td> tags
    for td in soup.find_all("tr"):
        content.append(td.get_text(strip=True))

    # Ensure each line is separated by a newline
    elegy_text = "\n".join(content)

    # Save the elegy as a standalone file
    filename = f"{title}.txt"
    with open(filename, "w", encoding="utf-8") as file:
        file.write(elegy_text)
    print(f"Saved standalone elegy: {filename}")

# Main scraping process
if __name__ == "__main__":
    # Call the function to scrape and save the elegy
    poem_url = "https://shakespeare.mit.edu/Poetry/elegy.html"
    title = "Elegy"
    scrape_and_save_elegy(poem_url, title)
