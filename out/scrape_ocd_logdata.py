import os

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
import pandas as pd
import re

# Base URL
base_url = "https://ocdimage.emnrd.nm.gov/imaging/LogFileCriteria.aspx"
wellfile_base_url = "https://ocdimage.emnrd.nm.gov/imaging/"

def start_driver():
    # Set up Selenium WebDriver
    options = webdriver.ChromeOptions()
    options.add_argument("--headless")  # Run in headless mode (optional)
    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=options)
    return driver


def scrape_data(start_page=1, end_page=3372, driver=None):
    print(f'Start page = {start_page}; End page = {end_page}')
    # Store data
    data = []

    # Loop through pages
    for page in range(start_page, end_page + 1):
        print(f"Processing page {page}/{start_page}-{end_page}...")

        # Load the page
        driver.get(f"{base_url}?Page={page}")

        # Wait for the page to load
        WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.TAG_NAME, "a")))

        # Try to find and click the "Search" button
        try:
            search_button = WebDriverWait(driver, 5).until(
                EC.element_to_be_clickable((By.ID, "btnContinueFull"))
            )
            search_button.click()

            # Wait for the search results to load
            WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.TAG_NAME, "a")))
        except:
            print(f"Search button not found on page {page}, skipping...")

        # Find all well file links
        links = driver.find_elements(By.TAG_NAME, "a")

        for link in links:
            href = link.get_attribute("href")
            text = link.text.strip()

            # Check if it's a well file link
            if href and "WellFileView.aspx?RefType=WL&RefID=" in href:
                if re.match(r"\d{2}-\d{3}-\d{5}", text):  # Ensure API number format
                    data.append({"API": text, "Link": wellfile_base_url + href})

    # Close the browser
    driver.quit()

    # Create DataFrame
    df = pd.DataFrame(data)

    # Save to CSV
    name = 'NMOCD_wellswithlogs.csv'

    # Check if the file already exists
    file_exists = os.path.isfile(name)

    # Open the CSV file and append the dataframe to it
    df.to_csv(name, mode='a', header=not file_exists, index=False)

    # df.to_csv(name, mode='a', header=False, index=False)

    print(f"Scraping complete. Data saved to {name}")

def main():
    scrape_data()
    # Total pages
    # total_pages = 3372  # Test with 5 pages before running all 3,372

if __name__ == '__main__':
    main()

