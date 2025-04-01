import pandas as pd
import os
from bs4 import BeautifulSoup
from selenium import webdriver
import time

# Working directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # Correct way to get the base directory
PROJECT_DIR = "C:\\Users\\jfbaa\\PycharmProjects\\ds2500\\2500 project" # Explicitly define your project directory

driver = webdriver.Chrome()

seasons_wanted = list(reversed(range(2008, 2024)))

# urls
root_url = 'https://www.oddsportal.com/football/england/premier-league/'
restof_url = '/results/'

# combine urls to get the main url
main_url = root_url + restof_url

# the urls for the other seasons
season_url = [root_url + '-' + str(season) + '-' + str(season + 1) + restof_url for season in seasons_wanted]

# combine the urls to get the main urls to scrape
all_urls = [main_url] + season_url
all_urls

# create df
df = pd.DataFrame()

# function to check null values
def is_null(col):
    try:
        result = col.text
    except:
        result = None
    return result

for url in all_urls:
    # selenium
    driver.get(url)
    html = driver.page_source
    soup = BeautifulSoup(html, 'html.parser', from_encoding='utf-8')

    while True:
        # store previous or current page number
        active_page_element = soup.find('a', class_='pagination-link', text=lambda text: text and text.strip() == '1')  # Adjust the "1"

        if active_page_element:
            previous_page = active_page_element['data-number']
        else:
            break

        for col in soup.find_all('tr', attrs={'deactivate'}):
            df = df.append(
                {
                    # match season
                    'season': soup.find_all('span', attrs={'class': 'active'})[1].text,
                    # match date
                    'date': col.findPreviousSibling(attrs={'center nob-border'}).text[0:-6],
                    # match name
                    'match_name': col.find('td', attrs={'class': 'name table-participant'}).text.replace('\xa0', ''),
                    # match result
                    'result': col.find('td', attrs={'class': 'center bold table-odds table-score'}).text,
                    # home winning odd
                    'h_odd': is_null(col.find('td', attrs={'class': "odds-nowrp"})),
                    # draw odd
                    'd_odd': is_null(col.find('td', attrs={'class': "odds-nowrp"}).findNext(attrs={'class': "odds-nowrp"})),
                    # away winning odd
                    'a_odd': is_null(col.find('td', attrs={'class': "odds-nowrp"}).findNext(attrs={'class': "odds-nowrp"}).findNext(attrs={'class': "odds-nowrp"}))
                },
                ignore_index=True
            )

        print("done!")

        # clicks on next page
        element = driver.find_element("partial link text", '»')
        driver.execute_script("arguments[0].click();", element)

        # sleep so that the page can load properly
        time.sleep(2)

        # reload soup objects on new page
        html = driver.page_source
        soup = BeautifulSoup(html, 'html.parser', from_encoding="utf-8")

        # get new page number
        active_page_element = soup.find('a', class_='pagination-link', text=lambda text: text and text.strip() == '1')  # Adjust the "1"

        if active_page_element:
            new_page = active_page_element['data-number']
        else:
            break

        # if there's no new pages left break
        if previous_page != new_page:
            continue
        else:
            break

    print(url, 'done!')

driver.quit()
print('scraping finished!')

# reordering columns
df = df[['season', 'date', 'match_name', 'result', 'h_odd', 'd_odd', 'a_odd']]

# saving full df to csv
df.to_csv(os.path.join(PROJECT_DIR, 'matches.csv'), index=False)