from selenium import webdriver
import time

from selenium.common import TimeoutException
from selenium.webdriver.common.by import By
from selenium.webdriver.firefox.service import Service as GeckoService

from webdriver_manager.firefox import GeckoDriverManager

import pandas as pd

options = webdriver.FirefoxOptions()

options.set_preference('profile', r'Path/to/Mozilla/Profile')
options.set_preference("dom.webdriver.enabled", False)
options.set_preference("useAutomationExtension", False)
options.set_preference("dom.webnotifications.enabled", False)
options.set_preference("security.sandbox.content.level", 0)

driver = webdriver.Firefox(service=GeckoService(GeckoDriverManager().install()), options=options)
driver.set_page_load_timeout(10)

data = {'title':[],
        'description':[],
        'date':[],
        'type':[]}

titles = []

def save_data(data):
    driver.quit()
    df_text = pd.DataFrame(data=data)
    df_text.to_csv('parsing_date_text_cbr.csv')


def scroll_down(driver):
    time.sleep(10)
    stop = False
    while True:
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        try:
            loadnext = driver.find_element(By.ID, '_buttonLoadNextEvt')
            loadnext.click()

        except Exception as e:
            stop = True

        if stop == True:
            break

driver.delete_all_cookies()
driver.execute_script("window.open('');")
driver.switch_to.window(driver.window_handles[-1])
driver.get('https://www.cbr.ru/')
time.sleep(10)

scroll_down(driver)
time.sleep(2)

news_item = driver.find_elements(By.CLASS_NAME, 'news_inner')
for item in news_item:
    links = item.find_element(By.CLASS_NAME, 'news_title').get_attribute('href')

    date = item.find_element(By.CLASS_NAME, "news_date").text
    if date == '02 марта 2022':
        save_data(data)
        driver.close()

    type = item.find_element(By.CLASS_NAME, "news_category").text

    if type == 'Интервью' or type == 'Выступление':
        continue

    title = item.find_element(By.CLASS_NAME, 'news_title').text
    driver.execute_script("window.open('');")
    driver.switch_to.window(driver.window_handles[-1])
    try:
        driver.get(links)
        try:
            description = driver.find_element(By.CLASS_NAME, 'lead-text').text
        except Exception as e:
            description = driver.find_element(By.CSS_SELECTOR, '.landing-text > p:nth-child(1)').text

        if title not in titles:
            titles.append(title)
            data['title'].append(title)
            data['description'].append(description)
            data['date'].append(date)
            data['type'].append(type)

        print('Тип:', type)
        print('Дата:', date)
        print("Заголовок:", title)
        print(f"Текст под заголовком: {description}")
        print("-" * 50)

        driver.close()
        driver.switch_to.window(driver.window_handles[-1])
    except TimeoutException:
        driver.close()
        driver.switch_to.window(driver.window_handles[-1])
        continue

save_data(data)
driver.close()



