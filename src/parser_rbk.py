
from selenium import webdriver
import time

from selenium.webdriver.common.by import By
from selenium.webdriver.firefox.service import Service as GeckoService

from webdriver_manager.firefox import GeckoDriverManager

from links import links2024, links2022, links2023

import pandas as pd
import numpy as np


options = webdriver.FirefoxOptions()

options.set_preference('profile', r'Path/to/Mozilla/Profile')
options.set_preference("dom.webdriver.enabled", False)
options.set_preference("useAutomationExtension", False)
options.set_preference("dom.webnotifications.enabled", False)
options.set_preference("security.sandbox.content.level", 0)

driver = webdriver.Firefox(service=GeckoService(GeckoDriverManager().install()), options=options)

def scroll_down(driver):
    time.sleep(10)
    stop = False
    while True:
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")

        all_date = driver.find_elements(By.CLASS_NAME, 'search-item__category')
        for items in all_date:
            if '09' in items.text.split(' ') or '10' in items.text.split(' '):
                stop = True
        if stop == True:
            break


def save_data(data):
    driver.quit()
    df_text = pd.DataFrame(data=data)
    df_text.to_csv('parsing_date_text_y2022.csv')

    #df_img = pd.DataFrame(data=image_data)
    #df_img.to_csv('parsing_date_img_v2024.csv')



titles = []

data = {'title':[],
        'description':[],
        'date':[],
        'type':[],
        'comp_short':[],
        'data_exchange':[],
        'data-price':[]}

image_data = {'image':[],
              'link':[]}

for link in links2022:
    driver.delete_all_cookies()
    driver.execute_script("window.open('');")
    driver.switch_to.window(driver.window_handles[-1])
    driver.get(link)
    time.sleep(2)

    scroll_down(driver)
    time.sleep(2)

    news_items = driver.find_elements(By.CLASS_NAME, 'search-item.js-search-item')
    for item in news_items:
            only_link = item.find_element(By.CLASS_NAME, 'search-item__link.js-search-item-link').get_attribute('href')
            driver.execute_script("window.open('');")
            driver.switch_to.window(driver.window_handles[-1])
            driver.get(only_link)
            all_items = driver.find_elements(By.CLASS_NAME, 'l-col-center-590.article__content')
            for text in all_items:
                try:
                        '''
                        image_item = text.find_element(By.CLASS_NAME, 'g-image.article__main-image__image').get_attribute('src')
                        response = requests.get(image_item)
                        if response.status_code == 200:

                            img = np.asarray(Image.open(BytesIO(response.content)).convert('RGB'))
                            image_data['image'].append(img)
                            image_data['link'].append(image_item)
                        else:
                            img = np.nan
                            print('не удалось загрузить изображение')
                        '''

                        title = text.find_element(By.CLASS_NAME,"article__header__title-in.js-slide-title").text

                        try:
                            description = text.find_element(By.CLASS_NAME,"article__text__overview").text

                        except Exception as e:
                            description = np.nan

                        date = text.find_element(By.CLASS_NAME, "article__header__date").text
                        type = text.find_element(By.CLASS_NAME, "article__header__category").get_attribute('content')

                        try:
                            wrap = text.find_elements(By.CLASS_NAME, 'q-item__company.js-company-ticker')
                            massiv1 = []
                            massiv2 = []
                            massiv3 = []
                            for val in wrap:
                                if title not in titles:

                                    comp_short = val.get_attribute('data-companyshortname')
                                    data_exchange = val.get_attribute('data-exchangepercent')
                                    data_price = val.get_attribute('data-price')

                                    massiv1.append(comp_short)
                                    massiv2.append(data_exchange)
                                    massiv3.append(data_price)
                            if title not in titles:
                                data['comp_short'].append(massiv1)
                                data['data_exchange'].append(massiv2)
                                data['data-price'].append(massiv3)
                                print('Акции:', massiv1, massiv2, massiv3)
                        except Exception as e:
                            massiv1 = ['NOTHING']
                            massiv2 = ['NOTHING']
                            massiv3 = ['NOTHING']


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
                        break
                except Exception as e:
                    print('')
                    driver.close()
                    driver.switch_to.window(driver.window_handles[-1])
                    continue
    driver.close()
    driver.switch_to.window(driver.window_handles[-1])



qqq = save_data(data)

print("\nВсе уникальные заголовки:", len(titles))
