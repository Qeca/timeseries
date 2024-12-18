import pandas as pd
import requests
def fetch_moex_data(security, from_date, till_date):
    url = f'http://iss.moex.com/iss/history/engines/stock/markets/shares/securities/{security}.json'
    start = 0
    data_frames = []

    while True:
        params = {
            'from': from_date,
            'till': till_date,
            'start': start
        }
        response = requests.get(url, params=params)
        if response.status_code != 200:
            print(f"Ошибка HTTP {response.status_code}")
            break

        json_data = response.json()

        if 'history' in json_data and 'data' in json_data['history']:
            columns = json_data['history']['columns']
            data = json_data['history']['data']

            if not data:
                break

            df = pd.DataFrame(data, columns=columns)
            data_frames.append(df)

            if len(data) < 100:
                break
            else:
                start += 100
        else:
            break

    if data_frames:
        result_df = pd.concat(data_frames, ignore_index=True)
        return result_df
    else:
        return pd.DataFrame()