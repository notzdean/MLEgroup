import requests

LTA_KEY = "your_api_key_here"

def fetch_mrt_volume(year_month: str) -> list:
    # year_month format: "YYYY-MM" e.g. "2016-01"
    url = "https://datamall2.mytransport.sg/ltaodataservice/PV/Train"
    headers = {"AccountKey": LTA_KEY, "accept": "application/json"}
    resp = requests.get(url, headers=headers, params={"Date": year_month})
    # Returns a download link — then fetch that link
    link = resp.json()["value"][0]["Link"]
    data = requests.get(link)
    # Save the CSV
    with open(f"data/raw/lta_mrt_{year_month}.csv", "wb") as f:
        f.write(data.content)