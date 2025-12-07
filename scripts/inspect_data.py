import requests
import json

url = "https://public.opendatasoft.com/api/records/1.0/search/"
params = {
    "dataset": "evenements-publics-openagenda",
    "rows": 1
}

response = requests.get(url, params=params)
if response.status_code == 200:
    data = response.json()
    if data.get("records"):
        print(json.dumps(data["records"][0]["fields"], indent=2))
    else:
        print("No records found.")
else:
    print(f"Error: {response.status_code}")
    print(response.text)
