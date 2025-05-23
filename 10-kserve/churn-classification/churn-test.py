import requests 

service_name = 'churn'
namespace='churn-classification'
host = f'{service_name}.{namespace}.example.com'

actual_domain = 'http://localhost:8081'
url = f'{actual_domain}/v1/models/{service_name}:predict'


headers = {'Host': host}

request = {
    "instances": [
            {'contract': 'one_year', 'tenure': 34, 'monthlycharges': 56.95},
            {'contract': 'month-to-month', 'tenure': 13, 'monthlycharges': 49.95}
    ]
}


response = requests.post(url, json=request, headers=headers)
print(response.json())
