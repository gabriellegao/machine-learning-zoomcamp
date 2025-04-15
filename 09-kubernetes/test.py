import requests
# import os
# import certifi
# os.environ['SSL_CERT_FILE'] = certifi.where()

# url = 'http://localhost:8080/predict'

url = 'http://a4a853d8db21b47f892015f964cc4ab5-1561477282.us-east-2.elb.amazonaws.com/predict'


data = {"url":"http://bit.ly/mlbookcamp-pants"}

result = requests.post(url = url, json = data).json()
print(result)