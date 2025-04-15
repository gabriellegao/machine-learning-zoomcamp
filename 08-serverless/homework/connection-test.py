import requests
# Docker Test URL
# url = 'http://localhost:8080/2015-03-31/functions/function/invocations'

# AWS API Test URL
url = 'https://fawt1rtxl9.execute-api.us-east-2.amazonaws.com/test/predict'

data = {'url':'https://habrastorage.org/webt/yf/_d/ok/yf_dokzqy3vcritme8ggnzqlvwa.jpeg'}

result = requests.post(url=url, json=data).json()
print(result)