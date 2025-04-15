# Deployment
## Pickle
***Pickle*** is a librabry designed to save machine learning models.

### Download Pickle Library
```bash
pip install pickle-mixin
```

### Save Model
```python
import pickle

with open('model.bin', 'wb') as f_out: # 'wb' means write-binary
    pickle.dump((dict_vectorizer, model), f_out)
```

### Use Model
```python
import pickle

with open('mode.bin', 'rb') as f_in: # very important to use 'rb' here, it means read-binary 
    dict_vectorizer, model = pickle.load(f_in)
## Note: never open a binary file you do not trust the source!
```

## Web Server: Flask Intro
### What is Web Service?
***Web Service*** is a method used to communicate between electronic devices. 
- `GET`: retrieve data.
- `POST`: send data to server.
- `PUT`: similar as `POST` but allow users to specify the destination path.
- `DELETE`: delete data from server.

### Create a Simple Web Server
#### Install Flask Library
```bash
pip install Flask
```

#### Create a Web Server
```python
from flask import Flask

app = Flask('ping') # give an identity to your web service

@app.route('/ping', methods=['GET']) # use decorator to add our function to Flask route function
def ping():
    return 'PONG'

if __name__ == '__main__':
   app.run(debug=True, host='0.0.0.0', port=9696) # run the code in local machine with the debugging mode true and port 9696
```
- `/ping`: URL subfix
- `0.0.0.0`: localhost

#### Test Web Service
Method1: Run commands in console
```bash
curl http://localhost:9696/ping
curl http://0.0.0.0:9696/ping
```
Method2: search URL in browser
```url
http://0.0.0.0:9696/ping
http://localhost:9696/ping
```

## Flask Deployment
### Read Model
Read model saved by pickle library
```python
model_file='model_C=1.0.bin'

with open(model_file, 'rb') as f_in:
    dv,model=pickle.load(f_in)
```

### Initialize Flask App Instance
Create a Flask instance named `app` (external) and `churn` (internal)
```python
app = Flask('churn')
```

### Create Web Server
```python
@app.route('/predict', methods=['POST'])
def predict():

    customer = request.get_json() #read data in json format

    X = dv.transform([customer])
    y_pred = model.predict_proba(X)[0,1]
    churn = y_pred >= 0.5

    result = {
        "churn_probability": float(y_pred), #convert numpy float to python float
        "churn": bool(churn) #convert numpy boolean to python boolean
    }
    return jsonify(result) #convert dict to json
```
- `/predict`: URL subfix
- `methods=['POST']`: receive data sent to web server URL.
- `.get_json()`: read data in `json` format (required).
- `result`: store output data in python dictionary.
- `float()` and `bool()` cast numpy data types to python native data types.
- `jsonify()`: cast `dict` to `json` (required).

### Setup Web Server URL
```python
app.run(debug=True, host='0.0.0.0', port=9696) #0.0.0.0: localhost
```

### Setup Data Sending Request
```python
import requests

url = 'http://localhost:9696/predict'

response = requests.post(url, json=customer).json()
```
- `import requests`: a library designed to send data to POST web service.
- `url`: use the same web server URL setup in previous step.
- `requests.post()`: send data to web server URL.
- `.json()`: translate server response to readable messages

## Gunicorn
这是一个适用于Prod Env的Python WSGI HTTP Server，用来承载Flask类型的应用，更加高效的的处理HTTP requests.
### Start Web Server
Install production server
```bash
pip install gunicorn
```
Run the web server
```bash
gunicorn --bind 0.0.0.0:9696 predict:app
```
- `--bind 0.0.0.0:9696`: URL
- `predict:app`: `predict` is the name of file containing the code `app = Flask('churn')` and `app` is the Flask instance's internal name.

### Send Data to Web Server
```bash
python3 predict-test.py
```

## Virtual Environment
### Basic
Python libraries normally is installed into the location `~/python/bin/` or `opt/anaconda3/bin/`. And `pip` is a library to install other python libraries, stored in this location `python/bin/pip`.  
Virtual environment provides an option to create different environments in one machine and allow different versions of libraries installed on those environments. 
### `Pipenv`
There are multiple libraries providing virtual environment functionality.
- `venv`
- `conda`
- `pipenv` (recommended by python community)
- `poetry`
```bash
pip install pipenv
```
### Install Libraries on Virtual Environment
After running this bash command, `Pipfile` and `Pipfile.lock` will be generated and loaded to the current directory.  
All installed libraries can be found in `Pipfile`.
```bash
pipenv install numpy scikit-learn==0.24.1 flask gunicorn
```
### Launch Subshell in Virtual Environment
This command launch the subshell, and also returns the location storing this virtual environment.
```bash
# Method1
pipenv shell
gunicorn --bin 0.0.0.0:9696 predict:app

# Method2
pipenv run gunicorn --bin 0.0.0.0:9696 predict:app
```
### Exit SubShell
```bash
# Method1
exit #log out subshell
# Method2
ctrl + D
```

## Docker Intro
```bash
docker run -it --rm --entrypoint=bash python:3.9
```
- `--rm`: remove container after it is closed.
- `--entrypoint=bash`: open container started with bash shell
```bash
# Build image
docker build -t zoomcamp-test .
# Run container
docker run -it --rm --entrypoint=bash --name=<container_name> zoomcamp-test
```
## Docker and Web Server
### Prepare `Dockerfile`
```dockerfile
FROM python:3.8.12-slim

# Install pipenv library
RUN pip install pipenv

WORKDIR /app

# Copy files to Docker container
COPY ["Pipfile", "Pipfile.lock", "./"] 

# Install libraries in Pipfile in the environment
# Instead of launch a virtual env
RUN pipenv install --system --deploy

# Copy script and model to Docker Container
COPY ["predict.py", "model_C=1.0.bin", "./"]

# Claim ports
EXPOSE 9696

# Run the gunicorn command while starting container
ENTRYPOINT [ "gunicorn", "--bind=0.0.0.0:9696", "predict:app" ]
```
- The flags `--deploy` and `--system` make sure that we install the dependencies directly inside the Docker container without creating an additional virtual environment (which pipenv does by default).  
  
### Build Image
```bash
docker build -t zoomcamp-test .
```
### Run Container with Port Mapping
```bash
docker run -it -p 9696:9696 --name=zoomcamp-container zoomcamp-test
```
- `-p 9696:9696`: map port `9696` of Docker container to port `9696` of localhost

## Deployment to Cloud
### Start VM
```bash
gcloud compute instances start de-zoomcamp --zone=us-central1-c
# Locate ssh config file
ssh de-zoomcamp
```
### Upload Files to VM
```bash
gcloud compute scp <file1> <file2> <file3> de-zoomcamp:<path/in/vm> --zone=us-central1-c
```
### Start Docker
```bash
# Build Image
docker build -t <image-name> .
# Create Container
docker run -it -p 9696:9696 --name=<container-name> <image_name>
```
### Test Model Connection
```bash
# Run predict-test.py on local machine
python3 predict-test.py
```
### GCP Firewall Rule and TCP Traffic
If the local `predict-test.py` cannot talk to VM externally (use external IP), one of reasons is Google Cloud is blocking traffic. The following command adds port `9696` to `allow-external-traffic` fixing the issue.
```bash
gcloud compute firewall-rules create allow-external-traffic \
    --allow tcp:9696 \
    --source-ranges=0.0.0.0/0 \
    --description="Allow external access to Flask servers"
```