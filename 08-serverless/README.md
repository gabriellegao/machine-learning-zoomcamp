# Serverless
## Tensorflow Lite
### Convert TF Model to TF Lite
```python
converter = tf.lite.TFLiteConverter.from_keras_model(model)

tflite_model = converter.convert()

with open('clothing-mode.tflite', 'wb') as f_out:
    f_out.write(tflite_model)
```

### Initialize TF Lite Model
这个模型还使用着tensorflow完整版的`preprocess_input` method
```python
import tensorflow.lite as tflite
from tensorflow.keras.preprocessing.image import load_img
from keras.applications.xception import Xception
from keras.applications.xception import preprocess_input

img = load_img('pants.jpg', target_size = (299,299))

x = np.array(img)
X = np.array([x])
X = preprocess_input(X)

interpreter = tflite.Interpreter(model_path = 'clothing-mode.tflite')
interpreter.allocate_tensors()

input_index = interpreter.get_input_details()[0]['index']
output_index = interpreter.get_output_details()[0]['index']

interpreter.set_tensor(input_index, X)
interpreter.invoke()
interpreter.get_tensor(output_index)
```

### Remove `Load_Image` and `Preprocess_Input` Dependency
```python
# 还原load_img method
from PIL import Image

path = 'pants.jpg'
with open(path, "rb") as f:
    img = img.resize((299,299), Image.NEAREST)

# 还原preprocess_input method
def preprocess_input(x):
    x /= 127.5
    x -= 1.0
    return x

# 更改img data type from integer to float, 以方便在def func中运算
x = np.array(img, dtype='float32')
X = np.array([x])
X = preprocess_input(X)

interpreter = tflite.Interpreter(model_path = 'clothing-mode.tflite')
interpreter.allocate_tensors()

input_index = interpreter.get_input_details()[0]['index']
output_index = interpreter.get_output_details()[0]['index']

interpreter.set_tensor(input_index, X)
interpreter.invoke()
interpreter.get_tensor(output_index)
```

### Download Tensorflow Lite
仅下载tensorflow lit package
```bash
pip install tflite-runtime
```
引用tensorflow lite package
```python
import tflite_runtime.interpreter as tflite
```

## Push Model to Docker
### Config Environment in Docker
这个步骤预设好代码环境，使用了`AWS Lambda Python 3.10`版本，以及`compiled tensorflow lite`  
- 如何配置`AWS Lambda`环境下的`tensorflow lite`，可以查看此[Link](https://github.com/alexeygrigorev/tflite-aws-lambda)
- 配置[Dockerfile](./Dockerfile)

### Config TF Lite Model
- 设置`tflite model`
- 构建`lambda handler` method  
  
  `lambda_handler`有两个parameters - `event` and `context`, 可以通过`event`以`python dictionary`格式，传输数据到`lambda_handler`中

- script: [`lambda_function.py`](./lambda_function.py)

### Test Connection
- 创建docker image 和 docker container, 并连接port`8080`
  
```bash
docker build -t clothing-model .

docker run -it --rm -p 8080:8080 clothing-model
```
- script: [配置`requests.post`](./connection-test.py)
- 测试连接
```python
pyton connection-test.py
```
## Push Model to AWS Lambda
AWS Lambda的作用在于，远程存储model，以及其需要的环境配置. 并且配置数据接收function.
### Cofig AWS CLI
下载完毕后，需要在账户内设置权限
```bash
pip install awscli
awscli configure
```

### Create Registry
```bash
aws ecr create-repository --repository-name clothing-tflite-images
```

### Login to AWS Container Registry
```bash
# 输出密码和网址
aws ecr get-login --no-include-email
# 用于login的command
$(aws ecr get-login --no-include-email)
```

### Define Remote URI
```bash
ACCOUNT=209479306945
REGION=us-east-2
REGISTRY=clothing-tflite-images
PREFIX=${ACCOUNT}.dkr.ecr.${REGION}.amazonaws.com/${REGISTRY}
TAG=clothing-model-xception-v4-001
REMOTE_URI=${PREFIX}:${TAG}
```

### Tag URI to Docker Image
将`REMOTE_URI`作为标签贴在本地的`image`上
```bash
docker tag clothing-model:latest ${REMOTE_URI}
```

### Push Images to AWS Elastic Container Registry
以`REMOTE_URI`为索引查找本地`image`，并且将此`image`上传至`AWS Elastic Container Registry`
```bash
docker push ${REMOTE_URI}
```

### Test on Lambda
在`Test`页面，将Pants `URL`输入到`Event JSON`

## AWS API Gateway
API的作用类似bridge, 连接存储在Lambda的model, 创建其专属的endpoint, 方便用户通过连接API(url, endpoint)传输数据到Lambda中.
### Steps of Creating API Gateway
Create REST API -> Create Resource (link to Lambda Function) -> Create Method (POST) -> Deploy API

## Setup Virtual Environment
```bash
# Download packages
pipenv install tensorflow jupyter ipykernel
# Register the current virtual environment as Jupyter Kernel
python -m pip install --upgrade ipykernel
python -m ipykernel install --user --name=homework --display-name "Python (Pipenv-TF)"
```