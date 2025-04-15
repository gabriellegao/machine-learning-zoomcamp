FROM public.ecr.aws/lambda/python:3.10

RUN pip install keras-image-helper
RUN pip install https://github.com/alexeygrigorev/tflite-aws-lambda/raw/main/tflite/tflite_runtime-2.14.0-cp310-cp310-linux_x86_64.whl
RUN pip install numpy==1.23.5

COPY clothing-model.tflite .
COPY lambda_function.py .

#在AWS Lambda环境中，可以直接调取以下function，不需要像在普通Docker镜像中，额外添加python3
CMD [ "lambda_function.lambda_handler" ]
