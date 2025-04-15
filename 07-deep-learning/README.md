# Neural Networks and Deep Learning
## Tensorflow and Keras
### Basic Concepts
- Each image consists of pixel and each of these pixels has the shape of 3 dimensions (**height, width, color channels**). And it can be converted to a 3-dimensional array using `np.array()`.
- A typical color image consists of three color channels: **red, green and blue**. Each color channel has 8 bits or 1 byte and can represent distinct values between 0-256 (uint8 type).

### Classes, functions, and methods:
- `import tensorflow as tf` -> to import tensorflow library
- `from tensorflow import keras` -> to import keras
- `from tensorflow.keras.preprocessing.image import load_img` -> to import load_img function
- `load_img('path/to/image', targe_size=(150,150))` -> to load the image of 150 x 150 size in PIL format. The `target_size` can only accept three image size: 299\*299, 224\*224, 150\*150.
- `np.array(img)` -> convert image into a numpy array of 3D shape, where each row of the array represents the value of red, green, and blue color channels of one pixel in the image.

## Pre-trained Convolutional Neural Networks
### Classes, functions, and methods:
- `from tensorflow.keras.applications.xception import Xception` -> import the model from keras applications
- `from tensorflow.keras.application.xception import preprocess_input` -> function to perform preprocessing on images
- `from tensorflow.keras.applications.xception import decode_predictions` -> extract the predictions class names in the form of tuple of list
- `model.predict(X)` -> function to make predictions on the test images

## Convolutional Neural Network: Concepts
### Simple Architecture of CNN
Simple CNN consists of three components: convolutional layer, pooling layer, vector representation, and dense layer.
### Convolutional Layer
目的：整合数据，发现数据特征  
过程：Convolutional Layers are the first step of CNN. In each convolutional layer, each filter generate one feature map. In other words, each convolutional layer consists of multiple filters and their corresponding feature maps.  
And then, the convolutional layer will pass filters to the next convolutional layer, which generate new combined filters upon old filters.   
### Pooling Layer
The main purpose of Pooling Layer is to shrink the size of convolution layers. 
### Vector Representation
Vector Representation receive data from convolutional layers and convert them to a vector
### Dense Layer
目的：接收数据以及特征，对特征进行非线性处理，输出预测结果  
过程：每个output对应着一种预测结果，并自带一组weights, Dense Layer将每一组weights与vector进行运算, 并放进`softmax`中转化, 得出最终的预测结果(soft max = sigmoid for multiple classes).

## Convolutional Neural Network: Coding
### Neural Network Architecture
- Input layer: prepare data, read raw image data
- Convolutional layer: base model
- Vector representation and Pooling
- Dense layer: custom model
- Output layer: multi-class prediction
### Prepare Data
`image_dataset_from_directory`可以自动解析feature values and target values
```python
from keras.preprocessing import image_dataset_from_directory

train_gen = image_dataset_from_directory(
    traning_dataset, 
    image_size = (150, 150), # the size of images
    batch_size = 32) # the number of images in each batch
```
`X`包含32张图片(`batch size`), 每张含有3个channels(`red`, `green`, `blue`), 每个channel大小为`150 * 150`(`image size`). `X.shap`e为`(32, 150, 150, 3)`, 其中第一个`150`表示有150行, 第二个`150`表示有150列, `3`表示每个像素中的三元素比例.   

`y`为图片所在folder的folder index (e.g. 1,2,3).
```python
X,y = train_gen \
.map(lambda x, y: (preprocess_input(x),y)) \
.as_numpy_iterator() \
.next()
# X.shape = (32,150,150,3)
```
可以通过`class_names`读取classes
```python
train_gen.class_names
```

### Convolutional Layer
In this case, we only use convolution layer from `Xception` model
```python
# Xception is a full neural network model, including convolutional layer and dense layer.
base_model = Xception(weights='imagenet',
                     include_top=False, # only use convolutional layer
                     input_shape=(150,150,3))

base_model.trainable=False # freeze convolutional layer (don't train)
```

### Vector Representation and Pooling
The process of reducing 3-D array to 1-D array is called `Pooling`.  
In this case, we apply average pooling method `GlobalAveragePooling2D` on 3-D array, calculating the average of each `5 * 5` array and storing the result in the new 1-D array
```python
# X
inputs = keras.Input(shape=(150,150,3)) 

# Convolution Layer, return 3-D array
base = base_model(inputs, training=False) 

# Pooling (reduce dimensionality) and Vectorizing, return 1-D array
vectors = keras.layers.GlobalAveragePooling2D()(base)

# Simple Dense Layer, assign weights for 10 classes
outputs = keras.layers.Dense(10)(vectors)

# Consolidation: grouping layers into a model
model = keras.Model(inputs, outputs)

# Prediction before adjusting weights
pred = model.predict(X)
```
#### Before Pooling
The shape of array - `32 * 5 * 5 * 2048`
- 3-D array generated from convolutional layer
- `32`: batch size
- `5 * 5`: feature map size
- `2048`: the number of feature maps

#### After Pooling
The shape of array - `32 * 2048`
- 1-D array generated from pooling layer
- `32`: batch size
- `2048`: the number of feature maps

### Dense Layer: Weights
Randonly assign weights to each class
```python
outputs = keras.layers.Dense(10)(vectors)
```

### Dense Layer: Optimizer and Losses
#### Losses
- Compare differences between prediction and target
- Send error info back to optimizer
#### Optimizer
- `Adam` is a popular adaptive learning rate method
- Adjust weights in terms of error sent from Losses and learning rate
- Use for finding the best weights in dense layer
```python
#Optimizer
learning_rate=0.01
optimizer = keras.optimizers.Adam(learning_rate=learning_rate)

#Losses
loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True) 

# Compile optimizer and loss before training the model
model.compile(optimizer=optimizer, loss=loss, metrics = ['accuracy'])
```
#### Losses for Different Models
- Multi-class classification: `CategoricalCrossentropy` (y是整数), `SparseCategoricalCrossentropy` (y是one-hot encoding)
- Binary classification: `BinaryCrossentropy`
- Linear regression: `MeanSquaredError`

#### Parameter in Losses
- `from_logits = True`: return row score without using activation method
- `from_logits = False`: return probabilities calculating by activation method. For example, `outputs = keras.layers.Dense(10, activation='softmax')(vectors)`

### Dense Layer: Translate Results
After calculating vector and weights, we have outputs from dense layer called raw scores. Activation method, like `softmax`, translate the predicted raw output to probabilities of classes.

### Train Model
This `fit` process is to train NN model in 10 times (defined by `epoch`) - losses evaluates weights and optimizer adjusts weights (defined in `compile`).  
In this case, training dataset is divided to `96` batches (batch size=`32`). For each epoch, weights are trained and adjusted for 96 times. For all the epochs, weights are trained for `96 * 10` times
```python
# Save model evaluation metrics to variable 'history', the same as `%%capture output`  
history = model.fit(train_ds, 
    epochs=10, # how many times to iterator the whole dataset
    validation_data=val_ds)
```
## Learning Rate
`Learning Rate` is a tuning parameter in an optimization function that determines the step size (how big or small) at each iteration while moving toward a mininum of a loss function.  
可以比作学习速度，学的太快（数值太大），忘得快，真正重要的内容没有吸收. 学的太慢（数值太小），学习周期过长，重要的内容没学到.  
Try different learning rate in `optimizer = keras.optimizers.Adam(learning_rate)`.

## CheckPoint
`ModelCheckpoint` is a method in `tensorflow.keras.callbacks` to save model and weigths.
```python
checkpoint = tf.keras.callbacks.ModelCheckpoint(
    'xception_v1_{epoch:02d}_{val_accuracy:.3f}.keras', # Name model file with epoch and validation accuracy
    save_best_only=True, # Only save model with best performance
    monitor='val_accuracy', # Define metrics to evaluate model performance
    mode = 'max' # Find the model with the highest val_accuracy
)
```
## Add More Layers in Dense Layer
Adding more layers in dense layer could help the NN model more accurate. And this kind of layer is called "intermediate layer“.  
Intermediate layer, same as output dense layer, has two important parameters - classes and activation method.  
```python
# Intermediate dense layer
inner = keras.layers.Dense(size_inner, activation = 'relu')(vectors) 
# Output dense layer
outputs = keras.layers.Dense(10)(inner)
```
Here are some commonly used activation functions in intermediate dense layer:
- `Sigmoid`: range [0, 1], the large negative number becomes 0 and the large positive number become 1.
- `Tanh`: range [-1,1], similar as `sigmoid`
- `ReLu`: range [0, inf], when x <= 0, f(x) = 0; when x > 0 , f(x) = x
- `Leaky ReLU`: "upgraded" ReLu, but lack of consistency
- `Maxout`: "upgraded" ReLu with expensive computing cost

In output dense layer, `softmax` is the most popular method for multi-class questions.

## Regularization and Dropout
Dropout is a technique that **prevents overfitting** by randomly freeze nodes in dense layers.In other words, dropout force NN model to learn data in a higher data and ignore outliers.
```python
## Inner 
inner = keras.layers.Dense(size_inner, activation = 'relu')(vectors)
## Dropout
### dropout = freeze partial neuros in inner dense layer
### drop rate determines how many neuros (%) to be frozen
drop = keras.layers.Dropout(droprate)(inner)
# Output
outputs = keras.layers.Dense(10)(drop)
```
`droprate` determines the percentage of frozen nodes to total.

## Data Augmentation
The benefits of data augmentation is to expand dataset size, increase the diversity and complexity of NN model.
### Commonly Used Techniques
- Flip：上下左右翻动
- Rotation: 360度旋转
- Shift: 左右平移和上下平移
- Shear：拉扯image的四个角，使图片变形
- Zoom: 按长宽zoom in and zoom out
- Brightness and contrast
- Dropout

### When to Use Data Augmentation
- If there is any variance in images?
- Train data with data augmentation in 10-20 epochs. Good performance, use it. Same or worse performance, dont use it

```python
data_augmentation = keras.Sequential([
    keras.layers.Lambda(preprocess_input),
    keras.layers.RandomFlip("vertical"), # 以垂直线为中轴线翻转
    # keras.layers.RandomRotation(0.1667), #转动角度比例 x/180
    keras.layers.RandomZoom(height_factor=0.1, width_factor=0.1) # 缩放比例
    # keras.layers.RandomTranslation(height_factor=0.0667, width_factor=0.0667) #平移比例 x/image size
])
```

## Test Model
- `keras.models.load_model(path)`: method to load saved model  
- `model.evaluate()`: method to evaluate the performance of the model based on the   evaluation metrics
- `model.predict()`: method to make predictions of output depending on the input  
