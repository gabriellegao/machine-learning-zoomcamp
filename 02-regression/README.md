# Regression: Car Price Prediction Project

## Data Preparation
### Pandas attributes and methods:  
- `pd.read_csv(<file_path_string>)` -> read csv files  
- `df.head()` -> take a look of the dataframe  
- `df.columns` -> retrieve colum names of a dataframe  
- `df.columns.str.lower()` -> lowercase all the letters  
- `df.columns.str.replace(' ', '_')` -> replace the space separator  
- `df.dtypes` -> retrieve data types of all features  
- `df.index` -> retrieve indices of a dataframe  
- `df.astype(int)` -> convert original data type to integer
## Exploratory Data Analysis (EDA)
### Math Numpy `log()`
The numpy method `log()` is to narrow the data range (e.g. 1 to 10000 -> 1 to 5) and remain its data order. The purpose of applying `log()` is to avoid long-tail effect. 
### Pandas attributes and methods:
- `df[col].unique()` -> return a list of unique values in the series  
- `df[col].nunique()` -> return the number of unique values in the series  
- `df.isnull().sum() `-> return the number of null values in the dataframe  
### Matplotlib and seaborn methods:
- `%matplotlib inline` -> assure that plots are displayed in jupyter notebook's cells  
- `sns.histplot()` -> show the histogram of a series  
### Numpy methods:
- `np.log()` -> apply log tranformation to a variable
- `np.log1p()` -> apply log transformation to a variable, after adding one to each input value.

## Setup Data Validation Framework
In general, the dataset is splitted into three parts: training, validation, and test.  
### Pandas attributes and methods:
- `df.iloc[]` -> return subsets of records of a dataframe, being selected by numerical indices
- `df.reset_index()` -> restate the orginal indices
- `del df[col]` -> eliminate a column variable
### Numpy methods:
- `np.arange()` -> return an array of numbers
- `np.random.shuffle()` -> return a shuffled array
- `np.random.seed()` -> set a seed for reproducibility

## Linear Regression: Simple Version
### Basic Formula  
$
g(x_{i}) = w_{0} + w_{1}×x_{i1} + w_{2}×x_{i2} + w_{3}×x_{i3} = w_0 + \sum_{j=1}^3 x_{ij} \times W_j
$ 
### Numpy `log()` and `expm1()`
`numpy.log()` and `numpy.exp()` are inversely related, the same as `log1p()` and `expm1()`

## Linear Regression: Vector Form
### Upgraded Formula
The upgraded fromula adds 1 to each record $x_{i}^T$   

$g(x_{i}) = w_{0} \times 1+ x_{i}^T \times W = (1 +x_{i}^T) \times W$  

$
X=
\begin{bmatrix}
    1+x_{1}^{T} \\
    1+ x_{2}^{T} \\
    \vdots \\
    1+x_{n}^{T}
\end{bmatrix}
$
$\ =$
$\begin{bmatrix}
    1 & x_{11} & x_{12} & \dots  & x_{1d} \\
    1 & x_{21} & x_{22} & \dots  & x_{2d} \\
    \vdots & \vdots & \ddots & \vdots \\
    1 & x_{n1} & x_{n2} & \dots  & x_{nd}
\end{bmatrix}$
$\quad$
$W =
\begin{bmatrix}
    w_{0} \\
    w_{1} \\
    \vdots \\
    w_{n}
\end{bmatrix}$


${g(X)} = {y} =
\begin{bmatrix}
    (1+x_{1}^{T}) \times W\\
    (1+x_{2}^{T}) \times W\\
    \vdots \\
    (1+x_{n}^{T}) \times W
\end{bmatrix} = {X \times W}$

## Training a Linear Regression Model
### Formula
$
X \times W = y \\
$

$
(X^T \times X)^{-1} \times X^T \times X \times W = (X^T \times X)^{-1} \times X^T \times y
$  

$
W = (X^T \times X)^{-1} \times X^T \times y
$

***Attention***   
***Add 1 to each record in Maxtrix $X$***

## Car Price Baseline Model
Linear regression only applies to numerical varibales.  

Weight array $W$ calculated in previous step can be splitted to two parts: $w_{0}$ and $w$.   

Use the weight array $W$ to calculate predition: $g(X) = w_{0} + X \times W$
## RMSE
Root Mean Squared Error (RMSE) is a way to evaluate regression models. It calculates the average value of differences between target values and predicted values.  

$
RMSE = \sqrt{\frac{1}{m} \sum_{i=1}^{m} (g(x_i) - y_i)^2}
$  
- $g(x_{i})$ is the predicted value
- $y_{i}$ is the target value
- $m$ is the number of observations in dataset

## Validate Model
Apply weight array $W$ to validation dataset $X_{val}$: $g(X_{val}) = X_{val} \times W = w_{0} + X_{val} \times w$  

Evaluate model by $RMSE$

## Simple Feature Engineering
Create additional features 

## Categorical Variables
Convert categorical variables to numerical variables by one-hot encoding.  
Category A -> [1, 0, 0]  
Category B -> [0, 1, 0]  
Category C -> [0, 0, 1]

## Regularization
The root cause of having extremely large weights is noise in dataset. One situation would be some rows or coloumns in a matrix are 99% similar (e.g. 1.00001 and 1). This matrix looks like a Singular Matrix but still invertible, and the weight caculated based on this matrix is unreasonaly hight.   
To solve this issue, one alternative is adding a small decimal to the diagonal of the feature matrix. One rule of this small decimal is the larger the small decimal, the smaller the weights. 

$
\begin{bmatrix}
    1 & 1.0001 & 2.001 \\
    2.001 & 1 & 1.00001 \\
    1 & 2 & 1
\end{bmatrix}
\ + 
\begin{bmatrix}
    0.01 & 0 & 0 \\
    0 & 0.01 & 0 \\
    0 & 0 & 0.01
\end{bmatrix}
=>
\begin{bmatrix}
    1.01 & 1.0001 & 2.001 \\
    2.001 & 1.01 & 1.00001 \\
    1 & 2 & 1.01
\end{bmatrix}
$

## Tuning Model
Find the best regularization parameter (small decimal) by applying it into validation dataset. 

## Using Model
After finding the best model and regularization parameter, we train the model again using training dataset and validation dataset, and calculate predicted values and RMSE using test dataset.
### Notebook
[Car Price Model Notebook](car_price.ipynb)

## Additional Notes
### Pandas Series
Series is similar to list, but it has some features that list doesn't have.
- Types: Series can be a row or a column. 
- Index: output of series is composed by pairs of inex and value, one-to-one match.
- Vector: output of series can be vertically aligned, containing two elements - index and values.
- Element-wise operations: supports `.sum()`, `.count()`, `mean()`.  

```python
# Example of Series
df.dtype
df.columns
# Return index of Series
df.dtype.index
# Return value of Series
df.dytype.values
# Return index of DataFrame
df.columns
```

### Vector and Sample Data
Assume we have a sample dataset $(n \times d)$, $n$ is the number of records, $d$ is the amount of features.  
- $X$ represent a dataset or a matrix. 
- $x_{i}$ represents each records in this matrix $X$. 
- $X$ and $x_{i}$ are normaly stored in vector fomats.   
$
\mathbf{X} =
\begin{bmatrix}
    x_{1}^{T} \\
    x_{2}^{T} \\
    \vdots \\
    x_{n}^{T}
\end{bmatrix}
\ =
\begin{bmatrix}
    x_{11} & x_{12} & \dots  & x_{1d} \\
    x_{21} & x_{22} & \dots  & x_{2d} \\
    \vdots & \vdots & \ddots & \vdots \\
    x_{n1} & x_{n2} & \dots  & x_{nd}
\end{bmatrix}
$
$\quad$
$
{x_{i}} =
\begin{bmatrix}
    x_{i1} \\
    x_{i2} \\
    \vdots \\
    x_{id}
\end{bmatrix}
$
### Matrix Inverse
The requirement of a matrix $X (m \times n)$ has inverse matrix $X^{-1}$:  
1. square matrix (m = n)
2. rank(X) = min(m,n)
3. 行列式不为零  
### Matrix Transpose
如果$X (6 \times 3)$, 他的逆转版为$X^{-1} (3 \times 6)$, $X \times X^{-1}(6 \times 6)$则是Singular Matrix.  
因为要求$rank(X)$和$rank(X)<=min(6,3)$, 因此$rank(X \times X^{-1}) <= min(6,3)$, 而$X \times X^{-1}$是一个$6 \times 6$的matrix, 远超过$3$.  
Singular Matrix 还有一种情况是linearly dependent rows or columns. 这个的意思是, 这个Matrix里的rows之间或者columns之间, 存在相似性.  
Singular Matrix cannot be inverted.
