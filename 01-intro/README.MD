# Intro: Basic Concepts
## What is Machine Learning?
Machine learning is a process of extracting patterns from data, including two types:
- features
- targets

## ML vs Rule-Based Systems
### Rule-Based Systems: Spam Email
Identify spam emails by keywords, email length, etc.  
The biggest disadvantage of rule-based system is that it's unable to track "upgraded" spam emails on real time and the complexity of code maintenance increases as system grows.  

### Machine Learning: Spam Email
#### Get Data
Collect spam emails and non-spam emails

#### Define and Calculate Features
Use conditions from rule-based systems to define features of ML model.  
Encode features and target to binary.

#### Train and Use Model
Input features, target and data to model.   
Output probabilities as predictions and define a threshold to clarify emails as spam or non-spam.  

## Supervised Machine Learning
### Basic Theory
- Feature matrix (X): made of obersations or obejcts (rows) and features (columns).
- Target variable (y): a vector (one-dimension array) of predictions. 
- Model (g): input X and output predicted y. Predict predicted y as close as possible to actual y (target variable), which is traning.
- Each row of X matches each y.
- Formula: $\mathbf {g(X) = y}$

### Types of SML
- Regression: numerical output
- Classification
  - Binary: 1 or 0, two categories
  - Multiclass: more than two categories
- Ranking: output top scores associated with corresponding items. It's applied in recommender systems.

## CRISP-DM
CRISP-DM stands for Cross-Industry Standard Process for Data Mining.
- Business understanding
- Data understaning
- Data Preparation
- Modeling
- Evaluation
- Deployment
- Iterate

## Model Selection 
### Multiple Comparisons Problem (MCP)
- Split datasets to tranning (60%), validation (20%), and test (20%)
- Train models (e.g. logistics regresion, decision tree, neural network)
- Evaluate models by metrics (e.g. accuracy)
- Select the best model 
- Apply best model to test datatest
- Compare performance metrics of validation and test

## Create Environment for Course
### Create Environment
```bash
conda create -n machine-learning-zoomcamp python=3.11
```

### Activate Environment
```bash
conda activate machine-learning-zoomcamp
```

### Install Libraries
```bash
conda install numpy pandas sciki-learn seaborn jupyter
```

## Numpy
### Notebook
[Numpy Notebook](numpy_tutorial.ipynb)

## Linear Algebra
### Vector
Vector = Column, Array = Row
### Row Vector
Transpose column vector to row vector    
$vecotr^T$

### Vector Operation
$vector1 = [2,4,5,6]$  
$vector2 = [1,0,0,2]$  
$vector1 + vector2 = [2+1, 4+0 + 5+0 + 6+2] = [3,4,5,8]$ 

### Vector-Vector Multiplication
$vector1 = [2,4,5,6]$  
$vector2 = [1,0,0,2]$  
$vector1 * vector2 = 2*1 + 4*0 + 5*0 + 6*2 = 14$

### Matrix-Vector Multiplication 
Formula $\mathbf{U *v}$  

$
\mathbf{U *v} =
\begin{bmatrix}
u_{0}^{T} * v \\
u_{1}^{T} * v \\
u_{2}^{T} * v
\end{bmatrix}
$  

Matrix $\mathbf{U}$   

$
\mathbf{U} =
\begin{bmatrix}
2 & 4 & 5 & 6 \\
1 & 2 & 1 & 2 \\
3 & 1 & 2 & 1
\end{bmatrix}
$  

Column vector $\mathbf{v}$  

$
\mathbf{v} =
\begin{bmatrix}
1 \\
0.5 \\
2 \\
1
\end{bmatrix}
$  

*Attention*  
*The row length of matrix should be the same as the length of vector*

### Matrix-Matrix Multiplication
Formula  
Matrix $\mathbf{U * V}$   

$
\mathbf{U * V} =
\begin{bmatrix}  
U * V_{0}  , U * V_{1} , U * V_{2} 
\end{bmatrix}
$  

$
\mathbf{U} =
\begin{bmatrix}
2 & 4 & 5 & 6 \\
1 & 2 & 1 & 2 \\
3 & 1 & 2 & 1
\end{bmatrix}
$  

Matrix $\mathbf{V}$   

$
\mathbf{V} =
\begin{bmatrix}
1 & 1 & 2 \\
0 & 0.5 & 1 \\
0 & 2 & 1 \\
2 & 1 & 0
\end{bmatrix}
$  

Column vector $\mathbf{V_0, V_1, V_2}$  

$
\mathbf{V_0} =
\begin{bmatrix}
1 \\
0 \\
0 \\
2
\end{bmatrix}
$
$
\mathbf{v_1} =
\begin{bmatrix}
1 \\
0.5 \\
2 \\
1
\end{bmatrix}
$
$
\mathbf{V_2} =
\begin{bmatrix}
2 \\
1 \\
1 \\
0
\end{bmatrix}
$  

*Attention*  
*The row length of matrix U should be same as the column length of matrix V*  
### Identity Matrix
Formula $\mathbf{U * I = U}$  

$
\mathbf{I} =
\begin{bmatrix}
1 & 0 & 0 & 0 \\
0 & 1 & 0 & 0 \\
0 & 0 & 1 & 0 \\
0 & 0 & 0 & 1
\end{bmatrix}
$  

### Matrix Inverse
The requirement of a matrix $A (m \times n)$ has inverse matrix $A^{-1}$:  
1. m = n
2. rank(A) = min(m,n)
3. 行列式不为零  
   
####  Formula  
$\mathbf{Matrix U * Matrix U^{-1} = I}$  

### Notebook
[Linear Algebra and Matrix Notebook](linear_algebra.ipynb)

## Pandas
### Notebook
[Pandas Notebook](pandas.ipynb)