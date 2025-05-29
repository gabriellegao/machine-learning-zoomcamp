# Classification: Churn Projection Project

## Intro
This project apply data to binary classification, and output liklihood as predicted values between 0 and 1.  
$$g(x_{i})=y_{i}$$

## Data Preparation
Commands, functions, and methods:
- `!wget` -> Linux shell command for downloading data
- `pd.read.csv()` -> read csv files
- `df.head()` -> take a look of the dataframe
- `df.head().T` -> take a look of the transposed dataframe
- `df.columns` -> retrieve column names of a dataframe
- `df.columns.str.lower()` -> lowercase all the letters in the columns names of a dataframe
- `df.columns.str.replace(' ', '_')` -> replace the space separator in the columns names of a dataframe
- `df.dtypes` -> retrieve data types of all series
- `df.index` -> retrieve indices of a dataframe
- `pd.to_numeric()` -> convert a series values to numerical values. The errors='coerce' argument allows making the transformation despite some encountered errors.
- `df.fillna()` -> replace NAs with some value
- `(df.x == "yes").astype(int)` -> convert x series of yes-no values to numerical values.

## Setup Validation Framework
Classes, functions, and methods:
- `train_test_split` -> Scikit-Learn class for splitting a dataset into two parts. The test_size argument states how large the test set should be. The random_state argument sets a random seed for reproducibility purposes.
- `df.reset_index(drop=True)` -> reset the indices of a dataframe and delete the previous ones.
- `df.x.values` -> extract the values from x series.
- `del df['x']` -> delete x series from a dataframe.
  
## EDA
Functions and methods:
- `df.isnull().sum()` -> returns the number of null values in the dataframe.
- `df.x.value_counts()` -> returns the number of values for each category in x series. The `normalize=True` argument retrieves the percentage of each category. In this project, the mean of churn is equal to the churn rate obtained with the value_counts method.
- `round(x, y)` -> round an x number with y decimal places
- `df[x].nunique()` -> returns the number of unique values in x series

## Feature Importance: Churn Rate and Risk Ratio
1. Churn rate
   Difference between global mean of the target variable and mean of the target variable for categories of a feature. If this difference is greater than 0, it means that the category is less likely to churn, and if the difference is lower than 0, the group is more likely to churn. The larger differences are indicators that a variable is more important than others.

2. Risk ratio
   Ratio between mean of the target variable for categories of a feature and global mean of the target variable. If this ratio is greater than 1, the category is more likely to churn, and if the ratio is lower than 1, the category is less likely to churn. It expresses the feature importance in relative terms.

Functions and methods:

- `df.groupby('x').y.agg([mean()])` -> returns a dataframe with mean of y series grouped by x series
- `display(x)` -> displays an output in the cell of a jupyter notebook.

## Feature Importance: Mutal Information
Mutual Information measures the "correlation" between categorical variables and target variables. The higher the score, the stronger the correlation.

Classes, functions, and methods:
- `mutual_info_score(x, y)` -> Scikit-Learn class for calculating the mutual information between one x target variable and one y feature. This functions is more suitable for categorical variables and discretized numerical variables. 
- `df[x].apply(y_func)` -> apply a y function to the x series of the df dataframe. This `apply()` method is similar as `map()` in RDD.  
- `df.sort_values(ascending=False).to_frame(name='x')` -> sort values in an ascending order and called the column as x.

## Feature Importance: Correlation
Correlation coefficient measures the degree of dependency between two variables. This value is negative if one variable grows while the other decreases, and it is positive if both variables increase. Depending on its size, the dependency between both variables could be low, moderate, or strong. It allows measuring the importance of numerical variables.  

If r is correlation coefficient, then the correlation between two variables is:  
- **LOW** when r is between [0, -0.2) or [0, 0.2)
- **MEDIUM** when r is between [-0.2, -0.5) or [2, 0.5)
- **STRONG** when r is between [-0.5, -1.0] or [0.5, 1.0]  

Positive Correlation vs. Negative Correlation
- When r is positive, an increase in x will increase y.
- When r is negative, an increase in x will decrease y.
- When r is 0, a change in x does not affect y.  

Functions and methods:
- `df[x].corrwith(y)` -> returns the correlation between x and y series. This is a function from pandas.  

***Attention***  
***Mutual Information and Correlation are two ways of measuring the importance of feature variables to target variables. Mutual information is suitable for categorical values and correlation is suitable for numerical values.***  

## One-Hot Encoding
One-Hot Encoding allows encoding categorical variables in numerical ones. This method represents each category of a variable as one column, and a 1 is assigned if the value belongs to the category or 0 otherwise.

Classes, functions, and methods:
- `df[x].to_dict(orient='records')` -> convert x series to dictionaries, oriented by rows.
- `DictVectorizer().fit_transform([x_dict])` -> Scikit-Learn class for one-hot encoding by converting a list of x dictionaries into a sparse matrix. It does not affect the numerical variables.
- `DictVectorizer().get_feature_names_out()` -> return the names of the columns in the sparse matrix.

## Logistics Regression: Concept
There are two types of classification: binary and multi-class.  
In general, supervised models can be represented with this formula:
$$g(x_{i}) = y_{i}$$
Depending on what is the type of target variable, the supervised task can be regression or classification (binary or multiclass). Binary classification tasks can have negative (0) or positive (1) target values. The output of these models is the **probability** of $x_{i}$ belonging to the **positive class**.

Logistic regression is similar to linear regression because both models take into account the bias term and weighted sum of features. The difference between these models is that the output of linear regression is a **real number**, while logistic regression outputs a value between **zero(0) and one(1)**, applying the sigmoid function to the linear regression formula.  
$$g(x_{i}) = Sigmoid(w_{0} + w_{1}x_{1} + w_{2}x_{2} + ... + w_{n}x_{n})$$
$$z = w_{0} + w_{1}x_{1} + w_{2}x_{2} + ... + w_{n}x_{n}$$
$$Sigmoid(z)=\frac{1}{1 + exp( -z )}$$
In this way, the sigmoid function allows transforming a score ($z$) calculated from linear regression into a probability ($Sigmoid(z)$).

## Train Logistics Regression with Scikit-Learn
Classes, functions, and methods:
- `LogisticRegression().fit(x)` -> Scikit-Learn class for training the logistic regression model.
- `LogisticRegression().coef_[0]` -> return the coefficients or weights of the LR model
- `LogisticRegression().intercept_[0]` -> return the bias or intercept of the LR model
- `LogisticRegression().predict[x]` -> make predictions on the x dataset, hard prediction.  
- `LogisticRegression().predict_proba[x]` -> make predictions on the x dataset by returning two columns with their probabilities for the two categories (0 and 1), soft predictions.  

### Accuracy Rate:  
Compare target values and predicted values, if they are the same, return True; otherwise, return False.  
$$\frac{Numbers\quad of\quad True}{Numbers\quad of\quad True + Numbers\quad of\quad False}$$

## Model Interpretation
Classes, functions, and methods:
- `zip(x,y)` -> returns a new list with elements from x joined with their corresponding elements on y  
- `LogisticRegression().intercept_[0]` -> returns $w0$.  
- `LogisticRegression().coef_[0]` -> returns a list of weights ($w$).  
- `dict(zip(dv.get_feature_names_out(),w.round(3)))` -> `dv` is an instance of class `DictVectorizer()` and `w` is coefficients of LR model. This command returns a 1-to-1 match of feature names and their corresponding weights.  

The fundamental logic of binary logistic regression involves calculating a score for each record using assigned weights. This score is then applied to a sigmoid function to determine its probability.  

## Using Model
We trained the logistic regression model with the full training dataset (training + validation), considering numerical and categorical features. Thus, predictions were made on the test dataset, and we evaluated the model using the accuracy metric.
- Apply `DictVectorizer().fit_tranform()` and `LogisticsRegression().fit()`on full training dataset.  
- Apply `DictVectorizer().tranform()` and `LogisticsRegression().predict_proba()` on test datatset.  
- Calculate `(y_pred == y_test).mean()` to determine accruracy of the mode. 