# Decision Trees and Ensemble Learning: Credit Risk Scoring

## Data Cleaning and Preparation
- `repalce(to_replace, value)` -> replace `to_replace` with `value`.
- `np.nan` -> numpy null value.
- `df.columns.str.lower()` -> lowercase column names.
- `df['column_name'].map(dict)` -> match key in dict to cell value in df and replace the cell value with value in dict.
- `reset_index(drop=True)` -> reset df index and remove index column

## Decision Trees
- `DictVectorizer()` -> one-hot encoding.
- `DecisionTreeClassifier()` -> decision tree model from `sklearn.tree` class.
- `max_depth` -> hyperparameter to control the maximum depth of decision tree algorithm.
- `export_text(model, feature_names)` -> method from `sklearn.tree` class to display the text report showing the conditions in each node of decision tree. 
- `get_feature_names_out()` -> method from `DictVectorizer` to display encoded feature names and can be applied in `export_text()` after casting as `list`

## Decision Trees Concepts
### Classes, Functions and Methods
- `series.value_counts(normalize=True)` -> calculate the percentage of each feature values in one series. 
### Concepts
#### Structure of Decision Tree  
A decision tree is composed **nodes** (conditions), **branches** (True or False from conditions), and **leaves** (final predicted class).  

#### Depth of Decision Tree  
The **depth** of a tree is the number of levels it has, or simply the length of the longest path from the root node to a leaf node.

#### Conditions and Thresholds  
Each condition is composed of a feature and a threshold. The learning algorithm for a decision tree involves determining the best conditions to split the data at each node in order to achieve the best possible classifier. In essence, at each node, the algorithm evaluates all possible thresholds for every feature and calculates the resulting misclassification rate. It then selects the best condition (feature and threshold) that yields the **lowest impurity**.

#### Misclassification Rate  
It's also called Impurity Rate. After each split, the goal is to divide the data into two sets that are as pure as possible, which mean each dataset should contain **only one class**. The misclassification rate is a **weighted average** of the error rates obtained after splitting the data into two sets. The predicted class for each set is determined by the **majority class** present in this set.

#### Impurity Criteria  
Common misclassification rate measurements are **GINI Impurity** and **Entropy**. It is also possible to use **MSE** for regression problems.  

#### Stopping Criteria  
The model recursively split the data at each child node. The consequence is model overfitting. The criteria below can prevent from overfitting:
- The group is already **pure**: 0% impurity.
- The **maximum** depth has been reached.
- The group is **smaller** than the **minimum** size set for groups.
-  The **maximum** number of leaves/terminal nodes has been reached.

#### Decision Tree Learning Algorithm Methodology
- Find the best condition and threshold to split data
- Stop splitting if max depth reaches
- At each node, if the dataset is sufficiently large and not pure, repeat the process

## Decision Tree Parameter Tuning
### Importance of `max_depth` and `min_samples_leaf`
#### Controlling Overfitting
These parameters play a critical role in preventing overfitting.
  - `max_depth` limits the tree's complexity, preventing it from growing too deep and memorizing the training data.
  - `min_samples_leaf` ensures that leaf nodes have a sufficient number of samples, reducing the chance of creating nodes that are too specific to the training data.
  
#### Impact on Bias and Variance
They also affect the model's bias and variance.
  - Increasing `max_depth` and decreasing `min_samples_leaf` can lead to a more complex model with lower bias but higher variance.
  - Decreasing `max_depth` and increasing `min_samples_leaf` results in a simpler model with higher bias but lower variance.

### Fine-Tuning Process
- Tune `max_depth` first and then `min_samples_leaf`.
- This method is computationally efficient for `large datasets`, though it may not be optimal for smaller ones.

### Heatmap for Visualization
Store the scores (e.g., AUC) obtained during tuning in a pivot table, and create a heatmap with seaborn to easily identify high score areas, which helps pinpoint the optimal max_depth and min_samples_leaf combination.
```python
import seaborn as sns
from matplotlib import pyplot as plt
%matplotlib inline

df_scores_pivot=df_scores. \
    pivot(index='min_samples_leaf',
    columns=['max_depth'], 
    values=['auc'])

sns.heatmap(df_scores_pivot, annot=True, fmt='.3f')
```

## Ensemble Learning and Random Forest
### Classes, Functions and Methods
- `RandomForestClassifier()` -> random forest model from `sklearn.ensemble` class. 

### What is Ensmeble Model?
**Ensmeble Model** is a machine learning paradigm where multiple models, often referred to as 'weak learners', are strategically combined to solve a particular computational intelligence problem. This approach frequently yields superior predictive performance compared to using a single model.

### What is Random Foreast?
**Decision Tree** is an example of ensemble learning where each model is a decision tree and their predictions are aggregated to identify the most popular result. Random forest only selects a random subset of features from the original data to make predictions. The **randomness** in Random Forest stems from two key aspects:

- Random subset of data: Each tree is potentially trained on a **bootstrapped** sample of the original data, introducing randomness at the row level.
- Random subset of features: At each node during tree construction, only a **random subset** of features is considered for splitting. This feature randomness helps decorrelate the trees, preventing overfitting and promoting generalization to unseen data.

Also, in Random Forest, each decision tree is trained independently.

### Boostrapping
**Bootstrapping** is a resampling technique where numerous subsets of the data are created by sampling the original data **with replacement**. This means that some data points may appear multiple times in a single bootstrap sample, while others may be excluded. In Random Forest, each decision tree is trained on a distinct bootstrap sample, further contributing to the diversity and robustness of the ensemble.

### Parameter Tuning
- `max_depth`: controls the maximum levels of each decision tree
- `n_estimators`: determines the number of trees in the forest.
- `min_samples_leaf`: define the mimimum number of samples in each leaf node. 

***Higher `max_depth` leads to higher probability of overfitting.***  
***More `n_estimators` leads to more computational costs.***

## Gradient Boosting and XGBoost
### Classes, functions, and methods
- `xgboost.DMatrix(X, label, feature_names=[List])` -> transform X dataset to XGBoost required format for better performance.
- `xgb.train(xgb_params, dtrain, num_boost_round, evals, verbose_evals)` -> train the XGBoost model with pre-defined `xgb_params` on transformed X dataset.   
- `xgb_model.predict(dval)` -> make a prediction on transformed `davl`.
- `%%capture output` -> capture "print-out" message and store it in `output` variable.
### Gradient Boost
Unlike Random Forest where each decision tree trains **independently**, in the Gradient Boosting Trees, the models are combined **sequentially**, where each model learns the prediction errors made by the previous model and improve the prediction. This process continues to `n` number of iterations, and in the end, all the predictions get combined to make the final prediction.

### XGBoost
**XGBoost** is one of the libraries which implements the gradient boosting technique.
```bash
conda install -c conda-forge xgboost 
conda install -c conda-forge llvm-openmp #xgboost依赖文件
```
#### Prepare Dataset
Transform X dataset to XGBoost required format for better performance.
```python
xgboost.DMatrix(X, 
                label, # y actual value
                feature_names=[List] #a list of `DictVectorizer-transformed` feature names.
                )
```

#### Tuning Parameters
Define more parameters in `xgb_params`
```python
xgb_params={
    'eta':0.3, # learning rate, which indicates how fast the model learns.
    'max_depth':6, # same as max_depth in Random Forest
    'min_child_weight':1, # ame as min_samples_leaf in Random Forest
    'objective':'binary:logistic', # identify model types, either regression or classification
    'eval_metric':'auc', # define evaluation metrics, like auc or log loss
    'nthread':8, # used for parallelized training.
    'seed':1, # random state
    'verbosity':1 # training warning; 0: silence mode; 1: warning
}
```
#### Train Model
Train the XGBoost model with pre-defined `xgb_params` on transformed X dataset `dtrain`.
```python
xgb.train(xgb_params, 
          dtrain, 
          num_boost_round, 
          evals, 
          verbose_evals 
          )
``` 
- `evals` provides more evaluation datasets in this format `[(dtrain, 'train'), (dval, 'val')]`. The evaluation metrics is defined at `eval_metrics` in `xgb_params`.
- `num_boost_round` defines the number of tree to grow, similar as `n_estimators` in Random Forest. 
- `verbose_evals` defines the steps of showing evaluation results. For example, if `verbose_evals=5`, it means showing evaluation results in every 5 steps. 

### Prediction
```python
model.preidct(dval)
```

## XGBoost Parameter Tuning
### `eta`
`eta`可以理解成XGBoost模型的学习率或者成长率, 这个参数很大程度影响着模型的稳定性和准确性. XGBoost中的每个sub model都是根据前一个模型的prediction error来修正自己的模型架构, 而`eta`决定了这次修正误差的幅度大小. Default = 0.3, range: [0, 1]. 
- `eta`越小, 代表修正幅度越小, 接近正确值的可能性越大, 但缺点是需要训练更多的sub models.
- `eta`越大, 代表修正幅度越大, 偏离正确值的可能性越大.

### `max_depth`
`max_depth` defines the depth of each sub model. Default = 6, range: [0, inf]

### `num_child_weight`
`num_child_weight` defines the minimum number of samples in each leaf node. Default = 1. range: [0, inf].

### Tuning Sequence
`eta` --> `max_depth` --> `num_child_weight` (这只是其中一种tuning方式)

### Other Useful Parameters
- `subsample` (default=1)  
Subsample ratio of the training instances. Setting it to 0.5 means that model would randomly sample half of the training data prior to growing trees. range: (0, 1]

- `colsample_bytree` (default=1)  
This is similar to random forest, where each tree is made with the subset of randomly choosen features.

- `lambda` (default=1)  
Also called reg_lambda. L2 regularization term on weights. Increasing this value will make model more conservative.  
L2 regularization通过约束叶子节点的权重, 使模型更保守.


- `alpha` (default=0)  
Also called reg_alpha. L1 regularization term on weights. Increasing this value will make model more conservative.  
L1 regularization也是通过约束叶子节点的权重, 使模型更保守. 但L1的约束更激进, 他会强制权重变为0, 减少feature数量.

### More about Tree Models
- XGBoost models perform better on tabular data than other machine learning models.
- XGBoost can deal with null values
- Trees can also be used for solving the regression problems: check `DecisionTreeRegressor`, `RandomForestRegressor` and the `objective=reg:squarederror parameter` for XGBoost.
- There's a variation of random forest caled "extremely randomized trees", or "extra trees". Instead of selecting the best split among all possible thresholds, it selects a few thresholds randomly and picks the best one among them. Because of that extra trees never overfit. In Scikit-Learn, they are implemented in `ExtraTreesClassifier`