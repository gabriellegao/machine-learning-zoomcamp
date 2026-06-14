# CS50AI大纲

## Logic/Knowledge Representation
| 概念              | 你需要掌握什么                               |
| --------------- | ------------------------------------- |
| Symbol          | 用一个变量代表一个命题，比如 `P = It is Tuesday`    |
| Implication     | `P => Q`，如果 P 成立，那么 Q 成立              |
| Entailment      | `KB ⊨ α`，知识库是否必然推出结论                  |
| And / Or / Not  | 基础逻辑连接词                               |
| De Morgan’s Law | `not(P or Q) = not P and not Q`       |
| CNF             | 把逻辑句子转成一组 OR clauses 的 AND            |
| Resolution      | 用 `P or Q` 和 `not P or S` 推出 `Q or S` |
| Model Checking  | 枚举所有可能世界，看结论是否总成立                     |

## Probability and Bayes' Rule
| 概念                        | 含义                       |     |
| ------------------------- | ------------------------ | --- |
| Unconditional probability | 不带条件的概率，比如 `P(A)`        |     |
| Conditional probability   | 带条件的概率，比如 `P(A\|B)` |
| Joint probability         | 两件事同时发生的概率，比如 `P(A ∩ B)` |     |
| Bayes’ Rule               | 根据观察到的证据反推原因的概率          |     |

### 公式  
- `P(A ∩ B) = P(A | B)P(B)`  
- `P(A ∩ B) = P(B | A)P(A)`  
这两个公式本质是在计算`A`和`B`同时发生的概率. 一个是从A的角度去算,一个是从B的角度去算.

## Search/Optimization
| 概念                      | 含义              |
| ----------------------- | --------------- |
| Hill Climbing           | 一直往更好的邻居走       |
| Local Maximum / Minimum | 局部最优，但不一定是全局最优  |
| Simulated Annealing     | 一开始允许走差路，后期逐渐稳定 |
| Temperature             | 控制接受差解的概率       |
| Optimization            | 在众多解中寻找最优或较优解   |

## Machine Learning Basics
| 概念                    | 含义                    |
| --------------------- | --------------------- |
| Supervised Learning   | 有 label 的学习           |
| Unsupervised Learning | 没有 label，让模型自己找结构     |
| KNN                   | 根据最近的 K 个邻居分类         |
| K-means               | 把无标签数据分成 K 个 cluster  |
| Loss Function         | 衡量模型预测错了多少            |
| Regularization        | 防止模型过拟合               |
| Gradient Descent      | 通过下降 loss 来优化参数       |
| Learning Rate         | 控制每次参数更新的步长           |
| Optimizer             | 实际更新参数的方法，比如 SGD、Adam |

## Neural Networks
| 概念                       | 含义                     |
| ------------------------ | ---------------------- |
| Neuron / Unit            | 接收输入，计算输出              |
| Weight                   | 每个输入的重要程度              |
| Bias                     | 调整神经元输出                |
| Activation Function      | 引入非线性，比如 sigmoid、ReLU  |
| Dense Layer              | 每个 neuron 和上一层所有输出相连   |
| Backpropagation          | 计算每个参数对 loss 的影响       |
| Optimizer                | 根据 gradient 更新 weights |
| Overfitting              | 模型太会背训练数据              |
| Dropout / Regularization | 减少过拟合                  |


## CNN
| 概念                  | 含义                       |
| ------------------- | ------------------------ |
| Convolutional Layer | 用 filters 提取图片中的 pattern |
| Filter / Kernel     | 一组可学习 weights            |
| Feature Map         | filter 扫图后得到的响应图         |
| Pooling Layer       | 压缩 feature map           |
| Max Pooling         | 每个区域取最大值                 |
| Flatten             | 把 feature maps 拉平成一维     |
| Dense Layer         | 根据提取到的 features 做最终分类    |

## Natrual Language
| 概念                                            | 含义                                                                                                                                                                                                                    |
| --------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| NLP / Language                                | 让 AI 处理和理解人类语言，比如文本分类、情感分析、翻译和文本生成                                                                                                                                                                                    |
| Token / Tokenization                          | Token 是文本被拆分后的基本单位；Tokenization 是把文本拆成 words、subwords 或 punctuation 的过程                                                                                                                                               |
| Sequence / Markov Chain                       | Sequence 是有顺序的一串 values，比如一句话中的 tokens；Markov Chain 是一种处理 sequence 的模型，用当前或前面的 token 预测下一个 token                                                                                                                      |
| Text Classification                           | 把一段 text 分配到某个类别，比如 spam 或 not spam                                                                                                                                                                                   |
| Sentiment Analysis                            | 一种 text classification，用于判断文本是 positive、negative 或 neutral                                                                                                                                                            |
| Bag-of-Words                                  | 把文本表示成一组 words，主要关注哪些 words 出现，通常忽略 word order 和上下文                                                                                                                                                                   |
| Naive Bayes Classifier                        | 一种基于 Bayes’ Rule 的分类算法。在 NLP 和 sentiment analysis 中，它根据 message 中的 words 判断文本更可能属于哪个类别                                                                                                                                |
| Naive Assumption                              | 假设在类别已知的情况下，message 中的 words 彼此独立，从而简化计算                                                                                                                                                                              |
| Vector / One-Hot / Distributed Representation | Vector representation 是把文字转换成数字 vector；one-hot representation 是其中一种方式，每个 word 用一个位置为 1、其他位置为 0 的 vector 表示，但不能体现 words 之间的关系；distributed representation 也是一种 vector representation，它用多个连续数值表示 word，并能体现 words 之间的语义关系 |
| Word2Vec                                      | 一种学习 distributed word vectors 的算法，通过预测 target word 周围可能出现的 context words，学习 words 之间的语义关系                                                                                                                             |
| Encoder–Decoder                               | Encoder 把 input sequence 转换成内部 representations；Decoder 根据这些 representations 和之前的 outputs 生成 output sequence                                                                                                           |
| Attention                                     | Decoder 在生成当前 output word 时，判断 encoder 的哪些 input representations 最重要                                                                                                                                                  |
| Self-Attention                                | Sequence 中的每个 token 查看同一 sequence 中的其他 tokens，从而根据上下文更新自己的 representation                                                                                                                                             |
| Positional Encoding                           | 给 word vector 加入位置信息，让 Transformer 知道 tokens 在 sequence 中的顺序                                                                                                                                                          |
| Transformer Encoder                           | 并行处理 input tokens，通过 positional encoding、self-attention 和 neural network 生成每个 token 的 encoded representation                                                                                                          |
| Transformer Decoder                           | 根据之前生成的 output tokens，同时关注 output context 和 encoder representations，预测下一个 output token                                                                                                                                |
| Transformer                                   | 一种以 attention 为核心的 neural network architecture，可以同时处理多个 tokens，而不依赖传统 RNN 的循环处理方式                                                                                                                                     |

### 公式推导
Naive Bayes作用于Sentiment Analysis的公式推导, `P(positive | message)` and `P(negative | message)` 是基于message推导出不同的情感概率.  

$$P(positive | message) = \cfrac {P(message | positive) * P(positive)}{P(message)}$$
因为在比较`positive` and `negative`时分母$P(message)$相同, 可以忽略. 可推导出公式两端是成比例的   
$$P(positive | message) = ∝ P(message | positive) * P(positive)$$
Naive Bayes 假设 message 里的 words 彼此独立，所以：
$$
P(positive | message)
= P(positive)
* P(word1 | positive)
* P(word2 | positive)
* ...
* P(wordn | positive)
$$
在Probabilities相乘运算中,可能会出现某个Probability为零的现象,在此引入`Laplace Smoothing`概念,给每个value都加1,避免0值的出现.


### Transformer
不同于RNN逐层处理token, Transformer可以依靠`self-attnetion`和`positional encoding`同时处理多个tokens.  
Transformer is composed of two parts - `Encoder` and `Decoder`.               

#### Part1: Encoder
<center>Input words</center>
<center>↓</center>
<center>Token Vector + Positional Encoding</center>
<center>↓</center>
<center>Self-Attention</center>
<center>↓</center>
<center>Feed-forward Network</center>
<center>↓</center>
<center>Encoded Representation</center>

Self-attention计算的每个token之间的关联性, token value, position和token之间的关联性这些信息会被喂进Network中,而Encoder Network会加工组合这些碎片信息,并且总结出规律.最后所有的信息和规律都会被封装进Encoded Representations.

#### Par2: Decoder
<center>Previous Output Word + Positional Encoding</center>
<center>↓</center>
<center>Marked Self-Attention</center>
<center>↓</center>
<center>Cross-Attention</center>
<center>↓</center>
<center>Feed-forward Network</center>
<center>↓</center>
<center>Predict Token</center>

参考之前Predicted Token和现在的位置信息, 在Masked Self-Attention环节中理解目前已知的上下文以及token之间的关联性.Cross-Attention是调取Encoder环节中生成的Encoded Representation寻找当前位置所需的信息.这些信息会被放入Network中用来预测下一个token.