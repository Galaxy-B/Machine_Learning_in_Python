# Classic Machine Learning Models in Python

This project achieves some of classic machine learning models in python and specifically evaluated their performance with open source datasets.

Except for my own implementation, I also provide the code for calling models in `sklearn` library to achieve the same goal. The latter is of much higher speed without question.  
  
In order to improve the model's behavior under a low-capacity dataset, I introduce **[K-Fold Cross Validation](https://blog.csdn.net/weixin_40583722/article/details/120416029 "introduction in chinese")** to reducing the influence of data partition. Check it in source code like:

```Python
# 8-fold cross validation: divide train set into train and valid (7:1)
kf = KFold(n_splits = 8, shuffle = False)

for train_index, valid_index in kf.split(X_train):
    KX_train, KX_valid = X_train[train_index], X_train[valid_index]
    KY_train, KY_valid = Y_train[train_index], Y_train[valid_index]
```

## Achieved Models

* Single Layer Perceptron

* Decision Tree in C4.5 Algorithm

* K-Nearest Neighbors

* Support Vector Machine 

## Data Wrangling

Much effort gets taken to load data and wrangle them, including dropping useless columns, replacing NaN or 0 values, dealing with present features and so on. You may find the code of this part in functions like:

```Python
# read data from given file
def read_data(path = None):
    
    df = pd.read_excel(path)
    ...
    return dataset, features
```

```Python
# wrangle the given data and write the result into a new file
def wrangle(file_name = None):

    old_data = pd.read_csv(old_path, encoding = 'gbk')
    ...
    new_data.to_csv(new_path, index = False, encoding = 'gbk')
```

## Training

This project uses open source dataset to train the model. You can check the raw data in `train.*` and `test.*` files under each directory.

## Evaluation

Basically, models are evaluated by **[F1 score](https://baike.baidu.com/item/F1%E5%88%86%E6%95%B0/13864979?fr=ge_ala "definition in chinese")** indicator except in `Support Vector Machine` where the model gets evaluated through the built-in method `score()` of `GridSearchCV` package, which reports the mean accuracy of the model instead.