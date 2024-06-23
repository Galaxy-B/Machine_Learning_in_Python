# Support Vector Machine

My own implement of SVM can be found in `SVM.ipynb`, while the built-in KNN classifier of `sklearn` library is called in `sklearn_SVM.ipynb`.

## Dataset

Dataset that I pick for this model intends to predict whether a customer has lost interest in a restaurant according to given features.

However, original features make little contribution to the prediction. Therefore, you will find a new method `wrangle()` that I defined before `read_data()`. This method will preprocess the origin data to extract those useful information and form a new data table.

```Python
# wrangle the given data file and write the result into a new file
def wrangle(file_name = None):
    
    old_path = file_name + '.csv'
    new_path = file_name + '_wrangled.csv'
    
    old_data = pd.read_csv(old_path, encoding = 'gbk')
    ...
    # write the wrangled data into a new file
    new_data.to_csv(new_path, index = False, encoding = 'gbk')
```

Wrangled data will be stored in `*_wrangled.csv` after you run the program.
  
## Parameter

There are many kinds of **kernel functions** executing the core logic of the support vector machine. Each kernel function has its own list of parameters.

To reduce the workload of adjusting parameter, I imported `GridSearchCV` to automatically test all combination of given parameter values.

Take **Polynomial** function as an example. `degree` and `coef` are two of its key parameters.

```Python
# use grid search to find the best parameter for poly function
SVM_classifier = SVC(kernel = 'poly', decision_function_shape = 'ovo')
param_grid = {'C':[1.0], 'degree':np.arange(2, 6), 'coef0':np.arange(0, 10)}

algo = GridSearchCV(estimator = SVM_classifier, param_grid = param_grid, cv = 10)
algo.fit(X_train, Y_train)
```

## Evaluation

The built-in method `score()` of SVM classifier takes the responsibility of evaluating the model. It will provide the mean accuracy on test set of the model.