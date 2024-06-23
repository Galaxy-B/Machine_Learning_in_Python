# K-Nearest Neighbors

My own implement of KNN can be found in `KNN.ipynb`, while the built-in KNN classifier of `sklearn` library is called in `sklearn_KNN.ipynb`.

## Dataset

Dataset that I pick for this model is about classifying the air quality of  one date according to the statistic of relative features.
  
In other words, the target is a multi-class classification.
  
There are **null** values in features. Therefore, you will find that I made additional operation to **drop** all **null** values while loading the dataset.

```Python
# read data from given file
def read_data(path = None):

    data = pd.read_excel(path)
    
    # drop those records which contain 0 values
    data = data.replace(0, np.NaN)
    data = data.dropna()
```

## Parameter

Finding the best K parameter is a key step in training KNN models. Here I just simply used a for loop to iterate some K values, recorded their performance and then found the best one.
  
So as to intuitively witness the trend of performance changes over K values, `matplotlib` is imported to paint a line chart for us via following code:

```Python
# configuration for plt to visualize the performance of different K values
def show(performance, Ks):
    
    plt.figure(figsize = (9, 6))
    plt.grid(True, linestyle = '-.')
    plt.xticks(Ks)
    plt.plot(Ks, performance, marker = '.')
    plt.xlabel("K")
    plt.ylabel("F1 score")
    
    best_K = Ks[performance.index(np.max(performance))]
    plt.title("F1 scores of my KNN model\n(best K = %d)" % best_K)
    plt.show()
```

## Evaluation

Again, **F1 score** is adopted to evaluate my own version of KNN model. However, it is slightly different with that in `decision_tree.ipynb`, because I called the built-in `f1_score()` method of `sklearn` library. The latter brings in much more details.