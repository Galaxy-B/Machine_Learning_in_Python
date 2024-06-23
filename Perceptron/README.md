# Single Layer Perceptron

My own implement of Perceptron can be found in `perceptron.ipynb`, while the built-in Perceptron classifier of `sklearn` library is called in `sklearn_perceptron.ipynb`.

## Dataset

I choose the built-in **Iris** dataset of `sklearn` library which was initially raised by *Ronald Fisher* as the target of Perceptron model.
  
Dataset is loaded via following code:

```Python
# load the iris dataset from sklearn
iris = load_iris()

# convert the dataset into a dataframe
data = pd.DataFrame(iris.data, columns = iris.feature_names)
data['label'] = iris.target
```

## Evaluation

`matplotlib` library is imported to visualize original data and the classification line. You will get a scatter plot after running the following code through Jupyter Notebook:

```Python
# visualize the result line
plt.plot(xs, ys, 'r', label = 'classification line')

# visualize original data
plt.plot(data[:50, 0], data[:50, 1], 'o', color = 'blue', label = 'setosa')
plt.plot(data[50:100, 0], data[50:100, 1], 'o', color = 'orange', label = 'versicolor')
plt.xlabel('petal length')
plt.ylabel('petal width')
plt.legend()
plt.show()
```