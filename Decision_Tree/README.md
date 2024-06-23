# Decision Tree

I choose **C4.5** algorithm, which takes **information gain ratio** as the key factor, to build up my version of decision tree.
  
Detailed implement can be found in `decision_tree.ipynb`, while the built-in decision tree classifier of `sklearn` library is called in `sklearn_decision_tree.ipynb`.

## Dataset

Dataset that I pick for this model mainly focus on deciding whether to loan to someone according to given features.

There is a **continuous** value `revenue` in features. Therefore, you will find that I spent more time on **discretizing** this feature value while loading the dataset.

```Python
# read data and feature name from excel file
def read_data(path = None):
    df = pd.read_excel(path)

    # delete column 'nameid'
    df = df.drop(['nameid'], axis = 1)

    # discretize feature 'revenue'
    threshold = [0,10000,20000,30000,40000,50000]
    df['revenue'] = pd.cut(df['revenue'], threshold, labels = False)
```

## Evaluation

In my own version of decision tree model, only **F1 score** is adopted to evaluate the model. 
  
However, in another file I introduced `graphviz` to visualize the decision tree that we built based on the given train set. It is done via following code:

```Python
# paint the decision tree we've got
dot_data = export_graphviz(
            dt_model,
            out_file = None,
            feature_names = features,
            class_names = labels,
            rounded = True,
            filled = True,
            special_characters = True)

# in case there is shade on the image
dot_data = dot_data.replace('\n', '')
graph = pydotplus.graph_from_dot_data(dot_data)
Image(graph.create_png())
```