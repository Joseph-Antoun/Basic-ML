# Project1

## Folder Structure

    .
    ├── src             # Python source files
    ├── data            # Original datasets (contains wine only for now)
    ├── img             # Images generated (exploratory data analysis) by the code in /src
    ├── notebooks	# Jupyter Notebooks - not required
    └── README.md

>  I run Linux but the code should work on any system normally

## TODO

- check for correlation
- create one class per model (LDA and LR)
- in the init() of each class
   - define #features, parameters, percentage(train test validation),
- train test validation
- gradient descent for LDA only
- create fit() function, params(X,y, gradient descent iretarions)
- create predict()
- evaluate_acc()
- add validation()
- add function to compute runtime
- add function to generate subsets of features for the wine dataset
