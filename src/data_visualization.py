# coding: utf-8
import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Seaborn styling for the plots
import seaborn as sns
sns.set()

# Import custom data cleaning module
import data_cleaning


def plot_dist_by_category(df, x, y_cat, y_cont, img_name):
    
    fig = plt.figure(figsize=(16,4))
    
    #------------------------------------------------------------
    # First plot : P(Xi|Y=0) versus P(Xi|Y=1)
    #------------------------------------------------------------
    plt.subplot(1, 2, 1)
    
    pos_class   = df[df[y_cat] == 1.0] # positive class 
    neg_class   = df[df[y_cat] == 0.0] # negative class
    pos_mean    = float(np.mean(pos_class[[x]])) # mean, positive class
    neg_mean    = float(np.mean(neg_class[[x]])) # mean, negative class
    
    sns.distplot(pos_class[[x]], color='#7282ff')
    sns.distplot(neg_class[[x]], color='#e56666')
    plt.axvline(pos_mean, color='#7282ff')
    plt.axvline(neg_mean, color='#e56666')
    fig.legend(labels=['%s for y=1' % x,'%s for y=0' % x], loc='upper center')
    
    #------------------------------------------------------------
    # Seconf plot : Scatterplots
    #------------------------------------------------------------
    plt.subplot(1, 2, 2)
    sns.scatterplot(x=x, y=y_cont, data=df)
    # Horizontal line that separates good and bad quality wine
    plt.axhline(y=6.0, color='r')
 
    plt.savefig("../img/distributions/%s.png" % img_name)
    plt.close()
    print("Created %s.png" % img_name)

   

def draw_boxplots(df, img_name):
    """
    Boxplots columns in columns_to_clean
    """
    plt.figure(figsize=(16, 4))
    df.boxplot()
    plt.savefig("../img/boxplots/%s.png" % img_name)
    plt.close()
    print("Created %s.png" % img_name)


def draw_histogram(df, img_name, bins=50, width=12, height=5):
    """
    Create a histogram in ../img/histograms
    """
    df.hist(bins=bins, figsize=(width,height))
    plt.savefig("../img/histograms/%s.png" % img_name)
    plt.close()
    print("Created %s.png" % img_name)


def draw_correlations(df, img_name):
    """
    Create a color-coded correlation matrix
    """
    plt.figure(figsize=(16, 16))
    fig, ax = plt.subplots()
    sns.heatmap(df.corr(method='pearson'), annot=True, fmt='.2f',
            cmap=plt.get_cmap('coolwarm'), cbar=False, ax=ax)
    plt.savefig("../img/correlations/%s.png" % img_name)
    plt.close()
    print("Created %s.png" % img_name)


def data_analysis_plots(clean_data, x_vars, y_vars):

    # Boxplots
    # Sulfur variables have much higher values than the rest, so we seperated them
    sulfur_vars = ['free sulfur dioxide', 'total sulfur dioxide']
    other_vars  = [v for v in clean_data.columns if not v in sulfur_vars]

    draw_boxplots(clean_data[sulfur_vars], "sulfur_variables")
    draw_boxplots(clean_data[other_vars], "other_variables")

    # Histograms
    # Separate the input variables from the output variables
    draw_histogram(clean_data[x_vars], "x_vars", bins=50, width=16, height=10)
    draw_histogram(clean_data[y_vars], "y_vars", bins=10, width=12, height=3)
    
    # Distribution plots
    # P(Xi | Y=0) and P(Xi | Y=1)
    for xi in x_vars:
        plot_dist_by_category(clean_data, xi, 'y', 'quality', img_name=xi.replace(' ', '_'))    
    
    # Correlation plot
    draw_correlations(clean_data[x_vars + ['y']], "correlations")
   


def plot_predictions_results(model, X_test, y_test, custom_pred, sklearn_pred, img_name):
    """
    plots the predictions of our custom classifier versus
    the ones from the sklean library, as well as the actual
    y values
    """
    fig = plt.figure(figsize=(8,5))

    pred = custom_pred.flatten()
    tp, tn, fp, fn = [], [], [], []

    for i in range(X_test.shape[0]):
        if y_test[i] == 1: 
            # Positive class
            if pred[i] == 1:
                tp.append((X_test[i,1], X_test[i,2])) # True positive
            else:
                fn.append((X_test[i,1], X_test[i,2])) # False negative
        else:
            # Negative class
            if pred[i] == 1:
                fp.append((X_test[i,1], X_test[i,2])) # False positive
            else:
                tn.append((X_test[i,1], X_test[i,2])) # True negative

    # Convert the lists of tupples into numpy arrays for visualization
    tp = np.array(tp, dtype='float')
    tn = np.array(tn, dtype='float')
    fp = np.array(fp, dtype='float')
    fn = np.array(fn, dtype='float')

    sns.scatterplot(x=tp[:,0], y=tp[:,1])
    sns.scatterplot(x=fp[:,0], y=fp[:,1])
    sns.scatterplot(x=tn[:,0], y=tn[:,1])
    sns.scatterplot(x=fn[:,0], y=fn[:,1])

    # Show probability threshold area
    nx, ny = 200, 100
    x_min, x_max = plt.xlim()
    y_min, y_max = plt.ylim()
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, nx),
                         np.linspace(y_min, y_max, ny))
    Z = model.predict_proba(np.c_[xx.ravel(), yy.ravel()])
    Z = Z[:, 1].reshape(xx.shape)
    plt.pcolormesh(xx, yy, Z, cmap='red_blue_classes', norm=colors.Normalize(0., 1.), zorder=0)
    plt.contour(xx, yy, Z, [0.5], linewidths=2., colors='white')

    fig.legend(labels=['True Positives', 'False Positives', 'True Negatives', 'False Negatives'], loc='upper left')
    plt.title('Linear Discriminant Analysis')
    plt.savefig("../img/models/%s" % img_name)
    plt.close()
    print("Created ../img/models/%s" % img_name)

    

def main():

    # Load & clean the data
    file_path   = '../data/wine/winequality-red.csv'
    raw_data    = pd.read_csv(file_path, delimiter=';')
    clean_data  = data_cleaning.get_clean_data(raw_data, verbose=False)

    
    # this is to skip SettingWithCopyWarning from Pandas
    clean_df   = data_cleaning.get_clean_data(raw_data, verbose=False)
    clean_data = clean_df.copy()
    # Create the binary y column
    clean_data['y'] = np.where(clean_df['quality'] >= 6.0, 1.0, 0.0)
    
    # separate x and y variables
    y_vars = ['quality', 'y']
    x_vars = [var for var in clean_data.columns.tolist() if not var in y_vars]

    # Grenerate the same plots as in the jupyter notebook
    data_analysis_plots(clean_data, x_vars, y_vars)

    
if __name__ == "__main__":
    main()


