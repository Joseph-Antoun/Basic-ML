import numpy as np
import matplotlib.pyplot as plt


def header_to_array(file_path):
    # Read the header
    with open(file_path, 'r') as f:
        for row in f:
            header = row
            break
    header = [var.replace('\"','') for var in header.strip().split(';')]
    return header
    
    
def load_data(file_path):
    
    # Read the numerical data into a numpy array
    raw_data = np.genfromtxt(file_path, delimiter=';', skip_header=1)
    
    y_column = np.full([raw_data.shape[0], 1], np.nan) # Create y column filled with nan
    raw_data = np.append(raw_data, y_column, axis=1)   # append it to the dataset

    quality_index = raw_data.shape[1]-2
    y_col_index   = raw_data.shape[1]-1
    
    for n in range(raw_data.shape[0]):
        if raw_data[n,quality_index] >= 6.0:
            raw_data[n,y_col_index] = 1.0
        else:
            raw_data[n,y_col_index] = 0.0
            
    header = header_to_array(file_path)  # Read the header
    header.append('y_column') # Add the categorical y column
    return (raw_data, header)
    

def get_basic_stats(column):
    """
        Return some basic stats on the data column
    """
    mean  = np.mean(column)
    std   = np.std(column)
    min_  = np.min(column)
    max_  = np.max(column)
    return "mean=%.2f, std=%.2f, min=%.2f, max=%.2f" % (mean,std,min_,max_)


def draw_histogram(raw_data, header, variable, nbins=50):
    """
        This function draws the histogram corresponding to the requested variable
    """
    fig = plt.figure(figsize=(12, 5))
    ax  = fig.add_subplot(111)

    # the histogram of the data
    column = raw_data[:, header.index(variable)]
    stats  = get_basic_stats(column)
    title  = 'Histogram of variable %s, nbins = %s\n%s' % (variable, nbins, stats)
    
    ax.hist(column, nbins, density=True, facecolor='g', alpha=0.75)
    ax.set_xlabel(variable)
    ax.set_ylabel('Frequence')
    ax.set_title(title)

    img_name = "../img/histograms/%s.png" % variable.replace(' ', '_')
    plt.savefig(img_name)
    plt.close()
    print("Created %s" % img_name)
    


def draw_boxplot(raw_data, header, variable, nbins=50):
    """
        This function draws the histogram corresponding to the requested variable
    """
    fig = plt.figure(figsize=(12, 5))
    ax  = fig.add_subplot(111)

    # the histogram of the data
    column = raw_data[:, header.index(variable)]
    stats  = get_basic_stats(column)
    title  = 'Boxplot of variable %s, nbins = %s\n%s' % (variable, nbins, stats)
    
    ax.boxplot(column)
    ax.set_xlabel(variable)
    ax.set_title(title)

    img_name = "../img/boxplots/%s.png" % variable.replace(' ', '_')
    plt.savefig(img_name)
    plt.close()
    print("Created %s" % img_name)
 


def main():
    
    red_wine = '../data/wine/winequality-red.csv'
    (raw_data, header) = load_data(red_wine)
    print(raw_data)
    print(header)

    print("\nDrawing the histograms ...")
    for variable in header:
        draw_histogram(raw_data, header, variable)

    print("\nDrawing the boxplots ...")
    for variable in header:
        draw_boxplot(raw_data, header, variable)



if __name__ == "__main__":
    main()

