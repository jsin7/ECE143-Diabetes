# Import Neccessary libraries
import numpy as np 
import pandas as pd 

# Import Visualization libraries
import matplotlib.pyplot as plt
import seaborn as sns

#
def convert_smoking_history(status):
    """classifies a broad smoking category into a set list of groups

    Args:
        status (str): original classification

    Returns:
        str: new classification
    """    
    if status in ['current', 'ever', 'former', 'not current']:
        return 'smoker'
    elif status in ['never']:
        return 'non smoker'
    else:
        return 'No Info'
    
def calc_correlation_matrix(df):
    """calculate the pairwise correlation matrix for a dataframe with multiple columns

    Args:
        df (dataframe): dataframe of values to calculate correlation for
    
    Returns:
        dataframe: returns pairwise correlation matrix
    """    

    correlation_matrix = pd.DataFrame(index=df.columns, columns=df.columns)

# Calculate pairwise correlations
    for col1 in df.columns:
        for col2 in df.columns:
            if col1 == col2:
                correlation_matrix.loc[col1, col2] = 1.0
            else:
                # Filter out 'no info' only when dealing with smoking history
                if col1 == 'smoking_history' or col2 == 'smoking_history':
                    filtered_df = df.dropna(subset=['smoking_history'])
                    correlation_matrix.loc[col1, col2] = filtered_df[col1].corr(filtered_df[col2])
                else:
                    correlation_matrix.loc[col1, col2] = df[col1].corr(df[col2])

    correlation_matrix = correlation_matrix.astype(float)
    return correlation_matrix

def display_diabetes_correlation(df, title, xlabel):
    """display diabetes correlation matrix for given matrix

    Args:
        df (dataframe_): correlation matrix
        title (str): title of plot
        xlabel (str): label for x axis
    """    
    sns.set(font_scale=.9)
    sns.set_style("white")
    sns.set_palette("PuBuGn_d")
    sns.heatmap(df, cmap="coolwarm", annot=True, fmt='.2f')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.savefig(title, bbox_inches='tight', dpi=300)
    plt.show()

def age_stratification(df):
    """split dataframe by age

    Args:
        df (dataframe): dataframe to split

    Returns:
        tuple: tuple of dataframes, each containing data for that age group
    """    

    bins = [0, 5, 20, 50, 70, 120]  # Adjust the last bin as needed
    labels = ['0-4', '5-19', '20-49', '50-69', '70+']

    # Categorize the ages into the broader bins
    df['age_group'] = pd.cut(df['age'], bins=bins, labels=labels, right=False)

    # Create new DataFrames for each age group, dropping the 'age' column in the process
    df_0_4 = df[df['age_group'] == '0-4'].drop(['age'], axis=1)
    df_5_19 = df[df['age_group'] == '5-19'].drop(['age'], axis=1)
    df_20_49 = df[df['age_group'] == '20-49'].drop(['age'], axis=1)
    df_50_69 = df[df['age_group'] == '50-69'].drop(['age'], axis=1)
    df_70_plus = df[df['age_group'] == '70+'].drop(['age'], axis=1)

    return df_0_4, df_5_19, df_20_49, df_50_69, df_70_plus


# Define a function to map the existing categories to new ones
def recategorize_smoking(smoking_status):
    if smoking_status in ['never', 'No Info']:
        return 'non-smoker'
    elif smoking_status == 'current':
        return 'current'
    elif smoking_status in ['ever', 'former', 'not current']:
        return 'past_smoker'
    
def make_confusion_matrix(cf,
                          group_names=None,
                          categories='auto',
                          count=True,
                          percent=True,
                          cbar=True,
                          xyticks=True,
                          xyplotlabels=True,
                          sum_stats=True,
                          figsize=None,
                          cmap='Blues',
                          my_dpi=100,
                          title=None,
                          saved=False,
                          save_name='Unsupervised Learning.png'):
    '''
    This function will make a pretty plot of an sklearn Confusion Matrix cm using a Seaborn heatmap visualization.
    Arguments
    ---------
    cf:            confusion matrix to be passed in
    group_names:   List of strings that represent the labels row by row to be shown in each square.
    categories:    List of strings containing the categories to be displayed on the x,y axis. Default is 'auto'
    count:         If True, show the raw number in the confusion matrix. Default is True.
    normalize:     If True, show the proportions for each category. Default is True.
    cbar:          If True, show the color bar. The cbar values are based off the values in the confusion matrix.
                   Default is True.
    xyticks:       If True, show x and y ticks. Default is True.
    xyplotlabels:  If True, show 'True Label' and 'Predicted Label' on the figure. Default is True.
    sum_stats:     If True, display summary statistics below the figure. Default is True.
    figsize:       Tuple representing the figure size. Default will be the matplotlib rcParams value.
    cmap:          Colormap of the values displayed from matplotlib.pyplot.cm. Default is 'Blues'
                   See http://matplotlib.org/examples/color/colormaps_reference.html
                   
    title:         Title for the heatmap. Default is None.
    '''


    # CODE TO GENERATE TEXT INSIDE EACH SQUARE
    blanks = ['' for i in range(cf.size)]

    if group_names and len(group_names)==cf.size:
        group_labels = ["{}\n".format(value) for value in group_names]
    else:
        group_labels = blanks

    if count:
        group_counts = ["{0:0.1%}\n".format(value) for value in cf.flatten()]
    else:
        group_counts = blanks

    if percent:
        group_percentages = ["{0:.2%}".format(value) for value in cf.flatten()/np.tile(np.sum(cf,axis = 0),(3,))]
    else:
        group_percentages = blanks

    box_labels = [f"{v1}{v2}{v3}".strip() for v1, v2, v3 in zip(group_labels,group_counts,group_percentages)]
    box_labels = np.asarray(box_labels).reshape(cf.shape[0],cf.shape[1])

    # CODE TO GENERATE SUMMARY STATISTICS & TEXT FOR SUMMARY STATS
    if sum_stats:
        #Accuracy is sum of diagonal divided by total observations
        accuracy  = np.trace(cf) / float(np.sum(cf))
#         accuracy  = np.trace(cf) / 3
        #if it is a binary confusion matrix, show some more stats
        if len(cf)==2:
            #Metrics for Binary Confusion Matrices
            precision = cf[1,1] / sum(cf[:,1])
            recall    = cf[1,1] / sum(cf[1,:])
            f1_score  = 2*precision*recall / (precision + recall)
            stats_text = "\n\nAccuracy={:0.3f}\nPrecision={:0.3f}\nRecall={:0.3f}\nF1 Score={:0.3f}".format(
                accuracy,precision,recall,f1_score)
        else:
            stats_text = "\n\nBalanced Accuracy={:0.3f}".format(accuracy)
    else:
        stats_text = ""

    # SET FIGURE PARAMETERS ACCORDING TO OTHER ARGUMENTS
    if figsize==None:
        #Get default figure size if not set
        figsize = plt.rcParams.get('figure.figsize')

    if xyticks==False:
        #Do not show categories if xyticks is False
        categories=False

    # print(box_labels)
    # MAKE THE HEATMAP VISUALIZATION
    plt.figure(figsize=figsize, dpi=my_dpi)
    ax = sns.heatmap(cf, annot=box_labels, fmt="", cmap=cmap, cbar=cbar, xticklabels=categories, yticklabels=categories)
    for text in ax.texts:
        text.set_horizontalalignment('center')
        text.set_verticalalignment('center')

    if xyplotlabels:
        plt.ylabel('True label')
        plt.xlabel('Predicted label' + stats_text)
    else:
        plt.xlabel(stats_text)
    
    if title:
        plt.title(title)
    if saved:
        plt.savefig(save_name, dpi=my_dpi*10, bbox_inches='tight')
    
def plot_feature_importances(model_pipeline, model_name, feature_names):
    # Check if the model has the 'feature_importances_' attribute
    if hasattr(model_pipeline.named_steps['model'], 'feature_importances_'):
        # Get feature importances
        importances = model_pipeline.named_steps['model'].feature_importances_
        
        # Create a DataFrame for feature importances
        importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
        
        # Sort the DataFrame by importance
        importance_df.sort_values('Importance', ascending=False, inplace=True)

        # Visualize feature importances
        plt.figure(figsize=(10, 6), dpi=200)
        sns.barplot(x='Importance', y='Feature', data=importance_df)
        plt.title(f'Feature Importances for {model_name}', fontsize=16)
        plt.xlabel('Importance', fontsize=14)
        plt.ylabel('Feature', fontsize=14)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.tight_layout()  # Adjust layout to make room for the feature names
        plt.show()
    else:
        print(f"{model_name} does not support feature importance.")