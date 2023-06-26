import os
import io
import base64
import glob
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
from sklearn.metrics import roc_curve, roc_auc_score, accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.tree import DecisionTreeClassifier


app = Flask(__name__)

# Set the directory for uploaded files
UPLOAD_FOLDER = 'userinput'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 140 * 1024 * 1024  # Set the maximum file size to 16MB

# Create the userinput directory if it doesn't exist
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

plot_descriptions = {
    'logistic_regression': 'Logistic Regression is a supervised learning algorithm for binary classification tasks.',
    'k_means_algorithm': 'The K-means algorithm is used for clustering tasks where the goal is to identify groups or patterns in the data.',
    'k_nn_algorithm': 'The k-Nearest Neighbors (k-NN) algorithm is a supervised learning method that can be used for both classification and regression tasks. It is based on the idea that data points with similar features will likely have identical target values.',
    'linear_regression_algorithm': 'The Linear Regression algorithm is useful because it is a simple and efficient method for modeling the relationship between a dependent variable (target) and one or more independent variables (features).',
    'decision_tree_algorithm': 'The Decision Tree algorithm is a versatile supervised learning method for classification and regression tasks. It recursively splits the dataset into subsets based on the most informative feature at each step, resulting in a tree-like structure with decision nodes and leaf nodes representing the predicted target variable.',

}

def process_data():
    # Data
    df = pd.read_csv('cleaned_titles.csv')
    
    return df

def process_userInput(input_file):
    df = pd.read_csv(input_file)
    return df

def plot_histogram_imdb(df):
    img = io.BytesIO()
    plt.hist(df['imdb_score'], bins=20)
    plt.xlabel('IMDB Score')
    plt.ylabel('Frequency')
    plt.title('Distribution of IMDB Scores')
    plt.savefig(img, format='png')
    plt.clf()
    img.seek(0)
    return base64.b64encode(img.getvalue()).decode()

def plot_scatter_imdb_tmdb(df):
    img = io.BytesIO()
    plt.scatter(df['imdb_score'], df['tmdb_score'])
    plt.xlabel('IMDB Score')
    plt.ylabel('TMDB Score')
    plt.title('Relationship between IMDB Score and TMDB Score')
    plt.savefig(img, format='png')
    plt.clf()
    img.seek(0)
    return base64.b64encode(img.getvalue()).decode()

def plot_histogram_tmdb(df):
    img = io.BytesIO()
    plt.hist(df['tmdb_score'], bins=20)
    plt.xlabel('TMDB Score')
    plt.ylabel('Frequency')
    plt.title('Distribution of TMDB Scores')
    plt.savefig(img, format='png')
    plt.clf()
    img.seek(0)
    return base64.b64encode(img.getvalue()).decode()

def plot_line_imdb_by_year(df):
    img = io.BytesIO()
    imdb_scores_by_year = df.groupby('release_year')['imdb_score'].mean()
    plt.plot(imdb_scores_by_year)
    plt.xlabel('Release Year')
    plt.ylabel('Average IMDB Score')
    plt.title('IMDB Scores by Release Year')
    plt.savefig(img, format='png')
    plt.clf()
    img.seek(0)
    return base64.b64encode(img.getvalue()).decode()

def plot_bar_top_directors(df):
    img = io.BytesIO()
    top_directors = df.groupby('director')['imdb_score'].mean().sort_values(ascending=False)[:10]
    plt.bar(top_directors.index, top_directors.values)
    plt.xticks(rotation=90)
    plt.xlabel('Director')
    plt.ylabel('Average IMDB Score')
    plt.title('Top 10 Directors by IMDB Score')
    plt.savefig(img, format='png')
    plt.clf()
    img.seek(0)
    return base64.b64encode(img.getvalue()).decode()

def plot_bar_director_counts(df):
    img = io.BytesIO()
    director_counts = df.groupby('director')['id'].count().sort_values(ascending=False)[:10]
    plt.bar(director_counts.index, director_counts.values)
    plt.xticks(rotation=90)
    plt.xlabel('Director')
    plt.ylabel('Number of Movies and Shows')
    plt.title('Top 10 Directors by Number of Movies and Shows Directed')
    plt.savefig(img, format='png')
    plt.clf()
    img.seek(0)
    return base64.b64encode(img.getvalue()).decode()

def plot_bar_lowest_rated_directors(df):
    img = io.BytesIO()
    director_counts = df['director'].value_counts()
    lowest_rated_directors = df.groupby('director')['imdb_score'].mean().sort_values()[:10].index
    lowest_rated_director_counts = director_counts[lowest_rated_directors]
    plt.bar(lowest_rated_director_counts.index, lowest_rated_director_counts.values)
    plt.xticks(rotation=90)
    plt.xlabel('Director')
    plt.ylabel('Frequency')
    plt.title('Frequency of Top 10 Lowest Rated Directors')
    plt.savefig(img, format='png')
    plt.clf()
    img.seek(0)
    return base64.b64encode(img.getvalue()).decode()

def plot_scatter_runtime_imdb(df):
    img = io.BytesIO()
    plt.scatter(df['runtime'], df['imdb_score'])
    plt.xlabel('Runtime (minutes)')
    plt.ylabel('IMDB Score')
    plt.title('Relationship between Movie Runtime and IMDB Score')
    plt.savefig(img, format='png')
    plt.clf()
    img.seek(0)
    return base64.b64encode(img.getvalue()).decode()

def plot_hexbin_runtime_imdb(df):
    img = io.BytesIO()
    plt.hexbin(df['runtime'], df['imdb_score'], gridsize=20, cmap='YlOrRd')
    plt.xlabel('Runtime (minutes)')
    plt.ylabel('IMDB Score')
    plt.title('Relationship between Movie Runtime and IMDB Score')
    cb = plt.colorbar()
    cb.set_label('Frequency')
    plt.savefig(img, format='png')
    plt.clf()
    img.seek(0)
    return base64.b64encode(img.getvalue()).decode()

def plot_scatter_runtime_tmdb(df):
    img = io.BytesIO()
    plt.scatter(df['runtime'], df['tmdb_score'])
    plt.xlabel('Runtime (minutes)')
    plt.ylabel('TMDB Score')
    plt.title('Relationship between Movie Runtime and TMDB Score')
    plt.savefig(img, format='png')
    plt.clf()
    img.seek(0)
    return base64.b64encode(img.getvalue()).decode()

def plot_hexbin_runtime_tmdb(df):
    img = io.BytesIO()
    plt.hexbin(df['runtime'], df['tmdb_score'], gridsize=20, cmap='YlOrRd')
    plt.xlabel('Runtime (minutes)')
    plt.ylabel('TMDB Score')
    plt.title('Relationship between Movie Runtime and TMDB Score')
    cb = plt.colorbar()
    cb.set_label('Frequency')
    plt.savefig(img, format='png')
    plt.clf()
    img.seek(0)
    return base64.b64encode(img.getvalue()).decode()

def plot_histogram_movies_shows_imdb(df):
    img = io.BytesIO()
    # Separate the movies and shows data
    movies_data = df[df['type'] == 'MOVIE']
    shows_data = df[df['type'] == 'SHOW']
    plt.hist(movies_data['imdb_score'], alpha=0.5, label='Movies')
    plt.hist(shows_data['imdb_score'], alpha=0.5, label='Shows')
    plt.title('Distribution of IMDB Scores for Movies and Shows')
    plt.xlabel('IMDB Score')
    plt.ylabel('Frequency')
    plt.legend()
    plt.savefig(img, format='png')
    plt.clf()
    img.seek(0)
    return base64.b64encode(img.getvalue()).decode()

def plot_boxplot_movies_shows_imdb(df):
    img = io.BytesIO()
    # Separate the movies and shows data
    movies_data = df[df['type'] == 'MOVIE']
    shows_data = df[df['type'] == 'SHOW']

    plt.boxplot([movies_data['imdb_score'], shows_data['imdb_score']], labels=['Movies', 'Shows'])
    plt.title('Box Plot of IMDB Scores for Movies and Shows')
    plt.ylabel('IMDB Score')
    plt.savefig(img, format='png')
    plt.clf()
    img.seek(0)
    return base64.b64encode(img.getvalue()).decode()




#USER INPUT DATA

def plot_line(input1, input2, df):
    img = io.BytesIO()
    imdb_scores_by_year = df.groupby(input1)[input2].mean()
    plt.plot(imdb_scores_by_year)
    plt.xlabel(input1)
    plt.ylabel(input2)
    plt.title('Average ' + input2 + ' by ' + input1)
    plt.savefig(img, format='png')
    plt.clf()
    img.seek(0)
    return base64.b64encode(img.getvalue()).decode()

def plot_histogram_freq(input1, df):
    img = io.BytesIO()
    plt.hist(df[input1], bins=20)
    plt.xlabel(input1)
    plt.ylabel('Frequency')
    plt.title('Distribution of ' + input1)
    plt.savefig(img, format='png')
    plt.clf()
    img.seek(0)
    return base64.b64encode(img.getvalue()).decode()

def plot_bar_top(input1,input2,df):
    img = io.BytesIO()
    top_directors = df.groupby(input1)[input2].mean().sort_values(ascending=False)[:10]
    plt.bar(top_directors.index, top_directors.values)
    plt.xticks(rotation=90)
    plt.xlabel(input1)
    plt.ylabel(input2)
    plt.title('Top 10' + input1 + 'by' + input2)
    plt.savefig(img, format='png')
    plt.clf()
    img.seek(0)
    return base64.b64encode(img.getvalue()).decode()

def user_k_nn_algorithm(df, feature_col, target_col, in_threshold):
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.metrics import classification_report, confusion_matrix, roc_curve, roc_auc_score

    # Define the threshold 
    threshold = float(in_threshold)

    # Create a binary target variable for classification
    df['is_good'] = (df[target_col] >= threshold).astype(int)

    # Select the features and target
    X = df[[feature_col]]  # Feature
    y = df['is_good']       # Target

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Find the optimal number of neighbors for K-NN
    neighbors = range(1, 21)
    test_accuracies = []
    for n in neighbors:
        knn = KNeighborsClassifier(n_neighbors=n)
        knn.fit(X_train, y_train)
        test_accuracies.append(knn.score(X_test, y_test))

    # Save the Test Set Accuracies plot
    img = io.BytesIO()
    plt.plot(neighbors, test_accuracies)
    plt.xlabel('Number of Neighbors')
    plt.ylabel('Test Set Accuracy')
    plt.title('Test Set Accuracies for Different Numbers of Neighbors')
    plt.savefig(img, format='png')
    plt.clf()
    img.seek(0)
    test_accuracies_url = base64.b64encode(img.getvalue()).decode()

    # Apply K-NN with the optimal number of neighbors
    optimal_neighbors = neighbors[np.argmax(test_accuracies)]
    knn = KNeighborsClassifier(n_neighbors=optimal_neighbors)
    knn.fit(X_train, y_train)

    # Evaluate the model on the test set
    y_pred = knn.predict(X_test)
    y_pred_proba = knn.predict_proba(X_test)[:, 1]
    classification_report_str = classification_report(y_test, y_pred)

    # Plot confusion matrix with True Label on the Y-axis and Predicted Label on the X-axis
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='coolwarm', cbar=False)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.savefig(img, format='png')
    plt.clf()
    img.seek(0)
    confusion_matrix_url = base64.b64encode(img.getvalue()).decode()

    # Plot ROC curve which shows the True and False Positive Rates
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, label='ROC curve (AUC = %0.2f)' % roc_auc_score(y_test, y_pred_proba))
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    plt.savefig(img, format='png')
    plt.clf()
    img.seek(0)
    roc_curve_url = base64.b64encode(img.getvalue()).decode()

    return test_accuracies_url, confusion_matrix_url, roc_curve_url, classification_report_str



def decision_tree_user(input1, input2, input3, df):

    threshold = float(input3)
    df['is_good'] = (df[input2] >= threshold).astype(int)
    X = df[[input1]]
    y = df['is_good']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    dtree = DecisionTreeClassifier(max_depth=5, random_state=42)
    dtree.fit(X_train, y_train)

    y_pred = dtree.predict(X_test)
    y_pred_proba = dtree.predict_proba(X_test)[:, 1]

    classification_report_str = classification_report(y_test, y_pred)

    # Confusion Matrix plot
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='coolwarm', cbar=False)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    cm_data = base64.b64encode(buf.read()).decode()
    
    plt.close()

    # ROC Curve plot
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, label='ROC curve (AUC = %0.2f)' % roc_auc_score(y_test, y_pred_proba))
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    roc_data = base64.b64encode(buf.read()).decode()
    plt.close()
    return classification_report_str, cm_data, roc_data

def user_linear_regression(df, feature_col, target_col):
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_squared_error, r2_score
    # Select the features and target which are IMDB score and TMDB score
    X = df[[feature_col]]  # Feature
    y = df[target_col]    # Target

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    # Create a linear regression model
    regressor = LinearRegression()

    # Train the model using the training set
    regressor.fit(X_train, y_train)

    # Make predictions on the testing set
    y_pred = regressor.predict(X_test)

    # Calculate the mean squared error and R-squared score
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Display the results of the data
    plt.scatter(X_test, y_test, color='blue', label='Actual')
    plt.scatter(X_test, y_pred, color='red', label='Predicted')
    plt.xlabel(feature_col)
    plt.ylabel(target_col)
    plt.legend()

    plt.title(target_col + ' vs ' + feature_col)
    
    # Save the plot to a buffer and return it as a base64-encoded image
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plot_data = base64.b64encode(buf.read()).decode()
    
    plt.close()

    return mse, r2, plot_data





# Phase 2 data

def plot_logistic_regression(df):
    # Create and define the threshold for a "good" movie or show
    threshold = 7.0
    df['is_good'] = (df['imdb_score'] >= threshold).astype(int)

    # Define Feature and Target for model
    X = df[['tmdb_score']]  # Feature
    y = df['is_good']       # Target

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    # Create a logistic regression model
    classifier = LogisticRegression()

    # Train the model using the training set
    classifier.fit(X_train, y_train)

    # Make predictions on the testing set
    y_pred = classifier.predict(X_test)
    y_pred_proba = classifier.predict_proba(X_test)[:, 1]

    # Calculate the accuracy of the model
    accuracy = accuracy_score(y_test, y_pred)

    print("Accuracy: ", accuracy)

    # Print the confusion matrix and classification report
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))

    # Plot confusion matrix with True Label on the Y-axis and Predicted Label on the X-axis
    cm = confusion_matrix(y_test, y_pred)
    img1 = io.BytesIO()
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='coolwarm', cbar=False)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.savefig(img1, format='png', bbox_inches='tight')
    plt.clf()
    img1.seek(0)
    confusion_matrix_url = base64.b64encode(img1.getvalue()).decode()

    # Plot ROC curve which shows the True and False Positive Rates
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    img2 = io.BytesIO()
    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, label='ROC curve (AUC = %0.2f)' % roc_auc_score(y_test, y_pred_proba))
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    plt.savefig(img2, format='png', bbox_inches='tight')
    plt.clf()
    img2.seek(0)
    roc_curve_url = base64.b64encode(img2.getvalue()).decode()

    return confusion_matrix_url, roc_curve_url

def k_means_algorithm(df):
    # Choose relevant features for clustering
    features = ['release_year', 'seasons', 'imdb_score']
    X = df[features]

    # Normalize the numerical features using MinMaxScaler
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    # Find the optimal number of clusters using the silhouette score defined in an array 
    n_clusters = range(2, 11)
    silhouette_scores = []

    for n in n_clusters:
        kmeans = KMeans(n_clusters=n, n_init=10, random_state=42)
        cluster_labels = kmeans.fit_predict(X_scaled)
        silhouette_scores.append(silhouette_score(X_scaled, cluster_labels))

    # Apply K-Means clustering with the optimal number of clusters
    optimal_clusters = n_clusters[np.argmax(silhouette_scores)]
    kmeans = KMeans(n_clusters=optimal_clusters, n_init=10, random_state=42)
    cluster_labels = kmeans.fit_predict(X_scaled)

    # Plot the silhouette scores for each number of clusters
    plt.plot(n_clusters, silhouette_scores)
    plt.xlabel('Number of Clusters')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Scores for Different Numbers of Clusters')
    plt.show()

    # K-Means clustering with the optimal number of clusters
    optimal_clusters = n_clusters[np.argmax(silhouette_scores)]
    kmeans = KMeans(n_clusters=optimal_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(X_scaled)

    # add the cluster labels to the original dataframe
    df['cluster'] = cluster_labels

    # Save the Silhouette Scores plot
    img = io.BytesIO()
    plt.plot(n_clusters, silhouette_scores)
    plt.xlabel('Number of Clusters')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Scores for Different Numbers of Clusters')
    plt.savefig(img, format='png')
    plt.clf()
    img.seek(0)
    silhouette_scores_url = base64.b64encode(img.getvalue()).decode()

    # Save the K-Means Clustering plot
    img = io.BytesIO()
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x='release_year', y='imdb_score', hue='cluster', palette='Set2', s=50)
    plt.title(f'K-Means Clustering: {optimal_clusters} Clusters')
    plt.xlabel('Release Year')
    plt.ylabel('IMDb Score')
    plt.legend(title='Cluster', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.savefig(img, format='png')
    plt.clf()
    img.seek(0)
    k_means_plot_url = base64.b64encode(img.getvalue()).decode()

    return silhouette_scores_url, k_means_plot_url



def k_nn_algorithm(df):
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.metrics import classification_report, confusion_matrix, roc_curve, roc_auc_score

    # Define the threshold for a "good" movie or show
    threshold = 7.0

    # Create a binary target variable for classification
    df['is_good'] = (df['imdb_score'] >= threshold).astype(int)

    # Select the features and target
    X = df[['tmdb_score']]  # Feature
    y = df['is_good']       # Target

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Find the optimal number of neighbors for K-NN
    neighbors = range(1, 21)
    test_accuracies = []
    for n in neighbors:
        knn = KNeighborsClassifier(n_neighbors=n)
        knn.fit(X_train, y_train)
        test_accuracies.append(knn.score(X_test, y_test))

    # Save the Test Set Accuracies plot
    img = io.BytesIO()
    plt.plot(neighbors, test_accuracies)
    plt.xlabel('Number of Neighbors')
    plt.ylabel('Test Set Accuracy')
    plt.title('Test Set Accuracies for Different Numbers of Neighbors')
    plt.savefig(img, format='png')
    plt.clf()
    img.seek(0)
    test_accuracies_url = base64.b64encode(img.getvalue()).decode()

    # Apply K-NN with the optimal number of neighbors
    optimal_neighbors = neighbors[np.argmax(test_accuracies)]
    knn = KNeighborsClassifier(n_neighbors=optimal_neighbors)
    knn.fit(X_train, y_train)

    # Evaluate the model on the test set
    y_pred = knn.predict(X_test)
    y_pred_proba = knn.predict_proba(X_test)[:, 1]
    classification_report_str = classification_report(y_test, y_pred)

    # Plot confusion matrix with True Label on the Y-axis and Predicted Label on the X-axis
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='coolwarm', cbar=False)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.savefig(img, format='png')
    plt.clf()
    img.seek(0)
    confusion_matrix_url = base64.b64encode(img.getvalue()).decode()

    # Plot ROC curve which shows the True and False Positive Rates
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, label='ROC curve (AUC = %0.2f)' % roc_auc_score(y_test, y_pred_proba))
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    plt.savefig(img, format='png')
    plt.clf()
    img.seek(0)
    roc_curve_url = base64.b64encode(img.getvalue()).decode()

    return test_accuracies_url, confusion_matrix_url, roc_curve_url, classification_report_str




def linear_regression_algorithm(df):
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_squared_error, r2_score
    # Select the features and target which are IMDB score and TMDB score
    X = df[['tmdb_score']]  # Feature
    y = df['imdb_score']    # Target

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    # Create a linear regression model
    regressor = LinearRegression()

    # Train the model using the training set
    regressor.fit(X_train, y_train)

    # Make predictions on the testing set
    y_pred = regressor.predict(X_test)

    # Calculate the mean squared error and R-squared score
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Display the results of the data
    plt.scatter(X_test, y_test, color='blue', label='Actual')
    plt.scatter(X_test, y_pred, color='red', label='Predicted')
    plt.xlabel('TMDB Score')
    plt.ylabel('IMDB Score')
    plt.legend()

    plt.title('Linear Regression: TMDB Score vs IMDB Score')
    
    # Save the plot to a buffer and return it as a base64-encoded image
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plot_data = base64.b64encode(buf.read()).decode()
    
    plt.close()

    return mse, r2, plot_data

def decision_tree_algorithm(df):
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.metrics import classification_report, confusion_matrix, roc_curve, roc_auc_score
    import io
    import base64
    
    threshold = 7.0
    df['is_good'] = (df['imdb_score'] >= threshold).astype(int)
    X = df[['tmdb_score']]
    y = df['is_good']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    dtree = DecisionTreeClassifier(max_depth=5, random_state=42)
    dtree.fit(X_train, y_train)

    y_pred = dtree.predict(X_test)
    y_pred_proba = dtree.predict_proba(X_test)[:, 1]

    classification_report_str = classification_report(y_test, y_pred)

    # Confusion Matrix plot
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='coolwarm', cbar=False)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    cm_data = base64.b64encode(buf.read()).decode()
    
    plt.close()

    # ROC Curve plot
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, label='ROC curve (AUC = %0.2f)' % roc_auc_score(y_test, y_pred_proba))
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    roc_data = base64.b64encode(buf.read()).decode()
    
    plt.close()
    
    return classification_report_str, cm_data, roc_data





    
@app.route('/', methods=['GET', 'POST'])
def movie():
    if request.method == 'POST':
        df = process_data()

        # Get user input from the front-end
        plot_choice = request.form.get('plot_choice')
        plot_description = None


        # Call the corresponding plotting function based on user input
        if plot_choice == 'histogram_imdb':
            plot_url = plot_histogram_imdb(df)
        elif plot_choice == 'scatter_imdb_tmdb':
            plot_url = plot_scatter_imdb_tmdb(df)
        elif plot_choice == 'histogram_tmdb':
            plot_url = plot_histogram_tmdb(df)
        elif plot_choice == 'line_imdb_by_year':
            plot_url = plot_line_imdb_by_year(df)
        elif plot_choice == 'bar_directors_imdb':
            plot_url = plot_bar_top_directors(df)
        elif plot_choice == 'bar_director_counts':
            plot_url = plot_bar_director_counts(df)
        elif plot_choice == 'bar_lowest_rated_directors':
            plot_url = plot_bar_lowest_rated_directors(df)
        elif plot_choice == 'scatter_runtime_imdb':
            plot_url = plot_scatter_runtime_imdb(df)
        elif plot_choice == 'hexbin_runtime_imdb':
                plot_url = plot_hexbin_runtime_imdb(df)
        elif plot_choice == 'scatter_runtime_tmdb':
            plot_url = plot_scatter_runtime_tmdb(df)
        elif plot_choice == 'hexbin_runtime_tmdb':
            plot_url = plot_hexbin_runtime_tmdb(df)
        elif plot_choice == 'histogram_movies_shows_imdb':
            plot_url = plot_histogram_movies_shows_imdb(df)
        elif plot_choice == 'boxplot_movies_shows_imdb':
            plot_url = plot_boxplot_movies_shows_imdb(df)

        # Phase 2 Plots
        elif plot_choice == 'logistic_regression':
            plot_description = plot_descriptions.get(plot_choice)
            confusion_matrix_url, roc_curve_url = plot_logistic_regression(df)
            return render_template('index.html', plot_description=plot_description, confusion_matrix_url=confusion_matrix_url, roc_curve_url=roc_curve_url)
        elif plot_choice == 'k_means_algorithm':
            plot_description = plot_descriptions.get(plot_choice)
            silhouette_scores_url, k_means_plot_url = k_means_algorithm(df)
            return render_template('index.html', plot_description=plot_description, silhouette_scores_url=silhouette_scores_url, k_means_plot_url=k_means_plot_url)   
        elif plot_choice == "k_nn_algorithm":
            plot_description = plot_descriptions.get(plot_choice)
            test_accuracies_url, confusion_matrix_url, roc_curve_url, classification_report_str = k_nn_algorithm(df)
            return render_template("index.html", plot_description=plot_description, plot_choice=plot_choice, test_accuracies_url=test_accuracies_url, confusion_matrix_url=confusion_matrix_url, roc_curve_url=roc_curve_url, classification_report_str=classification_report_str)
        elif plot_choice == 'linear_regression_algorithm':
            plot_description = plot_descriptions.get(plot_choice)
            mse, r2, plot_url = linear_regression_algorithm(df)
            return render_template('index.html', plot_description=plot_description, plot_url=plot_url, plot_choice=plot_choice, mse=mse, r2=r2)
        elif plot_choice == 'decision_tree_algorithm':
            plot_description = plot_descriptions.get(plot_choice)
            classification_report_str, confusion_matrix_url, roc_curve_url = decision_tree_algorithm(df)
            return render_template('index.html', plot_description=plot_description, plot_choice=plot_choice, confusion_matrix_url=confusion_matrix_url, roc_curve_url=roc_curve_url, classification_report=classification_report_str)
        

        else:
            plot_url = None
            plot_choice = None
            plot_description = None
        return render_template('index.html', plot_description=plot_description, plot_url=plot_url, plot_choice=plot_choice)
    else:
        return render_template('index.html')
    

@app.route('/input', methods=['GET', 'POST'])
def input():
        current_file = next(iter(os.listdir(app.config['UPLOAD_FOLDER'])), None)
        if request.method == 'POST':
            folder_name = 'userinput'
            files = os.listdir(folder_name)

            if files:
                file_name = files[0]
                file_path = os.path.join(folder_name, file_name)
                df = process_userInput(file_path)
                # Get user input from the front-end
                plot_choice = request.form.get('plot_choice')
                input1 = request.form.get('input_1')
                input2 = request.form.get('input_2')
                input3 = request.form.get('input_3')
                plot_description = None


                # Call the corresponding plotting function based on user input
                if plot_choice == 'histogram_freq':
                    plot_url = plot_histogram_freq(input1,df)
                elif plot_choice == 'bar':
                    plot_url = plot_bar_top(input1,input2,df)
                elif plot_choice == 'line':
                    plot_url = plot_line(input1,input2,df)
                elif plot_choice == 'histogram_freq':
                    plot_url = plot_histogram_freq(input1,df)  
                elif plot_choice == 'linear_regression':
                    plot_description = plot_descriptions.get(plot_choice)
                    mse, r2, plot_url = user_linear_regression(df, input1, input2)
                    return render_template('input.html',plot_description=plot_description, plot_choice=plot_choice, plot_url=plot_url, mse=mse, r2=r2)
                elif plot_choice == 'k_nn_algorithm':
                    plot_description = plot_descriptions.get(plot_choice)
                    test_accuracies_url, confusion_matrix_url, roc_curve_url, classification_report_str = user_k_nn_algorithm(df, input1, input2, input3)
                    return render_template("input.html", plot_description=plot_description, plot_choice=plot_choice, test_accuracies_url=test_accuracies_url, confusion_matrix_url=confusion_matrix_url, roc_curve_url=roc_curve_url, classification_report_str=classification_report_str)
                elif plot_choice == 'decision_tree_algorithm':
                    plot_description = plot_descriptions.get(plot_choice)
                    classification_report_str, confusion_matrix_url, roc_curve_url = decision_tree_user(input1, input2, input3, df)
                    return render_template('input.html', plot_description=plot_description, plot_choice=plot_choice, confusion_matrix_url=confusion_matrix_url, roc_curve_url=roc_curve_url, classification_report=classification_report_str)                       
                else:
                    plot_url = None
                    plot_description = None
            else:
                print("No files found in the 'userinput' folder")
            return render_template('input.html', plot_url=plot_url, plot_choice=plot_choice,current_file=current_file)
        else:
            return render_template('input.html',current_file=current_file) 
    
@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(url_for('input'))
    
    file = request.files['file']
    if file.filename == '':
        return redirect(url_for('input'))

    if file and file.filename.lower().endswith('.csv'):
        # Remove existing files in the userinput directory
        existing_files = glob.glob(os.path.join(app.config['UPLOAD_FOLDER'], '*'))
        for f in existing_files:
            os.remove(f)

        # Save the new file
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        return redirect(url_for('input'))
    return redirect(url_for('input'))

if __name__ == '__main__':
    app.run(debug=True)
