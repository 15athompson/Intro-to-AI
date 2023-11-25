
import pandas as pd
import re
import string
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorize
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier


# Load the data
train_data = pd.read_csv('train_data.csv')
test_data = pd.read_csv('test_data.csv')

# Initialize CountVectorizer
vectorizer = CountVectorizer(max_features=3000)

# Fit and transform the training data
X_train_counts = vectorizer.fit_transform(train_data['Text'])

# Transform the test data
X_test_counts = vectorizer.transform(test_data['Text'])


#####################################################################
# PERFORM DATA CLEANING AND PRE-PROCESSING ON THE DATA:
r

# Remove URLs, mentions, and punctuation from the text
def clean_text(text):
    # Remove URLs
    text = re.sub(r'http\S+', '', text)
    # Remove mentions
    text = re.sub(r'@\w+', '', text)
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Convert to lowercase
    text = text.lower()
    return text

# Remove stop words from the text
def remove_stopwords(text):
    # Download stopwords
    nltk.download('stopwords')
    stop_words = set(stopwords.words('english'))
    # Remove stopwords
    words = text.split()
    words = [word for word in words if not word in stop_words]
    text = ' '.join(words)
    return text

# Pre-process the text data
def preprocess_text(data):
    # Clean the text data
    data['Text'] = data['Text'].apply(clean_text)
    # Remove stopwords
    data['Text'] = data['Text'].apply(remove_stopwords)
    return data

# Pre-process the training and test data
train_data = preprocess_text(train_data)
test_data = preprocess_text(test_data)

# Create the document-term matrix using CountVectorizer
vectorizer = CountVectorizer(max_features=3000)
X_train = vectorizer.fit_transform(train_data['Text'])
X_test = vectorizer.transform(test_data['Text'])
y_train = train_data['Sentiment']
y_test = test_data['Sentiment']




#####################################################################
# DATA MODELLING:
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import GridSearchCV

# code to split the pre-processed data into training and validation sets:
X_train, X_val, y_train, y_val = train_test_split(preprocessed_df['text'], preprocessed_df['sentiment'], test_size=0.2, random_state=42)


# train a Naive Bayes model on the training set using GridSearchCV to tune the hyperparameters, such as alpha:

# Define the Naive Bayes model
nb = MultinomialNB()

# Define the hyperparameters to tune
param_grid = {'alpha': [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]}

# Use GridSearchCV to search for the best hyperparameters
nb_gs = GridSearchCV(nb, param_grid, cv=5, n_jobs=1, verbose=1)

# Train the model on the training data
nb_gs.fit(X_train, y_train)


#####################################################################
# MODEL EVALUATION:

# Test the model on the test set
y_pred = nb_grid.predict(X_test)

# Evaluate the performance of the model
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

print("Model Performance on Test Set:")
print("Accuracy: {:.4f}".format(accuracy))
print("Precision: {:.4f}".format(precision))
print("Recall: {:.4f}".format(recall))
print("F1-Score: {:.4f}".format(f1))


#####################################################################
# MODEL EVALUATION:

# using Logistic Regression and Random Forest models for sentiment analysis and comparing their performance with Naive Bayes model:

# Logistic Regression Model:



# initialize logistic regression model
logreg = LogisticRegression(max_iter=1000)

# define parameter grid for grid search
param_grid = {
    'penalty': ['l1', 'l2'],
    'C': [0.001, 0.01, 0.1, 1, 10],
    'solver': ['liblinear', 'saga']
}

# perform grid search to tune hyperparameters
logreg_gs = GridSearchCV(logreg, param_grid=param_grid, cv=5, n_jobs=-1, verbose=1)
logreg_gs.fit(X_train, y_train)

# make predictions on test set
y_pred_logreg = logreg_gs.predict(X_test)

# evaluate performance using classification report
print('Logistic Regression Model Performance:')
print(classification_report(y_test, y_pred_logreg))


# Random Forest Model



# initialize random forest model
rf = RandomForestClassifier()

# define parameter grid for grid search
param_grid = {
    'n_estimators': [100, 200, 500],
    'max_depth': [5, 10, 20, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# perform grid search to tune hyperparameters
rf_gs = GridSearchCV(rf, param_grid=param_grid, cv=5, n_jobs=-1, verbose=1)
rf_gs.fit(X_train, y_train)

# make predictions on test set
y_pred_rf = rf_gs.predict(X_test)

# evaluate performance using classification report
print('Random Forest Model Performance:')
print(classification_report(y_test, y_pred_rf))

