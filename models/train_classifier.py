import sys
import pandas as pd
import numpy as np
import pickle
import re
from sqlalchemy import create_engine

# import tokenize_function
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.corpus import wordnet

# import sklearn
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB

import joblib

def load_data(database_filepath):
    
    """
        Load data from the sqlite database. 
    Args: 
        database_filepath: the path of the database file
    Returns: 
        X (DataFrame): messages 
        Y (DataFrame): One-hot encoded categories
        category_names (List)
    """
    
    # load data from database
    # Create connection to database
    engine = create_engine('sqlite:///data/DisasterResponses.db')
    con = engine.connect()
    
    # Read in data to pandas dataframe
    df = pd.read_sql_table("tables", con)
    
    print("Database connection successful! Returning appropriate data for further processing...")
    
    X = df['message']
    Y = df.drop(['message', 'id', 'original', 'genre'], axis=1)
    category_names = Y.columns
    
    return X, Y, category_names


def tokenize(text):
    """
    Clean the dataset- detect and find url regular expressions, tokenize text, lemmatize text 
    and remove stopwords
    Args:
    url_regex: Regular url expressions
    detected_urls: finding all url expressions
    tokens: tokenized words
    lemmatizer: Lemmatizing text
    Return:
    Clean tokens
    """
    
    #Detect url 
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    
    # get list of all urls using regex
    detected_urls = re.findall(url_regex, text)
    
    # replace each url in text string with placeholder
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")

    # tokenize text and initiate lemmatizer
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    
    # Remove stopwords
    tokens = [w for w in tokens if w not in stopwords.words("english")]

    # iterate through each token
    clean_tokens = []
    for tok in tokens:
        
        # lemmatize, normalize case, and remove leading/trailing white space
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    
    """
      build NLP pipeline - count words, tf-idf, multiple output classifier,
      grid search the best parameters
    Args: 
        None
    Returns: 
        cross validated classifier object
    """   
    
 # specify parameters for grid search
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', RandomForestClassifier())
    ])
    parameters = {
        'clf__min_samples_split': [3, 4],
        'clf__n_estimators': [50, 100]
    }
    cv = GridSearchCV(pipeline, param_grid=parameters, n_jobs= 2, cv=3)
    
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    
    """
        Evaluate the model performances, in terms of f1-score, precison and recall
    Args: 
        model: the model to be evaluated
        X_test: X_test dataframe
        Y_test: Y_test dataframe
        category_names: category names list defined in load data
    Returns: 
        perfomances (DataFrame)
    """  
    
    y_pred = model.predict(X_test)
    
    print(classification_report(Y_test, pd.DataFrame(y_pred, columns=Y_test.columns), target_names=category_names))

    return

def save_model(model, model_filepath):
    
    """
        Save model to pickle
    """
    
    joblib.dump(model, open(model_filepath, 'wb'))

    return


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
