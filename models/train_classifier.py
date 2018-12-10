import sys
import pandas as pd
from sqlalchemy import create_engine
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,accuracy_score
from sklearn.model_selection import GridSearchCV
import pickle
import nltk
nltk.download('punkt')
nltk.download('wordnet')

def load_data(database_filepath):
    """load data from database.
    
    Args:
        database_filepath: the file path database is stored.
        
    Returns:
        X: features 
        Y: labels
        categories: the string of category names
    
    """
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df = pd.read_sql_table('DisasterResponse',engine)
    X = df.loc[:,'message']
    Y = df.loc[:,'related':'direct_report']
    categories = df.loc[:, 'related':'direct_report'].columns
    return X,Y,categories


def tokenize(text):
    """return tokens of given text.
    
    Args:
        text: the text to be tokenized.
    """
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)
    return clean_tokens



def build_model():
    """Return a decision tree model that is pipelined with proprocessing and grid search for hypaparameters."""
    pipeline = Pipeline([('vect', CountVectorizer(tokenizer=tokenize)),
                     ('tfidf',TfidfTransformer()),
                     ('multiple_output_model',MultiOutputClassifier(DecisionTreeClassifier()))])  
    parameters = [{'multiple_output_model__estimator__max_features': ['sqrt','log2',None], 
                   'multiple_output_model__estimator__max_depth': [10, 30, 50]}]
    cv = GridSearchCV(pipeline,parameters)
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    """Evaluate the model on test set.
    
    Args:
        model: trained model
        X_test: test features
        Y_test: test labels
        category_names: names of categories in model output.
    """
    Y_pred = model.predict(X_test)
    for column in category_names:
        loc = Y_test.columns.get_loc(column)
        true = Y_test[column]
        pred = Y_pred[:,loc]
        print("For column {}'s classificaiton report:".format(column),'\n',classification_report(true,pred))
        print("Accuracy score is ", accuracy_score(true,pred))
    return None


def save_model(model, model_filepath):
    """save model as pickle file."""
    filename = model_filepath
    pickle.dump(model, open(filename, 'wb'))
    return None


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