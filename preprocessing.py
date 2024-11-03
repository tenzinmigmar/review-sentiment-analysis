import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import re

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('omw-1.4')

def advanced_clean_text(text):
    """
    Perform advanced text cleaning including tokenization, lemmatization, and stopword removal.
    """
    if not isinstance(text, str):
        return ''
    

    text = text.lower()
    # remove URLs, email addresses, special characters, extra white space
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\S+@\S+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = ' '.join(text.split())
    
    # tokenize
    tokens = word_tokenize(text)
    
    # remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]
    
    # lemmatization
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    
    return ' '.join(tokens)

def prepare_data(file_path):
    """
    Load and preprocess the data.
    """

    df = pd.read_csv(file_path)
    # clean the text & sentiment labels
    df['Clean_Review'] = df['Review Text'].apply(advanced_clean_text)
    df['Sentiment'] = df['Rating'].apply(lambda x: 'positive' if x >= 4 else ('negative' if x <= 2 else 'neutral'))
    
    # add text statistics
    df['Review_Length'] = df['Clean_Review'].str.len()
    df['Word_Count'] = df['Clean_Review'].str.split().str.len()
    
    return df
