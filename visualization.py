import matplotlib.pyplot as plt
import seaborn as sns
from nltk.tokenize import word_tokenize
from nltk.util import ngrams
from collections import Counter
from wordcloud import WordCloud
import warnings
warnings.filterwarnings('ignore')

def plot_basic_stats(df):
    """
    Plot basic statistics visualizations.
    """
    plt.style.use('seaborn')
    fig, axes = plt.subplots(2, 2, figsize=(20, 16))

    # rating distribution
    sns.countplot(data=df, x='Rating', ax=axes[0,0])
    axes[0,0].set_title('Distribution of Ratings')
    
    # age distribution
    sns.histplot(data=df, x='Age', bins=30, ax=axes[0,1])
    axes[0,1].set_title('Age Distribution of Reviewers')
    
    # department-wise rating
    sns.boxplot(data=df, x='Department Name', y='Rating', ax=axes[1,0])
    axes[1,0].set_title('Rating Distribution by Department')
    axes[1,0].tick_params(axis='x', rotation=45)
    
    # sentiment distribution
    sns.countplot(data=df, x='Sentiment', ax=axes[1,1])
    axes[1,1].set_title('Distribution of Sentiments')
    
    plt.tight_layout()
    plt.show()

def plot_text_stats(df):
    """
    Plot text statistics visualizations.
    """
    fig, axes = plt.subplots(2, 2, figsize=(20, 16))
    
    # review length distribution
    sns.histplot(data=df, x='Review_Length', bins=50, ax=axes[0,0])
    axes[0,0].set_title('Distribution of Review Lengths')
    
    # word count distribution
    sns.histplot(data=df, x='Word_Count', bins=50, ax=axes[0,1])
    axes[0,1].set_title('Distribution of Word Counts')
    
    # avg word count by rating
    avg_words = df.groupby('Rating')['Word_Count'].mean()
    avg_words.plot(kind='bar', ax=axes[1,0])
    axes[1,0].set_title('Average Word Count by Rating')
    
    # average word count by department
    avg_dept_words = df.groupby('Department Name')['Word_Count'].mean()
    avg_dept_words.plot(kind='bar', ax=axes[1,1])
    axes[1,1].set_title('Average Word Count by Department')
    
    plt.tight_layout()
    plt.show()

def plot_top_ngrams(text, n, top_k=20):
    """
    Plot top n-grams from text.
    """
    tokens = word_tokenize(text)
    ngram_list = list(ngrams(tokens, n))
    ngram_freq = Counter(ngram_list)
    
    # get top k n-grams
    top_ngrams = dict(sorted(ngram_freq.items(), key=lambda x: x[1], reverse=True)[:top_k])

    plt.figure(figsize=(12, 6))
    plt.bar(range(len(top_ngrams)), list(top_ngrams.values()))
    plt.xticks(range(len(top_ngrams)), [' '.join(ng) for ng in top_ngrams.keys()], rotation=45, ha='right')
    plt.title(f'Top {top_k} {n}-grams')
    plt.xlabel('N-gram')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.show()

def analyze_ngrams(df):
    """
    Analyze and plot n-grams for positive and negative reviews.
    """
    positive_text = ' '.join(df[df['Sentiment'] == 'positive']['Clean_Review'])
    negative_text = ' '.join(df[df['Sentiment'] == 'negative']['Clean_Review'])
    
    print("Top Bigrams in Positive Reviews:")
    plot_top_ngrams(positive_text, 2)
    
    print("\nTop Bigrams in Negative Reviews:")
    plot_top_ngrams(negative_text, 2)
