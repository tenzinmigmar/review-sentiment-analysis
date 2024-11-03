from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

class NaiveBayesModel:
    def __init__(self):
        self.pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(max_features=5000)),
            ('classifier', MultinomialNB())
        ])
        self.model_name = 'Naive Bayes'
    
    def train_and_evaluate(self, df):
        """
        Train and evaluate the Naive Bayes model.
        """
        # prepare data
        df_sentiment = df[df['Sentiment'] != 'neutral'].copy()
        X = df_sentiment['Clean_Review']
        y = (df_sentiment['Sentiment'] == 'positive').astype(int)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # train
        self.pipeline.fit(X_train, y_train)
        
        # predict
        y_pred = self.pipeline.predict(X_test)
        
        # calculate metrics
        accuracy = accuracy_score(y_test, y_pred)

        print(f"\nClassification Report for {self.model_name}:")
        print(classification_report(y_test, y_pred))
        
        # confusion matrix
        plt.figure(figsize=(8, 6))
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix - {self.model_name}')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.show()
        
        return accuracy
    
    def cross_validate(self, df, cv=5):
        """
        Perform cross-validation.
        """
        df_sentiment = df[df['Sentiment'] != 'neutral'].copy()
        X = df_sentiment['Clean_Review']
        y = (df_sentiment['Sentiment'] == 'positive').astype(int)
        
        scores = cross_val_score(self.pipeline, X, y, cv=cv)
        print(f"\n{self.model_name} Cross-Validation Scores:")
        print(f"Mean Accuracy: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")
        
        return scores
