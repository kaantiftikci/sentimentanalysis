import pandas as pd
import nltk
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from textblob import TextBlob

yandex_data = pd.read_excel('chatgpt_translate.xlsx')
knidos_data = pd.read_excel('hotel_reviews.xlsx')

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

def preprocess_text(text):
    if isinstance(text, str):
        text = text.lower()
        text = re.sub("[^a-zA-Z]", " ", text)
        words = nltk.word_tokenize(text)
        stop_words = set(stopwords.words("english"))
        words = [word for word in words if not word in stop_words]

        lemmatizer = WordNetLemmatizer()
        words = [lemmatizer.lemmatize(word) for word in words]

        processed_text = " ".join(words)
        return processed_text
    else:
        return ""

yandex_data['Clean_Review'] = yandex_data['review'].apply(preprocess_text)
knidos_data['Clear_Review'] = knidos_data['hotel_review'].apply(preprocess_text)

classifiers = {
    'Naive Bayes': MultinomialNB(),
    'Decision Tree': DecisionTreeClassifier(),
    'Gradient Boosting': GradientBoostingClassifier(),
    'KNN': KNeighborsClassifier(),
    'Random Forest': RandomForestClassifier()
}

results = pd.DataFrame(columns=['Review', 'TextBlob_Sentiment'])

for classifier_name, classifier in classifiers.items():

    X_train, X_test, y_train, y_test = train_test_split(yandex_data['Clean_Review'], yandex_data['sentiment'], test_size=0.2, random_state=42)
    vectorizer = CountVectorizer()
    X_train_vectors = vectorizer.fit_transform(X_train)
    X_test_vectors = vectorizer.transform(X_test)

    classifier.fit(X_train_vectors, y_train)

    accuracy = classifier.score(X_test_vectors, y_test)
    print(f"{classifier_name} - {accuracy}")

    knidos_data['TextBlob_Sentiment'] = knidos_data['Clear_Review'].apply(lambda x: TextBlob(x).sentiment.polarity)
    knidos_data[f'{classifier_name}_Model_Sentiment'] = classifier.predict(vectorizer.transform(knidos_data['Clear_Review']))

    results = pd.concat([results, knidos_data[['hotel_review', 'Clear_Review', 'TextBlob_Sentiment', f'{classifier_name}_Model_Sentiment']]], axis=1)

output_file = 'chatgpt_new_sentiment_analysis_results.xlsx'
with pd.ExcelWriter(output_file) as writer:
    results.to_excel(writer, sheet_name='Chatgpt_New_Sentiment_Analysis_Results', index=False)
