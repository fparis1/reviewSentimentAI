import pandas as pd
import string
import re
import nltk
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import roc_auc_score, roc_curve

# Ako prvi put koristite nltk stop riječi, otkomentirajte sljedeću liniju:
# nltk.download('stopwords')

# Učitavanje skupa podataka (datoteka 'IMDB Dataset.csv' treba biti u radnom direktoriju)
df = pd.read_csv('IMDB Dataset.csv')
print("Prvih 5 redaka skupa podataka:")
print(df.head())

# Funkcija za čišćenje teksta: mala slova, uklanjanje interpunkcije i stop riječi
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[' + string.punctuation + ']', '', text)
    tokens = text.split()
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(tokens)

# Primjena čišćenja na recenzije
df['clean_review'] = df['review'].apply(clean_text)

# Pretvorba oznaka sentimenta u binarni oblik ('positive' -> 1, 'negative' -> 0)
df['sentiment_binary'] = df['sentiment'].apply(lambda x: 1 if x == 'positive' else 0)

# Podjela podataka na trening i test skup (80%/20%)
X_train, X_test, y_train, y_test = train_test_split(
    df['clean_review'], df['sentiment_binary'], test_size=0.2, random_state=42
)

# Transformacija teksta u TF-IDF značajke
tfidf_vectorizer = TfidfVectorizer()
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

# Treniranje Naivnog Bayesovog klasifikatora
nb_classifier = MultinomialNB()
nb_classifier.fit(X_train_tfidf, y_train)

# Predikcija vjerojatnosti za pozitivnu klasu
y_proba = nb_classifier.predict_proba(X_test_tfidf)[:, 1]

# Evaluacija modela korištenjem ROC-AUC metrike
roc_auc = roc_auc_score(y_test, y_proba)
print("ROC-AUC:", roc_auc)

# Izračunavanje vrijednosti za ROC krivulju
fpr, tpr, thresholds = roc_curve(y_test, y_proba)

# Generiranje grafičkog prikaza ROC krivulje
plt.figure()
plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--')  # linija slučajne klasifikacije
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.show()
