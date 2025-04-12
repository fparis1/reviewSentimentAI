import pandas as pd
import string
import re
import nltk
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, roc_curve

# Ako prvi put koristite nltk stop riječi, otkomentirajte sljedeću liniju:
# nltk.download('stopwords')

# Učitavanje skupa podataka
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

# Kreiranje pipeline-a sa definiranim vrijednostima parametara:
# - TfidfVectorizer: n-gram opseg (1, 2), max_df = 0.85, min_df = 1
# - MultinomialNB: alpha = 0.1
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(ngram_range=(1, 2), max_df=0.85, min_df=1)),
    ('nb', MultinomialNB(alpha=0.1))
])

# Treniranje modela na trening skupu
pipeline.fit(X_train, y_train)

# Evaluacija modela na test skupu
y_proba = pipeline.predict_proba(X_test)[:, 1]
roc_auc = roc_auc_score(y_test, y_proba)
print("\nROC-AUC na test skupu:", roc_auc)

# Izračunavanje vrijednosti za ROC krivulju
fpr, tpr, thresholds = roc_curve(y_test, y_proba)

# Generiranje grafičkog prikaza ROC krivulje
plt.figure()
plt.plot(fpr, tpr, label=f'ROC krivulja (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--')  # linija slučajne klasifikacije
plt.xlabel('Lažno pozitivna stopa (FPR)')
plt.ylabel('Istinski pozitivna stopa (TPR)')
plt.title('ROC krivulja')
plt.legend(loc='lower right')
plt.show()