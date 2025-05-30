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
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve, confusion_matrix
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import ttk
import numpy as np
import sys

# Status: Starting execution
print("[Status] Starting sentiment analysis GUI script execution.")

# Uncomment if needed to download stopwords
# nltk.download('stopwords')

# Status: Loading dataset
print("[Status] Loading dataset from 'IMDB Dataset.csv'.")
DF = pd.read_csv('IMDB Dataset.csv')
print(f"[Status] Dataset loaded: {len(DF)} records.")

# Define text cleaning function
print("[Status] Defining text cleaning function...")
def clean_text(text):
    text = text.lower()
    text = re.sub(fr"[{string.punctuation}]", "", text)
    tokens = text.split()
    sw = set(stopwords.words('english'))
    tokens = [w for w in tokens if w not in sw]
    return ' '.join(tokens)

# Status: Preprocessing text
print(f"[Status] Cleaning {len(DF)} reviews...")
DF['clean_review'] = DF['review'].apply(clean_text)
print("[Status] Text cleaning complete.")

# Convert labels
print("[Status] Converting sentiment labels to binary...")
DF['sentiment_binary'] = DF['sentiment'].map({'positive': 1, 'negative': 0})

# Status: Splitting dataset
print("[Status] Splitting data into train/test (80/20)...")
X_train, X_test, y_train, y_test = train_test_split(
    DF['clean_review'], DF['sentiment_binary'], test_size=0.2, random_state=42
)
print(f"[Status] Split done: {len(X_train)} train, {len(X_test)} test samples.")

# Build and train model
print("[Status] Building pipeline and training model...")
model = Pipeline([
    ('tfidf', TfidfVectorizer(ngram_range=(1, 2), max_df=0.85, min_df=1)),
    ('nb', MultinomialNB(alpha=0.1))
])
model.fit(X_train, y_train)
print("[Status] Model training complete.")

# Predictions & metrics
print("[Status] Generating predictions and computing metrics.")
y_proba = model.predict_proba(X_test)[:, 1]
roc_auc = roc_auc_score(y_test, y_proba)
fpr, tpr, _ = roc_curve(y_test, y_proba)
prec, rec, _ = precision_recall_curve(y_test, y_proba)
y_pred = (y_proba >= 0.5).astype(int)
cm = confusion_matrix(y_test, y_pred)
print(f"[Status] ROC-AUC: {roc_auc:.4f}")

# Define plot functions
print("[Status] Defining plot functions.")

def plot_roc():
    print("[Status] Plotting ROC Curve.")
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, label=f'ROC (AUC = {roc_auc:.2f})')
    ax.plot([0,1], [0,1], 'k--')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curve')
    ax.legend()
    fig.tight_layout()
    return fig


def plot_pr():
    print("[Status] Plotting Precision-Recall Curve.")
    fig, ax = plt.subplots()
    ax.plot(rec, prec, label='Precision-Recall')
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Precision-Recall Curve')
    ax.legend()
    fig.tight_layout()
    return fig


def plot_cm():
    print("[Status] Plotting Confusion Matrix.")
    fig, ax = plt.subplots()
    cax = ax.matshow(cm, cmap='Blues')
    for (i, j), val in np.ndenumerate(cm):
        ax.text(j, i, val, ha='center', va='center')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title('Confusion Matrix')
    fig.colorbar(cax)
    fig.tight_layout()
    return fig


def plot_prob_hist():
    print("[Status] Plotting Predicted-Probability Histogram.")
    fig, ax = plt.subplots()
    ax.hist(y_proba, bins=30, edgecolor='k')
    ax.set_xlabel('Predicted Positive Probability')
    ax.set_ylabel('Count')
    ax.set_title('Distribution of Predicted Probabilities')
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig

plot_funcs = {
    'ROC Curve': plot_roc,
    'Precision-Recall': plot_pr,
    'Confusion Matrix': plot_cm,
    'Probability Histogram': plot_prob_hist
}

# Build Tkinter GUI
print("[Status] Initializing Tkinter GUI.")
root = tk.Tk()
root.title('Sentiment Analysis Graph Visualizer')
root.geometry('900x600')

# Handle window close to exit cleanly
print("[Status] Setting up close handler.")
def on_closing():
    print("[Status] Closing GUI and exiting.")
    root.destroy()
    sys.exit(0)
root.protocol("WM_DELETE_WINDOW", on_closing)

nav_frame = ttk.Frame(root, width=200)
nav_frame.pack(side=tk.LEFT, fill=tk.Y)

graph_frame = ttk.Frame(root)
graph_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

# Function to display graphs
print("[Status] Defining graph display function.")
def display_graph(name):
    print(f"[Status] Displaying graph: {name}")
    for widget in graph_frame.winfo_children():
        widget.destroy()
    fig = plot_funcs[name]()  # generate fresh figure
    canvas = FigureCanvasTkAgg(fig, master=graph_frame)
    canvas.draw()
    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

# Add navigation buttons
print("[Status] Adding navigation buttons.")
for name in plot_funcs.keys():
    btn = ttk.Button(nav_frame, text=name, command=lambda n=name: display_graph(n))
    btn.pack(fill=tk.X, padx=5, pady=5)

# Show default graph
print("[Status] Displaying default graph (ROC Curve).")
display_graph('ROC Curve')

print("[Status] Entering GUI main loop.")
root.mainloop()
