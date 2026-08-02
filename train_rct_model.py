import pandas as pd
import numpy as np
import joblib
import time
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, classification_report

train_df = pd.read_csv("train_parsed.csv")
dev_df = pd.read_csv("dev_parsed.csv")
test_df = pd.read_csv("test_parsed.csv")

for df in (train_df, dev_df, test_df):
    df["text"] = df["text"].fillna("")
    df["line_pct"] = df["line_number"] / df["total_lines"].clip(lower=1)

le = LabelEncoder()
y_train = le.fit_transform(train_df["target"])
y_dev = le.transform(dev_df["target"])
y_test = le.transform(test_df["target"])

print("Fitting TF-IDF...")
t0 = time.time()
tfidf = TfidfVectorizer(max_features=30000, ngram_range=(1, 2), sublinear_tf=True, min_df=2)
X_train_text = tfidf.fit_transform(train_df["text"])
X_dev_text = tfidf.transform(dev_df["text"])
X_test_text = tfidf.transform(test_df["text"])
print(f"TF-IDF done in {time.time()-t0:.1f}s, shape={X_train_text.shape}")


def numeric_features(df):
    return csr_matrix(df[["line_number", "total_lines", "line_pct"]].values.astype(float))


X_train = hstack([X_train_text, numeric_features(train_df)]).tocsr()
X_dev = hstack([X_dev_text, numeric_features(dev_df)]).tocsr()
X_test = hstack([X_test_text, numeric_features(test_df)]).tocsr()

print("Training Logistic Regression...")
t0 = time.time()
clf = LogisticRegression(max_iter=200, C=5.0, n_jobs=-1, solver="saga")
clf.fit(X_train, y_train)
print(f"Training done in {time.time()-t0:.1f}s")

dev_pred = clf.predict(X_dev)
test_pred = clf.predict(X_test)

dev_acc = accuracy_score(y_dev, dev_pred)
test_acc = accuracy_score(y_test, test_pred)
test_f1 = f1_score(y_test, test_pred, average="weighted")

print(f"Dev accuracy: {dev_acc:.4f}")
print(f"Test accuracy: {test_acc:.4f}")
print(f"Test weighted F1: {test_f1:.4f}")
print(classification_report(y_test, test_pred, target_names=le.classes_))

joblib.dump(clf, "rct_model.joblib")
joblib.dump(tfidf, "rct_tfidf.joblib")
joblib.dump(le, "rct_label_encoder.joblib")
joblib.dump(
    {"dev_acc": dev_acc, "test_acc": test_acc, "test_f1": test_f1, "n_train": len(train_df)},
    "rct_metrics.joblib",
)
print("Saved artifacts.")
