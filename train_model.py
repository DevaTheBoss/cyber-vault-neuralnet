import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import numpy as np

# Trusted domains
TRUSTED_DOMAINS = {'google.com', 'facebook.com', 'amazon.com', 'microsoft.com', 'apple.com'}

# Generate a large dataset (3000 samples)
np.random.seed(42)
n_samples = 3000
data = {
    'has_https': np.random.choice([0, 1], n_samples, p=[0.2, 0.8]),
    'url_length': np.random.randint(15, 350, n_samples),
    'has_ip_address': np.random.choice([0, 1], n_samples, p=[0.9, 0.1]),
    'domain_length': np.random.randint(8, 60, n_samples),
    'has_subdomain': np.random.choice([0, 1], n_samples, p=[0.6, 0.4]),
    'has_at_symbol': np.random.choice([0, 1], n_samples, p=[0.95, 0.05]),
    'has_double_slash': np.random.choice([0, 1], n_samples, p=[0.95, 0.05]),
    'has_suspicious_words': np.random.choice([0, 1], n_samples, p=[0.7, 0.3]),
    'is_trusted_domain': np.random.choice([0, 1], n_samples, p=[0.8, 0.2]),
    'tld_score': np.random.choice([0, 1, 2], n_samples, p=[0.7, 0.2, 0.1]),
    'label': np.random.choice([0, 1], n_samples, p=[0.55, 0.45])
}
df = pd.DataFrame(data)

# Add realistic patterns
df.loc[df['has_ip_address'] == 1, 'label'] = 1
df.loc[df['url_length'] > 120, 'label'] = df['label'] | 1
df.loc[df['has_suspicious_words'] == 1, 'label'] = 1
df.loc[df['has_https'] == 1, 'label'] = df['label'] & np.random.choice([0, 1], n_samples, p=[0.85, 0.15])
df.loc[df['domain_length'] < 15, 'label'] = df['label'] & np.random.choice([0, 1], n_samples, p=[0.9, 0.1])
df.loc[df['is_trusted_domain'] == 1, 'label'] = 0
df.loc[df['tld_score'] == 2, 'label'] = df['label'] | np.random.choice([0, 1], n_samples, p=[0.3, 0.7])

# Split data
X = df.drop('label', axis=1)
y = df['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling for SVM and Logistic Regression
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train Random Forest
rf_model = RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)
print(f"Random Forest Accuracy: {accuracy_score(y_test, rf_pred) * 100:.2f}%")

# Train SVM
svm_model = SVC(probability=True, kernel='rbf', C=1.0, random_state=42)
svm_model.fit(X_train_scaled, y_train)
svm_pred = svm_model.predict(X_test_scaled)
print(f"SVM Accuracy: {accuracy_score(y_test, svm_pred) * 100:.2f}%")

# Train Logistic Regression
lr_model = LogisticRegression(max_iter=1000, random_state=42)
lr_model.fit(X_train_scaled, y_train)
lr_pred = lr_model.predict(X_test_scaled)
print(f"Logistic Regression Accuracy: {accuracy_score(y_test, lr_pred) * 100:.2f}%")

# Save models and scaler
with open('rf_phishing_model.pkl', 'wb') as f:
    pickle.dump(rf_model, f)
with open('svm_phishing_model.pkl', 'wb') as f:
    pickle.dump(svm_model, f)
with open('lr_phishing_model.pkl', 'wb') as f:
    pickle.dump(lr_model, f)
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

print("Models saved as rf_phishing_model.pkl, svm_phishing_model.pkl, lr_phishing_model.pkl")
print("Scaler saved as scaler.pkl")