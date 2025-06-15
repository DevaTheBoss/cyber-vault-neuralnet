import sqlite3
import os
from flask import Flask, request, jsonify
from flask_socketio import SocketIO, emit
import pandas as pd
import pickle
import re
from urllib.parse import urlparse
from datetime import datetime
import logging
from sklearn.preprocessing import StandardScaler
import bcrypt

# Flask app setup
app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key'  # Needed for Flask-SocketIO
socketio = SocketIO(app, cors_allowed_origins="*")

# Setup logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# SQLite database setup
if os.getenv('RENDER'):
    conn = sqlite3.connect(':memory:', check_same_thread=False)
else:
    conn = sqlite3.connect('database.db', check_same_thread=False)
cursor = conn.cursor()

# Create tables
cursor.execute('''
    CREATE TABLE IF NOT EXISTS users (
        name TEXT, username TEXT PRIMARY KEY, password TEXT, mobile TEXT, email TEXT
    )
''')
cursor.execute('''
    CREATE TABLE IF NOT EXISTS user_activity (
        username TEXT, timestamp TEXT, url TEXT, result TEXT
    )
''')
conn.commit()

# Trusted domains
TRUSTED_DOMAINS = {'google.com', 'facebook.com', 'amazon.com', 'microsoft.com', 'apple.com'}

# Load pre-trained model (default: Random Forest)
MODEL_FILE = 'rf_phishing_model.pkl'
try:
    with open(MODEL_FILE, 'rb') as f:
        model = pickle.load(f)
except FileNotFoundError:
    logging.error(f"Model file {MODEL_FILE} not found. Please train the model first.")
    exit(1)

# Load scaler for SVM and Logistic Regression
try:
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
except FileNotFoundError:
    logging.error("Scaler file scaler.pkl not found. Please train the model first.")
    exit(1)

# URL validation function
def is_valid_url(url):
    if not url:
        return False
    valid_tlds = ('.co', '.in', '.com')
    return any(url.lower().endswith(tld) for tld in valid_tlds)

# Normalize URL for feature extraction
def normalize_url(url):
    if not url.lower().startswith(('http://', 'https://')):
        return 'http://' + url
    return url

# Feature extraction functions
def has_https(url):
    return 1 if url.lower().startswith('https') else 0

def url_length(url):
    return len(url)

def has_ip_address(url):
    ip_pattern = r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b'
    return 1 if re.search(ip_pattern, url) else 0

def domain_length(url):
    parsed = urlparse(url)
    return len(parsed.netloc) if parsed.netloc else len(url)

def has_subdomain(url):
    parsed = urlparse(url)
    return 1 if parsed.netloc and len(parsed.netloc.split('.')) > 2 else 0

def has_at_symbol(url):
    return 1 if '@' in url else 0

def has_double_slash(url):
    return 1 if '//' in url[8:] else 0

def has_suspicious_words(url):
    suspicious = ['login', 'signin', 'verify', 'account', 'secure', 'update']
    return 1 if any(word in url.lower() for word in suspicious) else 0

def is_trusted_domain(url):
    parsed = urlparse(url)
    domain = parsed.netloc.lower()
    return 1 if any(trusted in domain for trusted in TRUSTED_DOMAINS) else 0

def tld_score(url):
    parsed = urlparse(url)
    tld = parsed.netloc.split('.')[-1].lower() if parsed.netloc else ''
    if tld in {'com', 'org', 'edu'}:
        return 0
    elif tld in {'net', 'info', 'biz'}:
        return 1
    else:
        return 2

def extract_features(url):
    normalized_url = normalize_url(url)
    features = {
        'has_https': has_https(normalized_url),
        'url_length': url_length(normalized_url),
        'has_ip_address': has_ip_address(normalized_url),
        'domain_length': domain_length(normalized_url),
        'has_subdomain': has_subdomain(normalized_url),
        'has_at_symbol': has_at_symbol(normalized_url),
        'has_double_slash': has_double_slash(normalized_url),
        'has_suspicious_words': has_suspicious_words(normalized_url),
        'is_trusted_domain': is_trusted_domain(normalized_url),
        'tld_score': tld_score(normalized_url)
    }
    return pd.DataFrame([features])

# Log user activity to SQLite
def log_activity(username, url, result):
    try:
        cursor.execute(
            "INSERT INTO user_activity (username, timestamp, url, result) VALUES (?, ?, ?, ?)",
            (username, datetime.now().isoformat(), url, result)
        )
        conn.commit()
        # Emit live update to all connected clients
        socketio.emit('activity_update', {'username': username, 'url': url, 'result': result})
    except Exception as e:
        logging.error(f"Error logging activity: {e}")

# Flask routes
@app.route('/signup', methods=['POST'])
def signup():
    data = request.get_json()
    cursor.execute("SELECT username FROM users WHERE username = ?", (data['username'],))
    if cursor.fetchone():
        return jsonify({'error': 'USERID ALREADY EXISTS'}), 400
    hashed_password = bcrypt.hashpw(data['password'].encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
    cursor.execute(
        "INSERT INTO users (name, username, password, mobile, email) VALUES (?, ?, ?, ?, ?)",
        (data['name'], data['username'], hashed_password, data['mobile'], data['email'])
    )
    conn.commit()
    socketio.emit('user_update', {'username': data['username']})
    logging.info(f"User {data['username']} signed up successfully")
    return jsonify({'message': 'REGISTRATION SUCCESSFUL'}), 200

@app.route('/login', methods=['POST'])
def login():
    data = request.get_json()
    username = data.get('username', '').strip()
    password = data.get('password', '').strip()

    if not username or not password:
        return jsonify({'error': 'AUTHENTICATION CREDENTIALS REQUIRED'}), 400

    cursor.execute("SELECT * FROM users WHERE username = ?", (username,))
    user = cursor.fetchone()
    if not user:
        logging.warning(f"Login failed: User {username} does not exist")
        return jsonify({'error': 'USER NOT FOUND. INITIATE SYSTEM REGISTRATION.'}), 401

    # user is a tuple: (name, username, password, mobile, email)
    stored_password = user[2]
    if not bcrypt.checkpw(password.encode('utf-8'), stored_password.encode('utf-8')):
        logging.warning(f"Login failed: Invalid password for {username}")
        return jsonify({'error': 'AUTHENTICATION FAILED: INVALID CREDENTIALS'}), 401

    logging.info(f"User {username} logged in successfully")
    return jsonify({'message': 'AUTHENTICATION SUCCESSFUL', 'username': username}), 200

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    url = data.get('url', '').strip()
    username = data.get('username', '').strip()

    if not url or not username:
        logging.warning("Prediction failed: Missing URL or username")
        return jsonify({'error': 'URL AND USERID REQUIRED FOR SCAN'}), 400

    if not is_valid_url(url):
        logging.warning(f"Prediction failed: Invalid URL {url}")
        return jsonify({'error': 'INVALID TARGET VECTOR: VALID TLD REQUIRED'}), 400

    features = extract_features(url)
    if MODEL_FILE in ['svm_phishing_model.pkl', 'lr_phishing_model.pkl']:
        features = scaler.transform(features)
    prediction = model.predict(features)[0]
    probability = float(model.predict_proba(features)[0][1])
    result = 'Phishing' if prediction == 1 else 'Legitimate'

    log_activity(username, url, result)

    response = {
        'result': result,
        'probability': round(probability * 100, 2),
        'url': url
    }
    logging.info(f"Prediction for {username}: {url} -> {result}")
    return jsonify(response), 200

@app.route('/')
def serve_index():
    return app.send_static_file('index.html')

# Run the app
if __name__ == '__main__':
    socketio.run(app, debug=True, host='0.0.0.0', port=5000)
else:
    import os
    port = int(os.getenv('PORT', 5000))
    socketio.run(app, host='0.0.0.0', port=port)