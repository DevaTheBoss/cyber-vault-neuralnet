import sqlite3
import os

# Use in-memory database for Render (or a file for local testing)
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

import http.server
import socketserver
import json
import pandas as pd
import pickle
import re
import hashlib
from urllib.parse import urlparse
from datetime import datetime
import os
import logging
from sklearn.preprocessing import StandardScaler

# Setup logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Trusted domains
TRUSTED_DOMAINS = {'google.com', 'facebook.com', 'amazon.com', 'microsoft.com', 'apple.com'}

# Initialize user CSV
USER_CSV = 'users.csv'
if not os.path.exists(USER_CSV):
    pd.DataFrame(columns=['name', 'username', 'password_hash', 'mobile', 'email', 'signup_date']).to_csv(USER_CSV, index=False)

# User activity CSV
USER_ACTIVITY_FILE = 'user_activity.csv'
if not os.path.exists(USER_ACTIVITY_FILE):
    pd.DataFrame(columns=['username', 'timestamp', 'url', 'result']).to_csv(USER_ACTIVITY_FILE, index=False)

# Load pre-trained model (default: Random Forest)
# To use SVM, change to 'svm_phishing_model.pkl'
# To use Logistic Regression, change to 'lr_phishing_model.pkl'
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

# Log user activity
def log_activity(username, url, result):
    try:
        df = pd.DataFrame({
            'username': [username],
            'timestamp': [datetime.now().isoformat()],
            'url': [url],
            'result': [result]
        })
        df.to_csv(USER_ACTIVITY_FILE, mode='a', header=not os.path.exists(USER_ACTIVITY_FILE), index=False)
    except Exception as e:
        logging.error(f"Error logging activity: {e}")

# Hash password
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

class RequestHandler(http.server.BaseHTTPRequestHandler):
    def do_POST(self):
        try:
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length).decode('utf-8')
            data = json.loads(post_data)
            logging.debug(f"Received POST data: {data}")

            if self.path == '/signup':
                name = data.get('name', '').strip()
                username = data.get('username', '').strip()
                password = data.get('password', '').strip()
                mobile = data.get('mobile', '').strip()
                email = data.get('email', '').strip()

                if not all([name, username, password, mobile, email]):
                    self.send_response(400)
                    self.send_header('Content-Type', 'application/json')
                    self.end_headers()
                    self.wfile.write(json.dumps({'error': 'All fields required for SYSTEM REGISTRATION'}).encode('utf-8'))
                    logging.warning("Sign-up failed: Missing fields")
                    return

                try:
                    df = pd.read_csv(USER_CSV)
                except Exception as e:
                    logging.error(f"Error reading users.csv: {e}")
                    self.send_response(500)
                    self.send_header('Content-Type', 'application/json')
                    self.end_headers()
                    self.wfile.write(json.dumps({'error': 'CRITICAL SERVER ERROR'}).encode('utf-8'))
                    return

                if username.lower() in df['username'].str.lower().values:
                    self.send_response(400)
                    self.send_header('Content-Type', 'application/json')
                    self.end_headers()
                    self.wfile.write(json.dumps({'error': 'Username already exists in DATABASE'}).encode('utf-8'))
                    logging.warning(f"Sign-up failed: Username {username} already exists")
                    return

                signup_date = datetime.now().isoformat()
                new_user = pd.DataFrame({
                    'name': [name],
                    'username': [username],
                    'password_hash': [hash_password(password)],
                    'mobile': [mobile],
                    'email': [email],
                    'signup_date': [signup_date]
                })
                try:
                    new_user.to_csv(USER_CSV, mode='a', header=not os.path.exists(USER_CSV), index=False)
                    logging.info(f"User {username} signed up successfully")
                except Exception as e:
                    logging.error(f"Error writing to users.csv: {e}")
                    self.send_response(500)
                    self.send_header('Content-Type', 'application/json')
                    self.end_headers()
                    self.wfile.write(json.dumps({'error': 'CRITICAL DATABASE ERROR'}).encode('utf-8'))
                    return

                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                self.wfile.write(json.dumps({'message': 'SYSTEM REGISTRATION COMPLETE'}).encode('utf-8'))

            elif self.path == '/login':
                username = data.get('username', '').strip()
                password = data.get('password', '').strip()

                if not username or not password:
                    self.send_response(400)
                    self.send_header('Content-Type', 'application/json')
                    self.end_headers()
                    self.wfile.write(json.dumps({'error': 'AUTHENTICATION CREDENTIALS REQUIRED'}).encode('utf-8'))
                    logging.warning("Login failed: Missing username or password")
                    return

                try:
                    df = pd.read_csv(USER_CSV)
                except Exception as e:
                    logging.error(f"Error reading users.csv: {e}")
                    self.send_response(500)
                    self.send_header('Content-Type', 'application/json')
                    self.end_headers()
                    self.wfile.write(json.dumps({'error': 'CRITICAL SERVER ERROR'}).encode('utf-8'))
                    return

                user_data = df[df['username'].str.lower() == username.lower()]
                if user_data.empty:
                    self.send_response(401)
                    self.send_header('Content-Type', 'application/json')
                    self.end_headers()
                    self.wfile.write(json.dumps({'error': 'USER NOT FOUND. INITIATE SYSTEM REGISTRATION.'}).encode('utf-8'))
                    logging.warning(f"Login failed: User {username} does not exist")
                    return

                if user_data['password_hash'].iloc[0] != hash_password(password):
                    self.send_response(401)
                    self.send_header('Content-Type', 'application/json')
                    self.end_headers()
                    self.wfile.write(json.dumps({'error': 'AUTHENTICATION FAILED: INVALID CREDENTIALS'}).encode('utf-8'))
                    logging.warning(f"Login failed: Invalid password for {username}")
                    return

                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                self.wfile.write(json.dumps({'message': 'AUTHENTICATION SUCCESSFUL', 'username': username}).encode('utf-8'))
                logging.info(f"User {username} logged in successfully")

            elif self.path == '/predict':
                url = data.get('url', '').strip()
                username = data.get('username', '').strip()
                if not url or not username:
                    self.send_response(400)
                    self.send_header('Content-Type', 'application/json')
                    self.end_headers()
                    self.wfile.write(json.dumps({'error': 'URL AND USERID REQUIRED FOR SCAN'}).encode('utf-8'))
                    logging.warning("Prediction failed: Missing URL or username")
                    return

                if not is_valid_url(url):
                    self.send_response(400)
                    self.send_header('Content-Type', 'application/json')
                    self.end_headers()
                    self.wfile.write(json.dumps({'error': 'INVALID TARGET VECTOR: VALID TLD REQUIRED'}).encode('utf-8'))
                    logging.warning(f"Prediction failed: Invalid URL {url}")
                    return

                features = extract_features(url)
                # Scale features for SVM or Logistic Regression
                if MODEL_FILE in ['svm_phishing_model.pkl', 'lr_phishing_model.pkl']:
                    features = scaler.transform(features)
                prediction = model.predict(features)[0]
                probability = float(model.predict_proba(features)[0][1])
                result = 'Phishing' if prediction == 1 else 'Legitimate'

                log_activity(username, url, result)

                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                response = {
                    'result': result,
                    'probability': round(probability * 100, 2),
                    'url': url
                }
                self.wfile.write(json.dumps(response).encode('utf-8'))
                logging.info(f"Prediction for {username}: {url} -> {result}")

            else:
                self.send_response(404)
                self.end_headers()

        except Exception as e:
            logging.error(f"Server error: {e}")
            self.send_response(500)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({'error': 'CRITICAL SYSTEM FAILURE: ' + str(e)}).encode('utf-8'))

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()
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
    return jsonify({'message': 'REGISTRATION SUCCESSFUL'}), 200

# Run server
PORT = 5000
with socketserver.TCPServer(("", PORT), RequestHandler) as httpd:
    print(f"Serving at http://localhost:{PORT}")
    logging.info(f"Server started at http://localhost:{PORT}")
    httpd.serve_forever()
if __name__ == '__main__':
    socketio.run(app, debug=True, host='0.0.0.0', port=5000)
else:
    import os
    port = int(os.getenv('PORT', 5000))
    socketio.run(app, host='0.0.0.0', port=port)