import tkinter as tk
from tkinter import messagebox, filedialog
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd

# GUI Setup
root = tk.Tk()
root.title('URL Classification')
root.geometry('500x300')

url_entry = tk.Entry(root, width=50)
url_entry.pack(pady=10)

# Modules
file_path = None
vectorizer = None
model = None
accuracy = None

# Upload Dataset Module
def upload_dataset():
    global file_path
    file_path = filedialog.askopenfilename(filetypes=[('CSV Files', '*.csv')])
    if file_path:
        messagebox.showinfo('Success', 'Dataset uploaded successfully!')
    else:
        messagebox.showerror('Error', 'No file selected!')

upload_button = tk.Button(root, text='Upload Dataset', command=upload_dataset)
upload_button.pack(pady=5)

# Preprocess Dataset Module
def preprocess_dataset():
    global vectorizer, X, y
    if not file_path:
        messagebox.showerror('Error', 'Please upload a dataset first!')
        return
    df = pd.read_csv(file_path)
    X = df['url'].astype(str)  # Ensure all URLs are strings
    y = df['type'].astype(str).apply(lambda x: x.lower())  # Ensure consistent labeling
    vectorizer = TfidfVectorizer()
    messagebox.showinfo('Success', 'Preprocessing completed!')

preprocess_button = tk.Button(root, text='Preprocess Dataset', command=preprocess_dataset)
preprocess_button.pack(pady=5)

# Train Model Module
def train_model():
    global model, accuracy
    if vectorizer is None:
        messagebox.showerror('Error', 'Please preprocess the dataset first!')
        return
    X_vectorized = vectorizer.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_vectorized, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=200, random_state=42)  # Increased estimators for better accuracy
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    accuracy_label.config(text=f'Model Accuracy: {accuracy * 100:.2f}%')
    messagebox.showinfo('Success', 'Model training completed!')

train_button = tk.Button(root, text='Train Model', command=train_model)
train_button.pack(pady=5)

# URL Classification Module
def classify_url():
    if model is None or vectorizer is None:
        messagebox.showerror('Error', 'Please train the model first!')
        return
    url = url_entry.get()
    if not url:
        messagebox.showwarning('Warning', 'Please enter a URL')
        return
    url_vectorized = vectorizer.transform([url])
    prediction = model.predict(url_vectorized)[0]
    if prediction in ['benign', 'defacement', 'malware', 'phishing']:
        messagebox.showinfo('Prediction', f'This URL is classified as: {prediction}')
    else:
        messagebox.showinfo('Prediction', 'Unknown classification')

classify_button = tk.Button(root, text='Classify URL', command=classify_url)
classify_button.pack(pady=10)

accuracy_label = tk.Label(root, text='Model Accuracy: N/A', fg='blue')
accuracy_label.pack(pady=5)

root.mainloop()
