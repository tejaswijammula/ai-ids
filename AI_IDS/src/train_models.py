# # AI-IDS using RF/LSTM/CNN/Encemble

# In[ ]:


# AI-Driven IDS using NSL-KDD Dataset (Google Colab Friendly)

# Step 1: Install required packages
print("Installing packages...")
get_ipython().system('pip install -q pandas numpy scikit-learn matplotlib seaborn tensorflow')

# Step 2: Import Libraries
print("Importing libraries...")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from tensorflow.keras.layers import LSTM, Dense, Dropout, Conv1D, MaxPooling1D, Flatten
from tensorflow.keras.callbacks import EarlyStopping

# Step 3: Load Dataset from Google Drive
print("Loading NSL-KDD dataset from Google Drive...")
url_train = '/content/drive/My Drive/NSL/KDDTrain20Percent.txt'
url_test = '/content/drive/My Drive/NSL/KDDTest-21.txt'

column_names = [
    'duration','protocol_type','service','flag','src_bytes','dst_bytes','land','wrong_fragment','urgent',
    'hot','num_failed_logins','logged_in','num_compromised','root_shell','su_attempted','num_root',
    'num_file_creations','num_shells','num_access_files','num_outbound_cmds','is_host_login','is_guest_login',
    'count','srv_count','serror_rate','srv_serror_rate','rerror_rate','srv_rerror_rate','same_srv_rate',
    'diff_srv_rate','srv_diff_host_rate','dst_host_count','dst_host_srv_count','dst_host_same_srv_rate',
    'dst_host_diff_srv_rate','dst_host_same_src_port_rate','dst_host_srv_diff_host_rate','dst_host_serror_rate',
    'dst_host_srv_serror_rate','dst_host_rerror_rate','dst_host_srv_rerror_rate','label','difficulty'
]

train_df = pd.read_csv(url_train, names=column_names)
test_df = pd.read_csv(url_test, names=column_names)
print("Train shape:", train_df.shape, " Test shape:", test_df.shape)

# Step 4: Binary Classification (Normal vs Attack)
print("Converting labels to binary...")
train_df['label'] = train_df['label'].apply(lambda x: 0 if x == 'normal' else 1)
test_df['label'] = test_df['label'].apply(lambda x: 0 if x == 'normal' else 1)

# Step 5: One-Hot Encoding for Categorical Features
print("Applying one-hot encoding...")
categorical_cols = ['protocol_type', 'service', 'flag']
full_df = pd.concat([train_df, test_df])
full_df = pd.get_dummies(full_df, columns=categorical_cols)

train_encoded = full_df.iloc[:len(train_df)].copy()
test_encoded = full_df.iloc[len(train_df):].copy()

# Step 6: Scale and separate features/labels
print("Cleaning non-numeric data and scaling features...")
X_train = train_encoded.drop(['label', 'difficulty'], axis=1).apply(pd.to_numeric, errors='coerce')
y_train = train_encoded['label']
X_test = test_encoded.drop(['label', 'difficulty'], axis=1).apply(pd.to_numeric, errors='coerce')
y_test = test_encoded['label']

X_train.fillna(0, inplace=True)
X_test.fillna(0, inplace=True)

rf = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
try:
    y_prob_rf = rf.predict_proba(X_test)[:, 1]
except IndexError:
    y_prob_rf = rf.predict_proba(X_test)[:, 0]
print("\nRandom Forest Report:")
# Step 8: LSTM Model (requires 3D input)
print("Preparing data and training LSTM model...")
X_train_lstm = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
X_test_lstm = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))

lstm_model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(1, X_train.shape[1])),
    Dropout(0.3),
    LSTM(32),
    Dropout(0.3),
    Dense(1, activation='sigmoid')
])

lstm_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
lstm_model.fit(X_train_lstm, y_train, epochs=10, batch_size=32, validation_split=0.2, callbacks=[EarlyStopping(patience=3)])
y_pred_lstm = (lstm_model.predict(X_test_lstm) > 0.5).astype("int32")
print("LSTM Report:")
# Step 9: CNN Model
print("Preparing data and training CNN model...")
X_train_cnn = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test_cnn = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

cnn_model = Sequential([
    Conv1D(64, kernel_size=3, activation='relu', input_shape=(X_train.shape[1], 1)),
    MaxPooling1D(pool_size=2),
    Flatten(),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid')
])

cnn_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
cnn_model.fit(X_train_cnn, y_train, epochs=10, batch_size=32, validation_split=0.2, callbacks=[EarlyStopping(patience=3)])
y_pred_cnn = (cnn_model.predict(X_test_cnn) > 0.5).astype("int32")
print("CNN Report:")
# Step 10b: Confusion Matrices for LSTM and CNN
plt.title("LSTM Confusion Matrix")
plt.show()

plt.title("CNN Confusion Matrix")
plt.show()

# Step 10c: ROC Curves
from sklearn.metrics import roc_curve, auc

# Get probabilities
y_prob_lstm = lstm_model.predict(X_test_lstm).flatten()
y_prob_cnn = cnn_model.predict(X_test_cnn).flatten()

fpr_rf, tpr_rf, _ = roc_curve(y_test, y_prob_rf)
fpr_lstm, tpr_lstm, _ = roc_curve(y_test, y_prob_lstm)
fpr_cnn, tpr_cnn, _ = roc_curve(y_test, y_prob_cnn)

plt.figure(figsize=(8,6))
plt.plot(fpr_rf, tpr_rf, label=f'Random Forest (AUC = {auc(fpr_rf, tpr_rf):.2f})')
plt.plot(fpr_lstm, tpr_lstm, label=f'LSTM (AUC = {auc(fpr_lstm, tpr_lstm):.2f})')
plt.plot(fpr_cnn, tpr_cnn, label=f'CNN (AUC = {auc(fpr_cnn, tpr_cnn):.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves for All Models')
plt.legend()
plt.grid(True)
plt.show()

# Step 11: F1-score Comparison
f1_rf = f1_score(y_test, y_pred_rf)
f1_lstm = f1_score(y_test, y_pred_lstm)
f1_cnn = f1_score(y_test, y_pred_cnn)
f1_ensemble = f1_score(y_test, y_pred_ensemble)

models = ['Random Forest', 'LSTM', 'CNN', 'Ensemble']
scores = [f1_rf, f1_lstm, f1_cnn, f1_ensemble]

plt.figure(figsize=(8, 5))
sns.barplot(x=models, y=scores)
plt.title('F1-Score Comparison Across Models')
plt.ylabel('F1-Score')
plt.ylim(0, 1)
plt.grid(True)
plt.show()

print("All steps completed successfully!")
