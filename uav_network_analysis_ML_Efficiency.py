import scapy.all as scapy
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load pcap file using Scapy
packets = scapy.rdpcap('UAV_WiFi_NetworkTraffic.pcap')
previous_packet_time = None

# Extract relevant features from packets and store in Pandas DataFrame
features = []
for packet in packets:
    feature = {}
    if packet.haslayer(scapy.IP):
        feature['src_ip'] = packet[scapy.IP].src
        feature['dst_ip'] = packet[scapy.IP].dst
        feature['protocol'] = packet[scapy.IP].proto
    feature['src_port'] = packet[scapy.TCP].sport if scapy.TCP in packet else None
    feature['dst_port'] = packet[scapy.TCP].dport if scapy.TCP in packet else None
    feature['packet_length'] = len(packet)
    feature['time_delta'] = packet.time - previous_packet_time if previous_packet_time is not None else 0
    #feature['protocol'] = packet[scapy.IP].proto
    features.append(feature)
    previous_packet_time = packet.time

df = pd.DataFrame(features)

# Convert categorical data to numerical data
df = pd.get_dummies(df, columns=['protocol'])

# Define target variable (i.e., whether packet is an attack or not)
attacks = ['jamming', 'replay']
df['jamming'] = [1 if 'WIFI jamming attack' in packet.summary() else 0 for packet in packets]
df['replay'] = [1 if 'reply attack' in packet.summary() else 0 for packet in packets]
df['is_attack'] = df[attacks].max(axis=1)

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df.drop(attacks + ['is_attack'], axis=1), df[attacks], test_size=0.2, random_state=42)

# Train Naive Bayes model
nb = GaussianNB()
nb.fit(X_train, y_train)
y_pred_nb = nb.predict(X_test)
print('Naive Bayes:')
print(f'Jamming - Precision: {precision_score(y_test["jamming"], y_pred_nb[:, 0])}')
print(f'Jamming - Recall: {recall_score(y_test["jamming"], y_pred_nb[:, 0])}')
print(f'Jamming - F1 score: {f1_score(y_test["jamming"], y_pred_nb[:, 0])}')
print(f'Replay - Precision: {precision_score(y_test["replay"], y_pred_nb[:, 1])}')
print(f'Replay - Recall: {recall_score(y_test["replay"], y_pred_nb[:, 1])}')
print(f'Replay - F1 score: {f1_score(y_test["replay"], y_pred_nb[:, 1])}\n')

# Train Support Vector Machines model
svm = SVC()
svm.fit(X_train, y_train)
y_pred_svm = svm.predict(X_test)
print('Support Vector Machines:')
print(f'Jamming - Precision: {precision_score(y_test["jamming"], y_pred_svm[:, 0])}')
print(f'Jamming - Recall: {recall_score(y_test["jamming"], y_pred_svm[:, 0])}')
print(f'Jamming - F1 score: {f1_score(y_test["jamming"], y_pred_svm[:, 0])}')
print(f'Replay - Precision: {precision_score(y_test["replay"], y_pred_svm[:, 1])}')
print(f'Replay - Recall: {recall_score(y_test["replay"], y_pred_svm[:, 1])}')
print(f'Replay - F1 score: {f1_score(y_test["replay"], y_pred_svm[:, 1])}\n')

# Train Random Forest model
rf = RandomForestClassifier()
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
print('Random Forest:')
print(f'Jamming - Precision: {precision_score(y_test["jamming"], y_pred_rf[:, 0])}')
print(f'Jamming - Recall: {recall_score(y_test["jamming"], y_pred_rf[:, 0])}')
print(f'Jamming - F1 score: {f1_score(y_test["jamming"], y_pred_rf[:, 0])}')
print(f'Replay - Precision: {precision_score(y_test["replay"], y_pred_rf[:, 1])}')
print(f'Replay - Recall: {recall_score(y_test["replay"], y_pred_rf[:, 1])}')
print(f'Replay - F1 score: {f1_score(y_test["replay"], y_pred_rf[:, 1])}\n')

# Train Gradient Boosting model
gb = GradientBoostingClassifier()
gb.fit(X_train, y_train)
y_pred_gb = gb.predict(X_test)
print('Gradient Boosting:')
print(f'Jamming - Precision: {precision_score(y_test["jamming"], y_pred_gb[:, 0])}')
print(f'Jamming - Recall: {recall_score(y_test["jamming"], y_pred_gb[:, 0])}')
print(f'Jamming - F1 score: {f1_score(y_test["jamming"], y_pred_gb[:, 0])}')
print(f'Replay - Precision: {precision_score(y_test["replay"], y_pred_gb[:, 1])}')
print(f'Replay - Recall: {recall_score(y_test["replay"], y_pred_gb[:, 1])}')
print(f'Replay - F1 score: {f1_score(y_test["replay"], y_pred_gb[:, 1])}\n')

# Train Logistic Regression model
lr = LogisticRegression()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)
print('Logistic Regression:')
print(f'Jamming - Precision: {precision_score(y_test["jamming"], y_pred_lr[:, 0])}')
print(f'Jamming - Recall: {recall_score(y_test["jamming"], y_pred_lr[:, 0])}')
print(f'Jamming - F1 score: {f1_score(y_test["jamming"], y_pred_lr[:, 0])}')
print(f'Replay - Precision: {precision_score(y_test["replay"], y_pred_lr[:, 1])}')
print(f'Replay - Recall: {recall_score(y_test["replay"], y_pred_lr[:, 1])}')
print(f'Replay - F1 score: {f1_score(y_test["replay"], y_pred_lr[:, 1])}\n')

# Train Decision Trees model
dt = DecisionTreeClassifier()
dt.fit(X_train, y_train)
y_pred_dt = dt.predict(X_test)
print('Decision Trees:')
print(f'Jamming - Precision: {precision_score(y_test["jamming"], y_pred_dt[:, 0])}')
print(f'Jamming - Recall: {recall_score(y_test["jamming"], y_pred_dt[:, 0])}')
print(f'Jamming - F1 score: {f1_score(y_test["jamming"], y_pred_dt[:, 0])}')
print(f'Replay - Precision: {precision_score(y_test["replay"], y_pred_dt[:, 1])}')
print(f'Replay - Recall: {recall_score(y_test["replay"], y_pred_dt[:, 1])}')
print(f'Replay - F1 score: {f1_score(y_test["replay"], y_pred_dt[:, 1])}\n')

# Train K-Nearest Neighbors model
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)
y_pred_knn = knn.predict(X_test)
print('K-Nearest Neighbors:')
print(f'Jamming - Precision: {precision_score(y_test["jamming"], y_pred_knn[:, 0])}')
print(f'Jamming - Recall: {recall_score(y_test["jamming"], y_pred_knn[:, 0])}')
print(f'Jamming - F1 score: {f1_score(y_test["jamming"], y_pred_knn[:, 0])}')
print(f'Replay - Precision: {precision_score(y_test["replay"], y_pred_knn[:, 1])}')
print(f'Replay - Recall: {recall_score(y_test["replay"], y_pred_knn[:, 1])}')
print(f'Replay - F1 score: {f1_score(y_test["replay"], y_pred_knn[:, 1])}\n')

# Train Neural Networks model
nn = MLPClassifier(hidden_layer_sizes=(50,), max_iter=1000)
nn.fit(X_train, y_train)
y_pred_nn = nn.predict(X_test)
print('Neural Networks:')
print(f'Jamming - Precision: {precision_score(y_test["jamming"], y_pred_nn[:, 0])}')
print(f'Jamming - Recall: {recall_score(y_test["jamming"], y_pred_nn[:, 0])}')
print(f'Jamming - F1 score: {f1_score(y_test["jamming"], y_pred_nn[:, 0])}')
print(f'Replay - Precision: {precision_score(y_test["replay"], y_pred_nn[:, 1])}')
print(f'Replay - Recall: {recall_score(y_test["replay"], y_pred_nn[:, 1])}')
print(f'Replay - F1 score: {f1_score(y_test["replay"], y_pred_nn[:, 1])}\n')

# Train Convolutional Neural Networks model
cnn = Sequential()
cnn.add(Conv1D(filters=32, kernel_size=3, activation='relu', input_shape=(X_train.shape[1], 1)))
cnn.add(MaxPooling1D(pool_size=2))
cnn.add(Flatten())
cnn.add(Dense(100, activation='relu'))
cnn.add(Dense(2, activation='sigmoid'))
cnn.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
cnn.fit(X_train.values.reshape(-1, X_train.shape[1], 1), y_train, epochs=10, batch_size=16, verbose=0)
y_pred_cnn = cnn.predict(X_test.values.reshape(-1, X_test.shape[1], 1))
y_pred_cnn = np.round(y_pred_cnn)
print('Convolutional Neural Networks:')
print(f'Jamming - Precision: {precision_score(y_test["jamming"], y_pred_cnn[:, 0])}')
print(f'Jamming - Recall: {recall_score(y_test["jamming"], y_pred_cnn[:, 0])}')
print(f'Jamming - F1 score: {f1_score(y_test["jamming"], y_pred_cnn[:, 0])}')
print(f'Replay - Precision: {precision_score(y_test["replay"], y_pred_cnn[:, 1])}')
print(f'Replay - Recall: {recall_score(y_test["replay"], y_pred_cnn[:, 1])}')
print(f'Replay - F1 score: {f1_score(y_test["replay"], y_pred_cnn[:, 1])}\n')

# Train Long Short-Term Memory model
lstm = Sequential()
lstm.add(LSTM(units=32, return_sequences=True, input_shape=(X_train.shape[1], 1)))
lstm.add(Dropout(0.2))
lstm.add(LSTM(units=32, return_sequences=True))
lstm.add(Dropout(0.2))
lstm.add(LSTM(units=32))
lstm.add(Dropout(0.2))
lstm.add(Dense(units=2, activation='sigmoid'))
lstm.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
lstm.fit(X_train.values.reshape(-1, X_train.shape[1], 1), y_train, epochs=10, batch_size=16, verbose=0)
y_pred_lstm = lstm.predict(X_test.values.reshape(-1, X_test.shape[1], 1))
y_pred_lstm = np.round(y_pred_lstm)
print('Long Short-Term Memory:')
print(f'Jamming - Precision: {precision_score(y_test["jamming"], y_pred_lstm[:, 0])}')
print(f'Jamming - Recall: {recall_score(y_test["jamming"], y_pred_lstm[:, 0])}')
print(f'Jamming - F1 score: {f1_score(y_test["jamming"], y_pred_lstm[:, 0])}')
print(f'Replay - Precision: {precision_score(y_test["replay"], y_pred_lstm[:, 1])}')
print(f'Replay - Recall: {recall_score(y_test["replay"], y_pred_lstm[:, 1])}')
print(f'Replay - F1 score: {f1_score(y_test["replay"], y_pred_lstm[:, 1])}\n')

# Train Recurrent Neural Networks model
rnn = Sequential()
rnn.add(SimpleRNN(units=32, return_sequences=True, input_shape=(X_train.shape[1], 1)))
rnn.add(Dropout(0.2))
rnn.add(SimpleRNN(units=32, return_sequences=True))
rnn.add(Dropout(0.2))
rnn.add(SimpleRNN(units=32))
rnn.add(Dropout(0.2))
rnn.add(Dense(units=2, activation='sigmoid'))
rnn.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
rnn.fit(X_train.values.reshape(-1, X_train.shape[1], 1), y_train, epochs=10, batch_size=16, verbose=0)
y_pred_rnn = rnn.predict(X_test.values.reshape(-1, X_test.shape[1], 1))
y_pred_rnn = np.round(y_pred_rnn)
print('Recurrent Neural Networks:')
print(f'Jamming - Precision: {precision_score(y_test["jamming"], y_pred_rnn[:, 0])}')
print(f'Jamming - Recall: {recall_score(y_test["jamming"], y_pred_rnn[:, 0])}')
print(f'Jamming - F1 score: {f1_score(y_test["jamming"], y_pred_rnn[:, 0])}')
print(f'Replay - Precision: {precision_score(y_test["replay"], y_pred_rnn[:, 1])}')
print(f'Replay - Recall: {recall_score(y_test["replay"], y_pred_rnn[:, 1])}')
print(f'Replay - F1 score: {f1_score(y_test["replay"], y_pred_rnn[:, 1])}\n')


