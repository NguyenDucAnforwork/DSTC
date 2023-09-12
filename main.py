import torch
from transformers import AutoTokenizer, AutoModel
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity


# Đọc tệp văn bản
def read_data(filepath):
    temp = ""
    data = []
    first_time = True
    with open(filepath, 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip()
            
            if line:
                first_time = True
                temp += " " + line

            elif not line and first_time:
                data.append(temp)
                first_time = False

            else:
                temp = ""
    return data

def embedding(data, tokenizer, model):
    embedding_vector = []
    for text in data:
        tokens = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            outputs = model(**tokens)
        sentence_embedding = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
        embedding_vector.append(sentence_embedding)
    return embedding_vector

def cost(y_pred, y_true):
    return sum([np.linalg.norm(pred_val - true_val) for pred_val, true_val in zip(y_pred, y_true)]) / len(y_pred)

def softmax(Z):
    e_Z = np.exp(Z - np.max(Z, axis=1, keepdims=True))
    return e_Z / e_Z.sum(axis=1, keepdims=True)

def softmax2(Z):
    e_Z = np.exp(Z - np.max(Z))
    return e_Z / np.sum(e_Z)

def scale(data):
    data = np.array(data)
    return data * 0.5 + 0.5
# Tải pretrained model PhoBERT và tokenizer
model_name = "vinai/phobert-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# Định nghĩa dữ liệu mẫu và nhãn tương ứng
data = []
labels = []

data = read_data('data/data.txt')
labels = read_data('data/labels.txt')

# Tiền xử lý dữ liệu và biến đổi thành vector embedding
embeddings = []
label_embeddings = []

embeddings = embedding(data, tokenizer, model)
label_embeddings = embedding(labels, tokenizer, model)

# Chuyển danh sách các vector embedding thành mảng numpy
X = np.array(scale(embeddings).T)
y = np.array(scale(label_embeddings).T)

input_dim = 768
output_dim = 768

# Số neuron trong các lớp ẩn và số dữ liệu input
hidden_dim = 100
num_of_point = X.shape[1]

# Khởi tạo trọng số cho các lớp. Ở đây ta dùng 1 hidden layer với activation lần lượt là ReLU và softmax
# loss function là MSE
W1 = 0.01 * np.random.randn(input_dim, hidden_dim)
b1 = np.zeros((hidden_dim, num_of_point))
W2 = 0.01 * np.random.randn(hidden_dim, output_dim)
b2 = np.zeros((output_dim, num_of_point))

# Tốc độ học (learning rate) + số epoch
eta = 1
epochs = 10

# Huấn luyện mô hình
for epoch in range(epochs):
    # Feedforward
    Z1 = np.dot(X.T, W1) + b1.T #  768 25  .  768   100   +    100 25
    A1 = np.maximum(Z1, 0)      #  25 100 
    Z2 = np.dot(A1, W2) + b2.T  #  25 100  .  100 768   +    768 25 
    y_pred = np.array(softmax(Z2)).T        #  25 768

    # Tính gradient theo MSE
    delta2 = y_pred - y     #   768 25  25 768
    dW2 = np.dot(delta2, A1)  #   768 25 . 25 100 
    db2 = np.array(sum([element for element in delta2]))   # 768 1

    delta1 = np.dot(W2, delta2)   # 100 768 . 768 25
    delta1[Z1.T <= 0] = 0  # Gradient của ReLU
    dW1 = np.dot(X, delta1.T)  # 768 25 . 100 25
    db1 = np.array(sum([element for element in delta1]))  # 100 1

    # Cập nhật trọng số
    W1 -= eta * dW1
    b1 -= eta * db1
    W2 -= eta * dW2.T
    b2 -= eta * db2

    # Tính hàm mất mát (MSE)
    loss = cost(y_pred, y)
    print(f"Epoch {epoch+1}/{epochs}, Loss: {loss}")

# Thử trên tập test 
X_test, y_test = read_data('data/X_test.txt'), read_data('data/y_test.txt')
X_test = np.array(scale(embedding(X_test, tokenizer, model)))
y_test = np.array(scale(embedding(y_test, tokenizer, model)))   
print(W1.shape, W2.shape, b1.shape, b2.shape)

for i in range(len(X_test)):

    Z1 = np.dot(X_test[i].T, W1) + (b1.T)[i]  # 768 1   768 100
    A1 = np.maximum(Z1.T, 0)    # 1 100
    Z2 = np.dot(A1, W2) + (b2.T)[i]  # 1 100 . 100 768
    
    y_pred = np.array(softmax2(Z2))  
    y_true = y_test[i].T 

    res = np.dot(y_pred, y_true) / np.linalg.norm(y_pred) / np.linalg.norm(y_true)

    print(f"Độ matchin là {res}%")