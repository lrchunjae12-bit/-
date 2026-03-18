# -mport numpy as np

def sigmoid(x):
  return 1 / (1 + np.exp(-x))
# y값에 출력에 대한 함수

def identity_function(x):
  return x

#1층 - 첫 번째 원소에서 2층  첫 번째 원소로 가는 코드

X = np.array([1.0, 0.5]) # 1 x 2
W1 = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.5]]) # 2 x 3
B1 = np.array([0.1,0.2,0.3])

A1 = np.dot(X,W1) + B1
Z1 = sigmoid(A1)

print(A1)
print(Z1)


X = np.array([1.0, 0.5]) # 1 x 2
W1 = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.5]]) # 2 x 3
B1 = np.array([0.1,0.2,0.3])

A1 = np.dot(X,W1) + B1
Z1 = sigmoid(A1)

W2 = np.array([[0.1, 0.4], [0.2, 0.5],[0.3, 0.6]]) # 3 x 2
B2 = np.array([0.1, 0.2]) # 1 x2

A2 = np.dot(Z1, W2) + B2 #[1x3] dot [3x2]
Z2 = sigmoid(A2) # [1 x 2]

W3 = np.array([[0.1, 0.3], [0.2, 0.3]])
B3 = np.array([0.1 , 0.2])

A3 = np.dot(Z2, W3) + B3 # Y = A3
Y = identity_function(A3)

print(Y)
