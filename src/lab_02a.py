import torch
from torch import nn

torch.manual_seed(7)
vocab_size = 12
d_model = 4

embedding = nn.Embedding(vocab_size, d_model)   # E: [12, 4]
input_ids = torch.tensor([[2, 5, 2, 9]])       # [B=1, n=4]
x = embedding(input_ids)                       # [1, 4, 4]

print("E:", embedding.weight.shape)
print("input_ids:", input_ids.shape)
print("x:", x.shape)
print("x[0] - jedna sekwencja:", x[0].shape)
print("x[0, 0] to E[2]:", torch.equal(x[0, 0], embedding.weight[2]))
print("powtórzone ID daje ten sam wektor:",
      torch.equal(x[0, 0], x[0, 2]))

# Korzystamy z input_ids, x, nn i d_model z poprzedniego fragmentu.
max_seq_len = 32
position_embedding = nn.Embedding(max_seq_len, d_model)

n = input_ids.shape[1]                         # w przykładzie n = 4
positions = torch.arange(n).unsqueeze(0)      # [[0, 1, 2, 3]]
position_vectors = position_embedding(positions)
x_with_position = x + position_vectors

print("cała tabela pozycji:", position_embedding.weight.shape)  # [32, 4]
print("wybrane pozycje:", positions)
print("P dla tej sekwencji:", position_vectors.shape)           # [1, 4, 4]
print("X:", x.shape)                                             # [1, 4, 4]
print("H = X + P:", x_with_position.shape)                       # [1, 4, 4]

print("przed dodaniem pozycji wektory ID 2 są identyczne:",
      torch.equal(x[0, 0], x[0, 2]))
print("po dodaniu pozycji reprezentacje się różnią:",
      not torch.allclose(x_with_position[0, 0], x_with_position[0, 2]))




h = x_with_position[0]                 # [n=4, d_model=4]
n, d_model = h.shape
d_k, d_v = 2, 3
torch.manual_seed(11)
w_q = torch.randn(d_model, d_k)        # [4, 2]
w_k = torch.randn(d_model, d_k)        # [4, 2]
w_v = torch.randn(d_model, d_v)        # [4, 3]
q = h @ w_q                            # [4, 2]
k = h @ w_k                            # [4, 2]
v = h @ w_v                            # [4, 3]
print("H:", tuple(h.shape))
print("W_Q, W_K, W_V:",
      tuple(w_q.shape), tuple(w_k.shape), tuple(w_v.shape))
print("Q, K, V:",
      tuple(q.shape), tuple(k.shape), tuple(v.shape))
print("QKᵀ:", tuple((q @ k.T).shape))
print("Pierwsza pozycja:")
print("q[0]:", q[0])
print("k[0]:", k[0])
print("v[0]:", v[0])


import math

toy_x = torch.tensor([
    [1.0, 0.0],
    [0.0, 1.0],
    [1.0, 1.0],
])                                            # [n=3, d=2]

q = k = v = toy_x

raw_scores = q @ k.T                         # [3, 3]
scores = raw_scores / math.sqrt(q.shape[-1]) # [3, 3]
weights = torch.softmax(scores, dim=-1)      # [3, 3]
result = weights @ v                         # [3, 2]

torch.set_printoptions(precision=3, sci_mode=False)
print("oceny przed skalowaniem:\n", raw_scores)
print("oceny po skalowaniu:\n", scores)
print("wagi po softmax:\n", weights)
print("suma każdego wiersza wag:", weights.sum(dim=-1))
print("wektory wynikowe:\n", result)



# Kontynuacja poprzedniego przykładu: korzystamy z q, v i scores.
n = q.shape[0]

# True oznacza tutaj pozycję przyszłą, którą chcemy zablokować.
future = torch.triu(
    torch.ones(n, n, dtype=torch.bool),
    diagonal=1,
)

masked_scores = scores.masked_fill(future, float("-inf"))
causal_weights = torch.softmax(masked_scores, dim=-1)
causal_context = causal_weights @ v

print("pozycje zablokowane przez maskę:\n", future)
print("wyniki po nałożeniu maski:\n", masked_scores)
print("wagi causal:\n", causal_weights)
print("wynik attention:\n", causal_context)

assert torch.allclose(causal_weights.sum(dim=-1), torch.ones(n))
assert torch.all(causal_weights[future] == 0)
