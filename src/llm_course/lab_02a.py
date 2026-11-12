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
      
      
      
      
import math
import torch

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

def attention_output(x: torch.Tensor, causal: bool) -> torch.Tensor:
    scores = x @ x.T / math.sqrt(x.shape[-1])

    if causal:
        n = x.shape[0]
        future = torch.triu(
            torch.ones(n, n, dtype=torch.bool, device=x.device),
            diagonal=1,
        )
        scores = scores.masked_fill(future, float("-inf"))

    weights = torch.softmax(scores, dim=-1)
    return weights @ x


a = torch.tensor([[1., 0.], [0., 1.], [1., 1.]])
b = torch.tensor([[1., 0.], [8., 8.], [-7., 2.]])

causal_a = attention_output(a, causal=True)[0]
causal_b = attention_output(b, causal=True)[0]
unmasked_a = attention_output(a, causal=False)[0]
unmasked_b = attention_output(b, causal=False)[0]

print("z maską:", causal_a, causal_b)
print("bez maski:", unmasked_a, unmasked_b)

assert torch.allclose(causal_a, causal_b)
assert not torch.allclose(unmasked_a, unmasked_b)