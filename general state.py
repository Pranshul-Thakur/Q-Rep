import numpy as np

a1 = np.array([[1], [0]]) # orthonormal basis vector = |0⟩
a2 = np.array([[0], [1]]) # orthonormal basis vector = |1⟩

# hermitian conjugates
b1 = a1.T.conj() # ⟨0|
b2 = a2.T.conj() # ⟨1|

# normalized superposition state
alpha = 1/np.sqrt(2)
beta = 1/np.sqrt(2)

state = alpha * a1 + beta * a2
inner_product = b1 @ state # inner dot product

print(state)
print(inner_product)