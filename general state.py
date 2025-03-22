import numpy as np
from scipy.linalg import sqrtm
from scipy.special import factorial
# from qiskit.quantum_info import Statevector


a1 = np.array([[1], [0]]) # orthonormal basis vector = |0⟩
a2 = np.array([[0], [1]]) # orthonormal basis vector = |1⟩

# hermitian conjugates
b1 = a1.T.conj() # ⟨0|
b2 = a2.T.conj() # ⟨1|

# normalized superposition state
alpha = 1/np.sqrt(2)
beta = 1/np.sqrt(2)
cutoff = 5

pure_state = alpha * a1 + beta * a2
mixed_state = 0.5 * (a1 @ b1) + 0.5 * (a2 @ b2)
entangled_state = (1/np.sqrt(2)) * np.kron(a1, a1) + (1/np.sqrt(2)) * np.kron(a2, a2)
coherent_state = coherent_state = np.array([np.exp(-np.abs(alpha)**2 / 2) * (alpha**n) / np.sqrt(factorial(n)) for n in range(cutoff)])


x = np.array([[1, 0], [0, -1]])
eigen_values, eigen_vector = np.linalg.eig(x)


y = np.array([[0, 0], [0, 1]])
temp = 1.0
beta = 1/ temp
thermal_state = np.exp(-beta * y) / np.trace(np.exp(-beta * y))


n = 3
fock_state = np.zeros((10, 1))
fock_state[n] = 1


# alpha1 = 1.0
# wigner_state = Statevector.from_label('0').evolve(np.exp(-np.abs(alpha)**2 / 2)) formula has issues