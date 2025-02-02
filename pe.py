import matplotlib.pyplot as plt
import pickle
import numpy as np

def pos_enc_matrix(L, d, n=1000):
    """Create positional encoding matrix
    Args:
        L: Input dimension (length)
        d: Output dimension (depth), even only
        n: Constant for the sinusoidal functions
        
    Returns:
        numpy matrix of floats of dimension L x d.
        At element 
            p(k, 2i) = sin(k/n^(2*i/d))
            p(k, 2i+1) = cos(k/n^(2*i/d))
    """
    
    assert d%2 == 0, "output dimension needs to be an even interger"
    
    d2 = d//2
    
    P = np.zeros((L, d))
    k = np.arange(L).reshape(-1, 1)     # L-column vector
    i = np.arange(d2).reshape(1, -1)    # d-row vector

    denom = np.power(n, -i/d2)          # n**(-2*i/2)
    args = k * denom
    P[:, ::2] = np.sin(args)
    P[:, 1::2] = np.cos(args)
    
    return P

# Plot the positional encoding matrix
pos_matrix = pos_enc_matrix(L=2048, d=512)
assert pos_matrix.shape == (2048, 512)
plt.pcolormesh(pos_matrix, cmap='RdBu')
plt.xlabel('Depth')
plt.ylabel('Position')
plt.colorbar()
plt.show()

    