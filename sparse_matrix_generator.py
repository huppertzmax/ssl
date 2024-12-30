import time
import numpy as np
from scipy.sparse import random, save_npz

n = 100_000  
n_formatted = formatted = f"{n:,}".replace(",", "_")
density = 2. / n

start_time = time.time()
rng = np.random.default_rng(seed=42) 
sparse_matrix = random(n, n, density=density, format='csr', data_rvs=lambda size: rng.integers(0, 2, size=size), random_state=rng)

end_time = time.time()
print(f"Runtime for matrix generation: {end_time - start_time:.6f} seconds")

save_npz(f"data/matrices/sparse_matrix_{n_formatted}/matrix.npz", sparse_matrix)
print(f"Matrix saved under: data/matrices/sparse_matrix_{n_formatted}/matrix.npz")