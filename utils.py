import jax.numpy as np
import faiss


poincarre_dist = lambda x,y: np.arccosh(\
    1 + 2*(\
        np.linalg.norm(x-y, ord=1)/((1-np.linalg.norm(x, ord=1))*(1-np.linalg.norm(y, ord=1)))
        )
    )