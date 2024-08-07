from jax import grad
import jax.numpy as jnp
def dm(x):
  return r @ jnp.array([-jnp.sin(x), jnp.cos(x), 0.])
from jax import jacfwd, jacrev
f = lambda x: B(p, dm(x), r)
J = jacfwd(F, argnums=1)(p, 1.)
print(J)
x=0.
print(Jf(p, dm(1.)))