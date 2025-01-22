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


X = cp.Constant(np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]]))
cons = cp.sum(X) == 6
print(cons.value())
ans = cp.sum(X, axis=0) <= 15
ans.value()


# Generating all combinations of angles
K=3
lins  = [np.linspace(0, 1.5*np.pi, 2) for i in range(K)]
# lins.append(np.linspace(0, 2*np.pi, d) + np.pi/4)
angles = np.array(np.meshgrid(*lins)).T.reshape(-1, K)
print(angles)


import jax.numpy as jnp

def B(p_i, theta):
  dm_i = r @ jnp.array([jnp.cos(theta), jnp.sin(theta), 0])
  r_i = target - p_i
  r_i_hat = r_i / jnp.linalg.norm(r_i)
  return mu_0 * moment / (4 * jnp.pi * jnp.linalg.norm(r_i) ** 3) * ((3 * jnp.outer(r_i_hat, r_i_hat) - jnp.eye(3)) @ dm_i)

def F(p_i, theta):
  dm_i = r @ jnp.array([jnp.cos(theta), jnp.sin(theta), 0])
  r_i = target - p_i
  r_i_hat = r_i / jnp.linalg.norm(r_i)
  return 3 * mu_0 * moment / (4 * jnp.pi * jnp.linalg.norm(r_i) ** 4) \
    * jnp.dot(
      jnp.outer(dm_i, r_i_hat) + 
      jnp.outer(r_i_hat, dm_i) - 
      ((5 * jnp.outer(r_i_hat, r_i_hat) - jnp.eye(3)) * jnp.dot(dm_i, r_i_hat))
      , mt)
    
    
  

t = np.random.rand(1)[0] * 3.14
def dm(x):
  return r @ jnp.array([-jnp.sin(x), jnp.cos(x), 0.])
from jax import jacfwd, jacrev
f = lambda x: B(p, dm(x), r)
J = jacfwd(F, argnums=1)(p, t)
print(J)
x=0.
print(Jf(p, dm(t)))



coll = magpy.Collection()
j = Br / mu_0 
angles = res.x

xrange = 0.01
zmax = target[2] + xrange
zmin = target[2] - xrange
ymax = target[1] + xrange
ymin = target[1] - xrange
xmax = ymax
xmin = ymin
xrange2 = 0.05

xy
xz

xy
yz

xz
yz

gridxy = np.array([[(x, y, target[2]) for x in np.linspace(xmin, xmax, 50)] for y in np.linspace(ymin, ymax, 50)])
x_xy, y_xy, _ = np.moveaxis(gridxy, 2, 0)

gridxz = np.array([[(x, target[1], z) for x in np.linspace(xmin, xmax, 50)] for z in np.linspace(zmin, zmax, 50)])

# Create an observer grid in the xy-symmetry plane
grid = np.mgrid[0.2-xrange2:0.2+xrange2:50j, 0.2-xrange2:0.2+xrange2:50j, 0.55:0.55:1j].T[0]
X1, Y1, _ = np.moveaxis(grid, 2, 0)

# Create an observer grid in the yz-symmetry plane
grid2 = np.array([[(0.2, y, z) for y in np.linspace(xmin, xmax, 50)] for z in np.linspace(zmin, zmax, 50)])
_, Y2, Z2 = np.moveaxis(grid2, 2, 0)

# Create an observer grid in the yz-symmetry plane
grid3 = np.array([[(x, 0.2, z) for x in np.linspace(xmin, xmax, 50)] for z in np.linspace(zmin, zmax, 50)])
X3, _, Z3 = np.moveaxis(grid3, 2, 0)


for ind, i in enumerate(inds):
  p, r = S[i]
  # magpy.magnet.Cuboid(magnetization=(M,0,0), dimension=(0.02,0.01,0.05), position=(-0.074806,0,0))
  coll.add(magpy.magnet.Cuboid(magnetization=(j, 0, 0), dimension=(l, l, l), position=p, orientation=R.from_matrix(r)).rotate_from_angax(angles[ind], axis=r.dot([0, 0, 1]), degrees=False))

print("Field at target: ", coll.getB(target))

b = coll.getB(grid)
Bx, By, _ = np.moveaxis(b, 2, 0)

b2 = coll.getB(grid2)
_, By2, Bz2 = np.moveaxis(b2, 2, 0)

b3 = coll.getB(grid3)
Bx3, By3, Bz3 = np.moveaxis(b3, 2, 0)

fig, [ax1,ax2] = plt.subplots(2, 1, figsize=(5, 10))
# splt = ax1.streamplot((X1-0.2)*100, (Y1-0.2)*100, Bx, By, color=1000*(norm(b, axis=2)), cmap="autumn_r")
# ax1.plot(0, 0, 'bo')
splt2 = ax2.streamplot((Y2-0.2)*100, (Z2-0.55)*100, By2, Bz2, color=1000*(norm(b2, axis=2)), cmap="spring_r")
ax2.plot(0, 0, 'bo')
splt = ax1.streamplot((X3-0.2)*100, (Z3-0.55)*100, Bx3, Bz3, color=1000*(norm(b3, axis=2)), cmap="autumn_r")
ax1.plot(0, 0, 'bo')

cb = fig.colorbar(splt.lines, ax=ax1, label="|B| (mT)")

fig.colorbar(splt2.lines, ax=ax2, label="|B| (mT)")

ax1.set(
    xlabel="x-position (cm)",
    ylabel="z-position (cm)",
    yticks=[-1, -0.5, 0, 0.5, 1],
    xlim=(-xrange*100, xrange*100),
    ylim=(-xrange*100, xrange*100),
)

ax2.set(
    xlabel="y-position (cm)",
    ylabel="z-position (cm)",
    yticks=[-1, -0.5, 0, 0.5, 1],
    xlim=(-xrange*100, xrange*100),
    ylim=(-xrange*100, xrange*100),
)


# magpy.show(coll, animation=False)
plt.show()