# %% [markdown]
# ## Magnet optimization

# %%
from scipy.spatial.transform import Rotation as R
import numpy as np
import cvxpy as cp

# %%
Br = 1.43 # (T) Residual flux density for N42
mu_0 = 4 * np.pi * 10**-7 # (H/m) Permeability of free space
l = 5.08e-2 # (m) Length of cube magnet
Volume = l ** 3 # (m^3)
moment = Br * Volume / mu_0 # (A m^2)
j = Br / mu_0 # (A/m)

# %%
Volume = 3.0e-3 ** 3
moment_target = Br * Volume / mu_0

# %%
target = np.array([0.2, 0.2, 0.55]) # target position is at 40 cm above the origin
workspace_length = 0.4 # workspace is a cube of 20 cm side length
mt = np.array([moment_target, 0, 0])

# %%
# return the magnetic field generated by a magnet at position p and orientation r
def generate_random_pose() -> tuple[np.ndarray, np.ndarray]:
    # generate a random pose
    r = R.random()
    p = np.random.rand(3) * workspace_length
    return p, r.as_matrix()

# %%
def B(p_i: np.ndarray, dm_i: np.ndarray):
  r_i = target - p_i
  r_i_hat = r_i / np.linalg.norm(r_i)
  return mu_0 * moment / (4 * np.pi * np.linalg.norm(r_i) ** 3) * ((3 * np.outer(r_i_hat, r_i_hat) - np.eye(3)) @ dm_i)

def F(p_i: np.ndarray, dm_i: np.ndarray):
  r_i = target - p_i
  r_i_hat = r_i / np.linalg.norm(r_i)
  return 3 * mu_0 * moment / (4 * np.pi * np.linalg.norm(r_i) ** 4) \
    * np.dot(
      np.outer(dm_i, r_i_hat) + 
      np.outer(r_i_hat, dm_i) - 
      ((5 * np.outer(r_i_hat, r_i_hat) - np.eye(3)) * np.dot(dm_i, r_i_hat))
      , mt)

def Jb(p_i: np.ndarray, dm_i: np.ndarray):
  r_i = target - p_i
  r_i_hat = r_i / np.linalg.norm(r_i)
  return mu_0 * moment / (4 * np.pi * np.linalg.norm(r_i) ** 3) * ((3 * np.outer(r_i_hat, r_i_hat) - np.eye(3)) @ dm_i)

def Jf(p_i: np.ndarray, dm_i: np.ndarray):
  r_i = target - p_i
  r_i_hat = r_i / np.linalg.norm(r_i)
  return 3 * mu_0 * moment / (4 * np.pi * np.linalg.norm(r_i) ** 4) \
    * np.dot(
      np.outer(dm_i, r_i_hat) + 
      np.outer(r_i_hat, dm_i) - 
      ((5 * np.outer(r_i_hat, r_i_hat) - np.eye(3)) * np.dot(dm_i, r_i_hat))
      , mt)

# %%
m = 50 # Number of random poses
K = 8 # Selection budget
d = 2 # Number of divisions for angles
n = d ** K

# %%
# Generating all combinations of angles
lins  = [np.linspace(0, 1.5*np.pi, d) for i in range(K)]
# lins.append(np.linspace(0, 2*np.pi, d) + np.pi/4)
angles = np.array(np.meshgrid(*lins)).T.reshape(-1, K)

# %%
# S is an array of tuples, each tuple contains a position and a rotation matrix
S = [generate_random_pose() for i in range(m)]

# %%
def calculate_max():
  global Bmax, Fmax
  Bs = []
  Fs = []
  for p, r in S:
    m_i = np.array([0, 0, moment]) # all magnets having north pole facing upwards
    Bs.append(np.linalg.norm(B(p, m_i)))
    Fs.append(np.linalg.norm(F(p, m_i)))

  Bmax = np.partition(Bs, -K)[-K:].sum() # Sum of the norms of K highest fields
  Fmax = np.partition(Fs, -K)[-K:].sum() # Sum of the norms of K highest forces

# %%
calculate_max()

# %%
# Create subsets of overlapping magnets
def overlapping(S):
  overlapping = []
  for i in range(len(S)):
    for j in range(i+1, len(S)):
      if np.linalg.norm(S[i][0] - S[j][0]) < 1.5 * l:
        overlapping.append([i, j])
  return overlapping


# %%
# Initizaling A
def init_A():
  global A
  A = np.zeros((n, K, m, 6, 6))
  calculate_max()

  for t, theta in enumerate(angles):
    for i in range(K):
      for j, (p, r) in enumerate(S):
        dmagnetization = r.dot([- np.sin(theta[i]), np.cos(theta[i]), 0])
        J = np.concatenate([Jb(p, dmagnetization)/Bmax, Jf(p, dmagnetization)/Fmax]) 
        A[t, i, j, :, :] = np.outer(J, J)

# %%
# # Initizaling f
# f = np.zeros((n, K, m, 6, 6))

# for t, theta in enumerate(angles):
#   for i in range(K):
#     for j, (p, r) in enumerate(S):
#       magnetization = r.dot([np.cos(theta[i]), np.sin(theta[i]), 0])
#       fj = np.concatenate([B(p, magnetization)/Bmax, F(p, magnetization)/Fmax])
#       f[t, i, j, :, :] = np.outer(fj, fj)

# %%
init_A()

# %%
def A_operator(X, t):
  return cp.sum([X[i][j] * A[t, i, j] for i in range(K) for j in range(m)])

def f_operator(X):
  return cp.sum([X[i][j] * f[t, i, j] for t in range(n) for i in range(K) for j in range(m)])

# %% [markdown]
# ### CVXPY Setup

# %%
def init_problem():
  global X, t, prob
  X = cp.Variable(shape=(K, m))
  t = cp.Variable(1)

  alpha = 0.1
  # obj = cp.Maximize(t + alpha * cp.atoms.lambda_min(f_operator(X)))
  obj = cp.Maximize(t)
  cons1 = X >= 0.0
  cons2 = X <= 1.0
  cons4 = cp.sum(X) == K # sum of all elements is K
  cons5 = cp.sum(X, axis=1) == 1.0 # sum of each row is 1
  cons6 = cp.sum(X, axis=0) <= 1.0 # sum of each col is le 1
  cons7 = t >= 0.0
  constraints = [cons1, cons2, cons5, cons6]
  for i in range(n):
    constraints.append(t <= cp.atoms.lambda_min(A_operator(X, i)))
  # for o in overlapping(S):
  #   constraints.append(cp.sum([X[i][j] for i in range(K) for j in o]) <= 0.8)
  prob = cp.Problem(obj, constraints)

# %%
init_problem()

# %%
def solve_problem():
  global X, t, prob
  prob.solve(solver=cp.CLARABEL)
  # tol = 1.0e-5
  # prob.solve(verbose=False, solver=cp.CLARABEL, tol_gap_abs=tol, tol_gap_rel=tol, tol_feas=tol)

# %% [markdown]
# ## Rounding

# %%
def top_k(soln, k):
  result = cp.sum(soln, axis=0)
  return np.argsort(result.value)[-k:]


# %%
def top_k_no_overlap(soln, k):
  result = cp.sum(soln, axis=0)
  sorted = np.argsort(result.value)
  # greedily pick the top k magnets that do not overlap
  selected = []
  selected.append(sorted[-1])
  #loop bakwards
  for i in range(2, len(sorted) + 1):
    if len(selected) == k:
      break
    passes = True
    for j in selected:
      if np.linalg.norm(S[sorted[-i]][0] - S[j][0]) < 1.5 * l:
        passes = False
        break
    if passes:
      selected.append(sorted[-i])
  return selected


# %% [markdown]
# ## Saving and Loading data

# %%
import pickle, random


def calculate_singular_values_rounded(inds):
  singular_values = []
  for theta in angles:
    J = np.zeros((K, 6))
    for i, ind in enumerate(inds):
      p, r = S[ind]
      dmagnetization = r.dot([-np.sin(theta[i]), np.cos(theta[i]), 0])
      Ji = np.concatenate([Jb(p, dmagnetization)/Bmax, Jf(p, dmagnetization)/Fmax])
      J[i] = Ji
    s = np.linalg.svd(J, compute_uv=False)
    singular_values.append(min(s))
  return singular_values

def calculate_singular_values_relaxed():
  singular_values = []

  for i in range(n):
    singular_values.append(cp.atoms.lambda_min(A_operator(X, i)).value)
  return singular_values



# %% [markdown]
# ## Greedy

# %%
g_angles = np.linspace(0, 1.5*np.pi, d)


# %%
def greedy_cost(j):
  lambda_mins = []
  for ang in g_angles:
    p, r = S[j]
    dm = r.dot([- np.sin(ang), np.cos(ang), 0])
    J = np.concatenate([Jb(p, dm)/Bmax, Jf(p, dm)/Fmax])
    lambda_mins.append(np.linalg.svd(np.outer(J, J), compute_uv=False)[0])
  return min(lambda_mins)
  

# %%
def get_greedy_inds():
  all_cost = np.array([greedy_cost(j) for j in range(m)])
  return np.argsort(all_cost)[-K:]
  

# %%
# singular_values = calculate_singular_values_rounded(inds)
singular_values_greedy = calculate_singular_values_rounded(get_greedy_inds())

# %%
num_trials = 1

# %%
import pandas as pd

# %%
df = pd.DataFrame(columns=["Relaxed", "Greedy", "OMASTAR"])

# %%
for i in range(num_trials):
  print("Trial", i)
  S = [generate_random_pose() for i in range(m)]
  init_A()
  init_problem()
  solve_problem()
  # print(t.value)
  inds = top_k(X, K)
  sv_rounded = calculate_singular_values_rounded(inds)
  sv_greedy = calculate_singular_values_rounded(get_greedy_inds())
  # sv_bruteforce = calculate_singular_values_rounded(brutefroce_inds)
 
  # sv_random = calculate_singular_values_rounded(np.random.randint(0, m, K))
  df.loc[i, "OMASTAR"] = min(sv_rounded)
  df.loc[i, "Relaxed"] = np.sqrt(min(calculate_singular_values_relaxed()))
  df.loc[i, "Greedy"] = min(sv_greedy)

pickle.dump(df, open("greedy_100_11.pkl", "wb"))