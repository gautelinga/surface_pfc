import numpy as np
import matplotlib.pyplot as plt


data = np.loadtxt("modes.dat")

ik = data[:, 0:2].astype(int)
a = data[:, 2]
b = data[:, 3]

abnorm = np.sqrt(np.sum(a**2 + b**2))

i_min = ik[:, 0].min()
i_max = ik[:, 0].max()
k_min = ik[:, 1].min()
k_max = ik[:, 1].max()


A = np.zeros((i_max-i_min+1, k_max-k_min+1))
B = np.zeros((i_max-i_min+1, k_max-k_min+1))
for ik, a_ik, b_ik in zip(ik, a, b):
    i = ik[0]
    k = ik[1]
    A[i-i_min, k-k_min] = a_ik/abnorm
    B[i-i_min, k-k_min] = b_ik/abnorm

fig = plt.figure()
plt.imshow(A)
plt.colorbar()

fig = plt.figure()
plt.imshow(B)
plt.colorbar()

plt.show()

Nx = 400
Ny = 400

Lx = 10
Ly = 10

X, Y = np.meshgrid(np.linspace(0., Lx, Nx),
                   np.linspace(0., Ly, Ny),
                   indexing="ij")
Z = np.zeros_like(X)

for i in range(i_min, i_max+1):
    phi_i = 2*np.pi*i*X/Lx
    for k in range(k_min, k_max+1):
        phi_k = 2*np.pi*k*Y/Ly

        a_ik = A[i-i_min, k-k_min]
        b_ik = B[i-i_min, k-k_min]

        Z[:, :] += (a_ik * np.cos(phi_i + phi_k)
                    + b_ik * np.sin(phi_i + phi_k))

print(Z.mean())
print(Z.std())

plt.imshow(Z)
plt.show()
