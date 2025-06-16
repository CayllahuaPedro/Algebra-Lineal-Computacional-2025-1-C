import numpy as np
from template_funciones_2 import calculaLU, resolver_LU, calcula_L

# Crear una matriz de ejemplo
A = np.array([
    [4, 3, 2],
    [1, 5, 6],
    [7, 8, 9]
])

# Probar factorización LU
print('Probando calculaLU:')
LU = calculaLU(A)
print('Matriz LU:')
print(LU)

# Probar resolución de sistema lineal
b = np.array([1, 2, 3])
print('Probando resolver_LU:')
x = resolver_LU(LU, b)
print('Solución del sistema Ax=b:')
print(x)

# Probar matriz laplaciana
print('Probando calcula_L:')
L = calcula_L(A)
print('Matriz Laplaciana:')
print(L)
