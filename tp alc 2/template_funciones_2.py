# Matriz A de ejemplo
#A_ejemplo = np.array([
#    [0, 1, 1, 1, 0, 0, 0, 0],
#    [1, 0, 1, 1, 0, 0, 0, 0],
#    [1, 1, 0, 1, 0, 1, 0, 0],
#    [1, 1, 1, 0, 1, 0, 0, 0],
#    [0, 0, 0, 1, 0, 1, 1, 1],
#    [0, 0, 1, 0, 1, 0, 1, 1],
#    [0, 0, 0, 0, 1, 1, 0, 1],
#    [0, 0, 0, 0, 1, 1, 1, 0]
#])

import numpy as np
import scipy.linalg

def calculaLU(A):
    # matriz es una matriz de NxN
    # Retorna la factorización LU a través de una lista con dos matrices L y U de NxN.
    # Completar! Have fun
    m=A.shape[0]#fila
    n=A.shape[1]
    Ac = A.copy()#U
    #L=np.eye(n)
    if m!=n:
        print('Matriz no cuadrada')
        return
    
    ## desde aqui -- CODIGO A COMPLETAR
    print(n)
    for j in range(n):
        for i in range (j+1,n):
            mult=Ac[i,j]/Ac[j,j] #escalonas/mult es el factor tipo f2-multf1
            Ac[i,j:] = Ac[i,j:]-mult*Ac[j,j:]#resta de finlas// j: dessde j hasta el final
            Ac[i,j]=mult
         #   Ac[i,:]==Ac[i,:]-L[i,j]*Ac[j,:] 
           # cant_op= cant_op+2
          #  return L, Ac, cant_op
                 
    #L = np.tril(Ac,-1) + np.eye(A.shape[0]) #np.eye es la matriz con 1 en diagonal ()
    #U = np.triu(Ac) #CAPTA LA DIAGONAL INFERIOE
         
            
    return Ac

def resolver_LU(LU, b):
    """
    Resuelve el sistema lineal Ax = b usando la descomposición LU de A.

    Args:
        LU: Matriz de la descomposición LU de A.
        b: Vector o matriz del lado derecho del sistema.

    Returns:
        La solución x del sistema Ax = b.
    """
    L = np.tril(LU,-1) + np.eye(LU.shape[0])
    U = np.triu(LU)
    y = scipy.linalg.solve_triangular(L, b, lower=True)  # Resolvemos Ly = b
    x = scipy.linalg.solve_triangular(U, y)  # Resolvemos Ux = y
    return x

def calcula_L(A):
    # La función recibe la matriz de adyacencia A y calcula la matriz laplaciana
    # Have fun!!
    diagonal=np.sum(A,axis=1)
    K= np.diag(diagonal)
    L=K-A
    return L

def calcula_R(A):
    # R=A-P
    # La funcion recibe la matriz de adyacencia A y calcula la matriz de modularidad
    # Have fun!!
    P=np.zeros(A.shape)
    diagonal=np.sum(A,axis=1)
    K= np.diag(diagonal)
   # print(K)
    E= np.sum(A) / 2
    for i in range(A.shape[0]) :
      for j in range(A.shape[1]):
          P[i,j]= (K[i,i]* K[j,j])/(2*E)
    print(E)
    print(P)

    R=A-P
    return R


def calcula_lambda(L,v):
    s=np.sign(v)
    # Recibe L y v y retorna el corte asociado
    # Have fun!
    #lamdom = 1/4st*l*s
    lambdon = 1/4 * s.transpose() @ L @ s
    return lambdon


def calcula_Q(R,v):
    s=np.sign(v)
    # La funcion recibe R y s y retorna la modularidad (a menos de un factor 2E)
    Q= s.transpose()*R *s
    return Q

def metpot1(A,tol=1e-8,maxrep=np.inf):
   # Recibe una matriz A y calcula su autovalor de mayor módulo, con un error relativo menor a tol y-o haciendo como mucho maxrep repeticiones
   v = np.random.uniform(-1, 1, A.shape[0]) # Generamos un vector de partida aleatorio, entre -1 y 1
   v = v / np.linalg.norm(v) # Lo normalizamos
   v1 = A @ v # Aplicamos la matriz una vez
   v1 = v1 / np.linalg.norm(v1) # normalizamos
   l = v.T @ A @ v # Calculamos el autovector estimado
   l1 = v1.T @ A @ v1 # Y el estimado en el siguiente paso
   nrep = 0 # Contador
   while np.abs(l1-l)/np.abs(l) > tol and nrep < maxrep: # Si estamos por debajo de la tolerancia buscada
      v = v1 # actualizamos v y repetimos
      l = l1
      v1 = A @ v # Calculo nuevo v1
      v1 = v1 / np.linalg.norm(v1) # Normalizo
      l1 = v1.T @ A @ v1 # Calculo autovector
      nrep += 1 # Un pasito mas
   if not nrep < maxrep:
      print('MaxRep alcanzado')
   l = v1.T @ A @ v1 # Calculamos el autovalor
   return v1,l,nrep<maxrep

def deflaciona(A,tol=1e-8,maxrep=np.inf):
    # Recibe la matriz A, una tolerancia para el método de la potencia, y un número máximo de repeticiones
    v1,l1,_ = metpot1(A,tol,maxrep) # Buscamos primer autovector con método de la potencia
    v1 = v1 / np.linalg.norm(v1,2) # Normalizamos
    deflA = A - l1 * np.outer(v1,v1) # Deflacionamos
    return deflA,v1,l1

def metpot2(A,v1,l1,tol=1e-8,maxrep=np.inf):
   # La funcion aplica el metodo de la potencia para buscar el segundo autovalor de A, suponiendo que sus autovectores son ortogonales
   # v1 y l1 son los primeors autovectores y autovalores de A}
   # Have fun!
   
   return metpot1(deflA,tol,maxrep)


def metpotI(A,mu,tol=1e-8,maxrep=np.inf):
    # Retorna el primer autovalor de la inversa de A + mu * I, junto a su autovector 
    n = A.shape[0]
    A= A+mu*np.eye(n)
    LU= calculaLU(A)
    A_inv= resolver_LU(LU, np.eye(n))
    v_min, sigma, _ = metpot1(A_inv,tol=tol,maxrep=maxrep)
    l_min=1/sigma
    return v_min, l_min

def metpotI2(A, mu, tol=1e-8, maxrep=np.inf):
    """
    Recibe la matriz A, y un valor mu y retorna el segundo autovalor y autovector de la matriz A, 
    suponiendo que sus autovalores son positivos excepto por el menor que es igual a 0.
    Retorna el segundo mas chico autovector y su autovalor.
    
    Args:
        A (numpy.ndarray): Matriz simétrica.
        mu (float): Coeficiente de shifting.
        tol (float): Tolerancia para la convergencia.
        maxrep (int): Máximo número de iteraciones.
    
    Returns:
        tuple: (autovector, autovalor) donde autovalor es el segundo autovalor más chico de A
        y autovector es el autovector asociado.
    """
    n = A.shape[0]
    X = A + mu * np.eye(n)  # Calculamos la matriz A shifteada en mu
    iX = resolver_LU(calculaLU(X), np.eye(n))  # La invertimos
    defliX, _, _ = deflaciona(iX, tol, maxrep)  # La deflacionamos
    v, l, converged = metpot1(defliX, tol=tol, maxrep=maxrep)  # Buscamos su segundo autovector
    l = 1/l  # Reobtenemos el autovalor correcto
    l -= mu
    return v, l


def laplaciano_iterativo(A,niveles,nombres_s=None):
    # Recibe una matriz A, una cantidad de niveles sobre los que hacer cortes, y los nombres de los nodos
    # Retorna una lista con conjuntos de nodos representando las comunidades.
    # La función debe, recursivamente, ir realizando cortes y reduciendo en 1 el número de niveles hasta llegar a 0 y retornar.
    if nombres_s is None: # Si no se proveyeron nombres, los asignamos poniendo del 0 al N-1
        nombres_s = range(A.shape[0])
    if A.shape[0] == 1 or niveles == 0: # Si llegamos al último paso, retornamos los nombres en una lista
        return([nombres_s])
    else: # Sino:
        L = calcula_L(A) # Recalculamos el L
        
        # Check if L is a valid matrix for eigenvector calculation
        if np.isnan(L).any() or np.isinf(L).any():
            # If L contains NaN or inf values, don't partition further
            return([nombres_s])
            
        try:
            v,l = metpotI2(L,1) # Encontramos el segundo autovector de L
            
            # Check if the eigenvector contains NaN or inf values
            if np.isnan(v).any() or np.isinf(v).any():
                return([nombres_s])
                
            # Recortamos A en dos partes, la que está asociada a el signo positivo de v y la que está asociada al negativo
            s = np.sign(v)
            
            # Check if all signs are the same (can't partition)
            if np.all(s >= 0) or np.all(s <= 0):
                return([nombres_s])
                
            idx_pos = np.where(s > 0)[0]
            idx_neg = np.where(s < 0)[0]
            
            # Check if we have a valid partition
            if len(idx_pos) == 0 or len(idx_neg) == 0:
                return([nombres_s])
                
            Ap = A[np.ix_(idx_pos,idx_pos)] # Asociado al signo positivo
            Am = A[np.ix_(idx_neg,idx_neg)] # Asociado al signo negativo
            
            return(
                    laplaciano_iterativo(Ap,niveles-1,
                                         nombres_s=[ni for ni,vi in zip(nombres_s,v) if vi>0]) +
                    laplaciano_iterativo(Am,niveles-1,
                                         nombres_s=[ni for ni,vi in zip(nombres_s,v) if vi<0])
                    )
        except Exception as e:
            # If any error occurs during the calculation, don't partition further
            print(f"Error in laplaciano_iterativo: {e}")
            return([nombres_s])

def modularidad_iterativo(A=None,R=None,nombres_s=None):
    # Recibe una matriz A, una matriz R de modularidad, y los nombres de los nodos
    # Retorna una lista con conjuntos de nodos representando las comunidades.

    if A is None and R is None:
        print('Dame una matriz')
        return(np.nan)
    if R is None:
        R = calcula_R(A)
    if nombres_s is None:
        nombres_s = range(R.shape[0])
    # Acá empieza lo bueno
    if R.shape[0] == 1: # Si llegamos al último nivel
        return([nombres_s])
    else:
        v,l,_ = metpot1(R,1e-8) # Primer autovector y autovalor de R
        # Modularidad Actual:
        Q0 = np.sum(R[v>0,:][:,v>0]) + np.sum(R[v<0,:][:,v<0])
        if Q0<=0 or all(v>0) or all(v<0): # Si la modularidad actual es menor a cero, o no se propone una partición, terminamos
            return([nombres_s])
        else:
            ## Hacemos como con L, pero usando directamente R para poder mantener siempre la misma matriz de modularidad
            s= np.sign(v)
            idx_pos = np.where(s > 0)[0]
            idx_neg = np.where(s < 0)[0]
            Rp = R[np.ix_(idx_pos,idx_pos)] # Parte de R asociada a los valores positivos de v
            Rm = R[np.ix_(idx_neg,idx_neg)] # Parte asociada a los valores negativos de v
            vp,lp,_ = metpot1(Rp,1e-8)  # autovector principal de Rp
            vm,lm,_ = metpot1(Rm,1e-8) # autovector principal de Rm
        
            # Calculamos el cambio en Q que se produciría al hacer esta partición
            Q1 = 0
            if not all(vp>0) or all(vp<0):
               Q1 = np.sum(Rp[vp>0,:][:,vp>0]) + np.sum(Rp[vp<0,:][:,vp<0])
            if not all(vm>0) or all(vm<0):
                Q1 += np.sum(Rm[vm>0,:][:,vm>0]) + np.sum(Rm[vm<0,:][:,vm<0])
            if Q0 >= Q1: # Si al partir obtuvimos un Q menor, devolvemos la última partición que hicimos
                return([[ni for ni,vi in zip(nombres_s,v) if vi>0],[ni for ni,vi in zip(nombres_s,v) if vi<0]])
            else:
                # Sino, repetimos para los subniveles
                return(modularidad_iterativo(A=Rp,nombres_s=[ni for ni,vi in zip(nombres_s,v) if vi>0]) + modularidad_iterativo(A=Rm,nombres_s=[ni for ni,vi in zip(nombres_s,v) if vi<0]))
