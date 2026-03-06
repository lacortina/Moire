#!/usr/bin/env python3
#!/usr/bin/env python3
# read_strict.py
"""
Lectura estricta de:
 - Red.txt       : 3 líneas con 3 floats (vectores de red)
 - Posiciones.txt: líneas "SYMBOL  x  y  z" (sin header de datos)
 - Orbitales.txt : líneas "SYMBOL  orb1 orb2 ..." (orbitales entre el conjunto permitido)

Se ignoran líneas que comiencen por '#' o líneas de encabezado que contengan
palabras como 'simbol', 'simbolo', 'posiciones', 'vectores', 'orbitales'.
Si el formato no coincide se lanza ValueError con mensaje explicativo.
"""

from typing import List, Tuple, Dict, Any
import numpy as np
import re
from scipy.linalg import eigh
from collections import defaultdict
import math

def leer_positions(path: str) -> Tuple[List[str], List[List[float]]]:
    """
    Lee un .txt con filas: <simbolo> <x> <y> <z>
    Devuelve (simbolos, posiciones) donde:
      - simbolos: lista de str
      - posiciones: lista de [x, y, z] (floats)
    Sin comprobaciones ni manejo de errores.
    """
    with open(path, "r") as f:
        lines = [l.strip() for l in f if l.strip()]

    simbolos: List[str] = []
    posiciones: List[List[float]] = []

    for l in lines:
        p = l.split()
        simbolos.append(p[0])
        posiciones.append([float(p[1]), float(p[2]), float(p[3])])

    return simbolos, posiciones

def leer_orbitales(path: str) -> Dict[str, List[str]]:
    """
    Lee un .txt con formato:
    simbolo orb1 orb2 orb3 ...

    Devuelve:
        dict[simbolo] = [orbitales...]
    """
    with open(path, "r") as f:
        lines = [l.strip() for l in f if l.strip() and not l.startswith("#")]

    tens: Dict[str, List[str]] = {}

    for l in lines:
        p = l.split()
        tens[p[0]] = p[1:]

    return tens

def leer_Red(path: str) -> Tuple[List[str], List[List[float]]]:
    """
    Lee un .txt con filas: <simbolo> <x> <y> <z>
    Devuelve (simbolos, posiciones) donde:
      - simbolos: lista de str
      - posiciones: lista de [x, y, z] (floats)
    Sin comprobaciones ni manejo de errores.
    """
    with open(path, "r") as f:
        lines = [l.strip() for l in f if l.strip()]

    posiciones: List[List[float]] = []

    for l in lines:
        p = l.split()
        posiciones.append([float(p[0]), float(p[1]), float(p[2])])

    return posiciones


def construir_supercelda(lattice, positions, n1, n2, n3, frac=True):
    """
    Construye una supercelda centrada en el origen.

    Genera traslaciones desde -n hasta +n en cada dirección.

    Parámetros
    ----------
    lattice : array (3,3)
        Vectores de red como filas
    positions : array (Nat,3)
        Posiciones atómicas
    n1, n2, n3 : int
        Número de replicas hacia cada lado
    frac : bool
        True si posiciones están en coordenadas fraccionarias

    Retorna
    -------
    new_positions : array
        Posiciones cartesianas de la supercelda
    new_lattice : array (3,3)
        Vectores de red totales de la supercelda
    """

    lattice = np.asarray(lattice, dtype=float)
    positions = np.asarray(positions, dtype=float)

    a1, a2, a3 = lattice

    # convertir a cartesianas si están en fraccionarias
    if frac:
        positions_cart = positions @ lattice
    else:
        positions_cart = positions.copy()

    super_positions = []

    for i in range(-n1, n1 + 1):
        for j in range(-n2, n2 + 1):
            for k in range(-n3, n3 + 1):

                shift = i*a1 + j*a2 + k*a3

                for pos in positions_cart:
                    super_positions.append(pos + shift)

    new_positions = np.array(super_positions)

    # nuevo tensor de red total
    new_lattice = np.array([
        (2*n1 + 1) * a1,
        (2*n2 + 1) * a2,
        (2*n3 + 1) * a3
    ])

    return new_positions, new_lattice

def rotar_eje_z(posiciones, theta, grados=False):
    """
    Rota un conjunto de posiciones alrededor del eje Z.

    Parámetros
    ----------
    posiciones : array (N,3)
        Coordenadas cartesianas
    theta : float
        Ángulo de rotación
    grados : bool
        True si theta está en grados

    Retorna
    -------
    array (N,3)
        Posiciones rotadas
    """

    posiciones = np.asarray(posiciones, dtype=float)

    if grados:
        theta = np.deg2rad(theta)

    c = np.cos(theta)
    s = np.sin(theta)

    Rz = np.array([
        [c, -s, 0.0],
        [s,  c, 0.0],
        [0.0, 0.0, 1.0]
    ])

    return posiciones @ Rz.T

def unir_posiciones(pos1, pos2):
    """
    Une dos conjuntos de posiciones (N1,3) y (N2,3).

    Retorna array (N1+N2,3)
    """
    pos1 = np.asarray(pos1, dtype=float)
    pos2 = np.asarray(pos2, dtype=float)

    return np.vstack((pos1, pos2))

import matplotlib.pyplot as plt

def visualizar_xy(posiciones, equal=True, size=1):
    """
    Visualiza la proyección XY de un conjunto de posiciones.

    Parámetros
    ----------
    posiciones : array (N,3)
    equal : bool
        Fuerza aspecto 1:1
    size : int
        Tamaño de los puntos
    """

    posiciones = np.asarray(posiciones)

    x = posiciones[:, 0]
    y = posiciones[:, 1]

    plt.figure()
    plt.scatter(x, y, s=size)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Proyección en plano XY")

    if equal:
        plt.gca().set_aspect("equal", adjustable="box")

    plt.show()

def visualizar_xy_dos(pos1, pos2, 
                      color1="blue", 
                      color2="red", 
                      size=0.1, 
                      equal=True):
    """
    Visualiza dos conjuntos de posiciones en el plano XY
    con colores diferentes.

    Parámetros
    ----------
    pos1, pos2 : array (N,3)
    color1, color2 : str
        Colores matplotlib
    size : float
        Tamaño de los puntos
    equal : bool
        Fuerza aspecto 1:1
    """

    pos1 = np.asarray(pos1)
    pos2 = np.asarray(pos2)

    plt.figure()

    plt.scatter(pos1[:,0], pos1[:,1],
                c=color1, s=size, label="Set 1")

    plt.scatter(pos2[:,0], pos2[:,1],
                c=color2, s=size, label="Set 2")

    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Proyección XY")
    plt.legend()

    if equal:
        plt.gca().set_aspect("equal", adjustable="box")

    plt.show()

def filtrar_celda_unidad(posiciones, L1, L2, tol=1e-12):
    """
    Filtra las posiciones que estén dentro del paralelogramo
    generado por L1 y L2 (desde el origen).

    Parámetros
    ----------
    posiciones : array (N,3) o (N,2)
    L1, L2 : array (3,) o (2,)
    tol : float
        Tolerancia numérica

    Retorna
    -------
    posiciones_filtradas : array
    """

    posiciones = np.asarray(posiciones)
    L1 = np.asarray(L1)
    L2 = np.asarray(L2)

    # trabajar solo en XY
    M = np.array([
        [L1[0], L2[0]],
        [L1[1], L2[1]]
    ])

    M_inv = np.linalg.inv(M)

    # coordenadas XY
    xy = posiciones[:, :2].T   # (2,N)

    # fraccionarias
    f = M_inv @ xy
    f1 = f[0]
    f2 = f[1]

    mask = (
        (f1 >= -tol) & (f1 <= 1 + tol) &
        (f2 >= -tol) & (f2 <= 1 + tol)
    )

    return posiciones[mask]

symbols, posiciones = leer_positions('Posiciones.txt')
posiciones = np.array(posiciones)
Red = leer_Red('Red.txt')

P, A = construir_supercelda(Red, posiciones, 20, 20, 0, False)
#print(P)

PR = rotar_eje_z(P, 3.8902381690076835, True)
#print(PR)

h = 0 #3.35
PR = PR + np.array([0.0, 0.0, h])

PT = unir_posiciones(P, PR)
#visualizar_xy(PT)


#visualizar_xy_dos(P, PR)

##-----------------AHORA SE CREA LA CELDA UNIDAD -----------------
m = 8
n = 9

Red[0]= np.asarray(Red[0], dtype=float)
Red[1]= np.asarray(Red[1], dtype=float)

L1 = m*Red[0]+ n*Red[1]

#rotamos los vectores, 
angle =  np.arccos(0.5* (m**2+n**2+4*m*n)/(m**2+n**2+m*n))
modL = 1.42*abs(n-m)/(2*np.sin(angle/2))
print(modL)
c = np.cos(np.deg2rad(60))
s = np.sin(np.deg2rad(60))

Rz = np.array([
        [c, -s, 0.0],
        [s,  c, 0.0],
        [0.0, 0.0, 1.0]
    ])
print(np.degrees(angle))
L2 = L1@ Rz.T


#ver si coinciden





def plot_xy_con_red(posiciones, L1, L2, punto_size=5):
    """
    Dibuja las posiciones en XY con los vectores de red L1 y L2.
    Marca en rojo las posiciones duplicadas.
    """

    posiciones = np.asarray(posiciones)
    x = posiciones[:, 0]
    y = posiciones[:, 1]

    fig, ax = plt.subplots()

    # redondear para evitar errores numéricos
    coords_rounded = np.round(posiciones[:, :2], decimals=2)

    # encontrar duplicados usando np.unique
    _, idx_unique, counts = np.unique(coords_rounded, axis=0, return_index=True, return_counts=True)
    # los puntos que aparecen más de una vez
    duplicadas_mask = np.isin(np.arange(len(posiciones)), 
                              [i for i, c in zip(idx_unique, counts) if c > 1])

    # dibujar los puntos
    ax.scatter(x[~duplicadas_mask], y[~duplicadas_mask], s=punto_size, c="blue", label="Normal")
    ax.scatter(x[duplicadas_mask], y[duplicadas_mask], s=punto_size*2, c="red", label="Duplicados")

    # dibujar vectores de red desde el origen
    ax.quiver(0, 0, L1[0], L1[1], angles='xy', scale_units='xy', scale=1, color="green", label="L1")
    ax.quiver(0, 0, L2[0], L2[1], angles='xy', scale_units='xy', scale=1, color="orange", label="L2")

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title("Proyección en plano XY con vectores de red y duplicados")
    ax.legend()
    ax.set_aspect("equal", adjustable="box")

    plt.show()
plot_xy_con_red(PT, L1, L2)


print( L1, L2)
""" 

SP = filtrar_celda_unidad(PT, L1 , L2)

#visualizar_xy(PT)

Red[0]= L1
Red[1]= L2 
SP , A= construir_supercelda(Red, PT, 4, 4, 0, False)
plot_xy_con_red(PT, L1, L2) """