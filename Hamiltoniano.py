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

def leer_delta_y_Onsite(path: str) -> Tuple[float, Dict[str, float]]:
    """
    Lee el fichero, ignora líneas que empiezan por '#'.
    Devuelve (Delta, Onsite) donde Onsite['E_s_S'] = valor (float).
    """
    Delta = 0.0
    Onsite: Dict[str, float] = {}
    with open(path, "r") as f:
        for l in f:
            s = l.strip()
            if not s or s.startswith("#"):
                continue
            p = s.split()
            if p[0] == "Delta":
                Delta = float(p[1])
            elif p[0].startswith("E"):
                Onsite[p[0]] = float(p[1])
    return Delta, Onsite

# ------------------ SK (V) ------------------
def leer_SK(path: str) -> Dict[Tuple[str,str], Dict[str, float]]:
    """
    Lee un fichero y devuelve SK con claves (A,B) -> dict{ 'V...': valor }.
    Sólo procesa líneas que empiezan por 'V' (ignora comentarios '#').
    Formato esperado por línea: Vorbital1orbital2resto_A_B  valor
    Ej: Vsds_S0_Mo 2.405
    """
    SK: Dict[Tuple[str,str], Dict[str, float]] = {}
    with open(path, "r") as f:
        for l in f:
            s = l.strip()
            if not s or s.startswith("#"):
                continue
            p = s.split()
            if not p[0].startswith("V"):
                continue
            # separar etiqueta en vname, A, B
            vtag, A, B = p[0].rsplit("_", 2)
            val = float(p[1])
            key = (A, B)
            if key not in SK:
                SK[key] = {}
            SK[key][vtag] = val
    return SK

#--------------Vectores de la red
def generar_vecinos(Red, extension=0):
    """
    Red: iterable con 3 vectores base cartesianos [R1, R2, R3]
    extension:
        0 -> n = [-1,0,1]
        1 -> n = [-2,-1,0,1,2]

    Devuelve lista de vectores cartesianos n1*R1+n2*R2+n3*R3
    """
    R1, R2, R3 = Red

    if extension == 0:
        vals = (-1, 0, 1)
    else:
        vals = (-2, -1, 0, 1, 2)

    vecinos = []

    for n1 in vals:
        for n2 in vals:
            for n3 in vals:
                v = [
                    n1*R1[0] + n2*R2[0] + n3*R3[0],
                    n1*R1[1] + n2*R2[1] + n3*R3[1],
                    n1*R1[2] + n2*R2[2] + n3*R3[2],
                ]
                vecinos.append(v)

    return vecinos

def reciprocal_lattice(cell):
    """
    cell: (3,3) array, filas = vectores de red directa a1,a2,a3
    devuelve: (3,3) array, filas = vectores recíprocos b1,b2,b3
    """
    a1, a2, a3 = cell

    V = np.dot(a1, np.cross(a2, a3))

    if abs(V) < 1e-12:
        raise RuntimeError("Volumen de celda nulo")

    b1 = 2*np.pi * np.cross(a2, a3) / V
    b2 = 2*np.pi * np.cross(a3, a1) / V
    b3 = 2*np.pi * np.cross(a1, a2) / V

    return np.vstack([b1, b2, b3])

## -----------------------------------------------
##FUNCION PRINCIPAL SLATER KOSTER
## -----------------------------------------------
#FUNCIONES DE AYUDA PARA SACAR LOS COEFICIENTES SLATERE-KOSTER
def funcionVpps(x, Vpp_sigma, Vpp_pi):
        b=x**2*Vpp_sigma+(1-x**2)*Vpp_pi
        return b
def funcionVppp(x, b, Vpp_sigma, Vpp_pi):
        c=x*b*(Vpp_sigma-Vpp_pi)
        return c

#funcion slater koster
def funcionslaterkoster(vector, orbital1, orbital2, Vss, Vsp, Vps, Vsds, Vdss, Vpp_sigma, Vpp_pi, Vpds, Vdps, Vpdp, Vdpp, Vdds, Vddp, Vddd):
    #cosenos directores
    r=np.linalg.norm(vector)
    l=vector[0]/r
    m=vector[1]/r
    n=vector[2]/r
    if orbital1 == 's':
        if orbital2 == 's':
            return Vss
        elif orbital2 == 'px':
            return Vsp*l
        elif orbital2 == 'py':
            return Vsp*m
        elif orbital2 == 'pz':
            return Vsp*n
#añadimos los d (dxy, dxz, dyz, dx2_y2, d3z2_r29)
        elif orbital2 == 'dxy':
            return np.sqrt(3) * l * m * Vsds
        elif orbital2 == 'dxz':
            return np.sqrt(3) * l * n * Vsds
        elif orbital2 == 'dyz':
            return np.sqrt(3) * n * m * Vsds
        elif orbital2 == 'dx2_y2':
            return np.sqrt(3) / 2 * (l**2 - m**2) * Vsds
        elif orbital2 == 'dr':
            return (n**2 - (l**2 + m**2) / 2 )* Vsds
    
    elif orbital1 == 'px':
        if orbital2 == 's':
            return Vps*(-l)
        elif orbital2 == 'px':
            return funcionVpps(l, Vpp_sigma, Vpp_pi)
        elif orbital2 == 'py':
            return funcionVppp(l, m, Vpp_sigma, Vpp_pi)
        elif orbital2 == 'pz':
            return funcionVppp(l, n, Vpp_sigma, Vpp_pi)
        #añadimos los d (dxy, dxz, dyz, dx2_y2, d3z2_r29)
        elif orbital2 == 'dxy':
            return np.sqrt(3) * l**2 * m * Vpds + m * (1 - 2 * l**2) * Vpdp
        elif orbital2 == 'dxz':
            return np.sqrt(3) * l**2 * n * Vpds + n * (1 - 2 * l**2) * Vpdp
        elif orbital2 == 'dyz':
            return np.sqrt(3) * l * n*m * Vpds -2* n * m*l * Vpdp
        elif orbital2 == 'dx2_y2':
            return np.sqrt(3) / 2 * l * (l**2 - m**2) * Vpds + l * (1 - l**2 + m**2) * Vpdp
        elif orbital2 == 'dr':
            return l * (n**2 - (l**2 + m**2) / 2) * Vpds - np.sqrt(3) * l * n**2 * Vpdp

    elif orbital1 == 'py':
        if orbital2 == 's':
            return Vps*(-m)
        elif orbital2 == 'px':
            return funcionVppp(l, m, Vpp_sigma, Vpp_pi)
        elif orbital2 == 'py':
            return funcionVpps(m, Vpp_sigma, Vpp_pi)
        elif orbital2 == 'pz':
            return funcionVppp(m, n, Vpp_sigma, Vpp_pi)
         #añadimos los d (dxy, dxz, dyz, dx2_y2, d3z2_r29)
        elif orbital2 == 'dxy':
            return np.sqrt(3) * m**2 * l * Vpds + l * (1 - 2 * m**2) * Vpdp
        elif orbital2 == 'dxz':
            return np.sqrt(3) * l * n*m * Vpds -2* n * m*l * Vpdp
        elif orbital2 == 'dyz':
            return np.sqrt(3) * m**2 * n * Vpds + n * (1 - 2 * m**2) * Vpdp
        elif orbital2 == 'dx2_y2':
            return np.sqrt(3) / 2 * m * (l**2 - m**2) * Vpds - m * (1 + l**2 - m**2) * Vpdp
        elif orbital2 == 'dr':
            return m * (n**2 - (l**2 + m**2) / 2) * Vpds - np.sqrt(3) * m * n**2 * Vpdp

    elif orbital1 == 'pz':
        if orbital2 == 's':
            return Vps*(-n)
        elif orbital2 == 'px':
            return funcionVppp(n, l, Vpp_sigma, Vpp_pi)
        elif orbital2 == 'py':
            return funcionVppp(n, m, Vpp_sigma, Vpp_pi)
        elif orbital2 == 'pz':
            return funcionVpps(n, Vpp_sigma, Vpp_pi)
         #añadimos los d (dxy, dxz, dyz, dx2_y2, d3z2_r29)
        elif orbital2 == 'dxy':
            return np.sqrt(3) * l * n*m * Vpds -2* n * m*l * Vpdp
        elif orbital2 == 'dxz':
            return np.sqrt(3) * n**2 * l * Vpds + l * (1 - 2 * n**2) * Vpdp           
        elif orbital2 == 'dyz':
            return np.sqrt(3) * n**2 * m * Vpds + m * (1 - 2 * n**2) * Vpdp
        elif orbital2 == 'dx2_y2':
            return np.sqrt(3) / 2 * n * (l**2 - m**2) * Vpds - n * ( l**2 - m**2) * Vpdp
        elif orbital2 == 'dr':
            return n * (n**2 - (l**2 + m**2) / 2) * Vpds + np.sqrt(3) * n * (l**2+m**2) * Vpdp

    elif orbital1 == 'dxy':
        if orbital2 == 's':
            return np.sqrt(3) * l * m * Vdss
        elif orbital2 == 'px':
            return np.sqrt(3) * l**2 * (-m) * Vdps + (-m) * (1 - 2 * l**2) * Vdpp
        elif orbital2 == 'py':
            return np.sqrt(3) * m**2 * (-l) * Vdps + (-l) * (1 - 2 * m**2) * Vdpp
        elif orbital2 == 'pz':
            return np.sqrt(3) * (-l) * n*m * Vdps +2* n * m*l * Vdpp
         #añadimos los d (dxy, dxz, dyz, dx2_y2, d3z2_r29)
        elif orbital2 == 'dxy':
            return 3 * l**2 * m**2 * Vdds + (l**2 + m**2 - 4 * l**2 * m**2) * Vddp+(n**2+l**2*m**2)*Vddd
        elif orbital2 == 'dxz':
            return 3 * l**2 * m * n * Vdds + m * n * (l**2 - 1) * Vddd  + m*n*(1-4*l**2)*Vddp       
        elif orbital2 == 'dyz':
            return 3 * l * n * m**2 * Vdds + l * n * (1 - 4 * m**2) * Vddp+ l*n*(m**2-1)*Vddd
        elif orbital2 == 'dx2_y2':
            return (3 / 2) * l * m * (l**2 - m**2) * Vdds + 2 * l * m * (m**2 - l**2) * Vddp + (l * m * (l**2 - m**2) / 2) * Vddd
        elif orbital2 == 'dr':
            return np.sqrt(3) * (l * m * (n**2 - (l**2 + m**2) / 2) * Vdds - 2 * l * m*n**2 * Vddp + (l * m * (1 + n**2) / 2) * Vddd)

    elif orbital1 == 'dxz':
        if orbital2 == 's':
            return np.sqrt(3) * l * n * Vdss
        elif orbital2 == 'px':
            return np.sqrt(3) * l**2 * (-n) * Vdps + (-n) * (1 - 2 * l**2) * Vdpp
        elif orbital2 == 'py':
            return np.sqrt(3) * (-l) * n*m * Vdps +2* n * m*l * Vdpp
        elif orbital2 == 'pz':
            return np.sqrt(3) * n**2 * (-l) * Vdps + (-l) * (1 - 2 * n**2) * Vdpp
         #añadimos los d (dxy, dxz, dyz, dx2_y2, d3z2_r29)
        elif orbital2 == 'dxy':
            return 3 * l**2 * m * n * Vdds + m * n * (l**2 - 1) * Vddd  + m*n*(1-4*l**2)*Vddp
        elif orbital2 == 'dxz':
            return 3 * l**2 * n**2 * Vdds + (l**2 + n**2 - 4 * l**2 * n**2) * Vddp+(m**2+l**2*n**2)*Vddd      
        elif orbital2 == 'dyz':
            return 3 * l * m * n**2 * Vdds + l * m * (1 - 4 * n**2) * Vddp+ l*m*(n**2-1)*Vddd
        elif orbital2 == 'dx2_y2':
            return (3 / 2) * n * l * (l**2 - m**2) * Vdds + n * l * (1 - 2 * (l**2 - m**2)) * Vddp - (n * l *(1- (l**2 - m**2) / 2)) * Vddd
        elif orbital2 == 'dr':
            return np.sqrt(3) * (n * l * (n**2 - (l**2 + m**2) / 2) * Vdds + l * n * (l**2 + m**2 - n**2) * Vddp - (l * n * (l**2 + m**2) / 2) * Vddd)
   
    elif orbital1 == 'dyz':
        if orbital2 == 's':
            return np.sqrt(3) * n * m * Vdss
        elif orbital2 == 'px':
            return np.sqrt(3) * (-l) * n*m * Vdps +2* n * m*l * Vdpp
        elif orbital2 == 'py':
            return np.sqrt(3) * m**2 *(-n) * Vdps + (-n) * (1 - 2 * m**2) * Vdpp
        elif orbital2 == 'pz':
            return np.sqrt(3) * n**2 *(-m) * Vdps + (-m) * (1 - 2 * n**2) * Vdpp
         #añadimos los d (dxy, dxz, dyz, dx2_y2, d3z2_r29)
        elif orbital2 == 'dxy':
            return 3 * l * n * m**2 * Vdds + l * n * (1 - 4 * m**2) * Vddp+ l*n*(m**2-1)*Vddd
        elif orbital2 == 'dxz':
            return 3 * l * m * n**2 * Vdds + l * m * (1 - 4 * n**2) * Vddp+ l*m*(n**2-1)*Vddd    
        elif orbital2 == 'dyz':
            return 3 * m**2 * n**2 * Vdds + (n**2 + m**2 - 4 * n**2 * m**2) * Vddp+(l**2+n**2*m**2)*Vddd
        elif orbital2 == 'dx2_y2':
            return (3 / 2) * n * m * (l**2 - m**2) * Vdds - n * m * (1 + 2 * (l**2 - m**2)) * Vddp + (n * m *(1+ (l**2 - m**2) / 2)) * Vddd
        elif orbital2 == 'dr':
            return np.sqrt(3) * (n * m * (n**2 - (l**2 + m**2) / 2) * Vdds + m * n * (l**2 + m**2 - n**2) * Vddp - (n * m * (l**2 + m**2) / 2) * Vddd)

    elif orbital1 == 'dx2_y2':
        if orbital2 == 's':
            return np.sqrt(3) / 2 * (l**2 - m**2) * Vdss
        elif orbital2 == 'px':
            return np.sqrt(3) / 2 * (-l) * (l**2 - m**2) * Vdps + (-l) * (1 - l**2 + m**2) * Vdpp
        elif orbital2 == 'py':
            return np.sqrt(3) / 2 * (-m) * (l**2 - m**2) * Vdps - (-m) * (1 + l**2 - m**2) * Vdpp
        elif orbital2 == 'pz':
            return np.sqrt(3) / 2 * (-n) * (l**2 - m**2) * Vdps - (-n) * ( l**2 - m**2) * Vdpp
         #añadimos los d (dxy, dxz, dyz, dx2_y2, d3z2_r29)
        elif orbital2 == 'dxy':
            return (3 / 2) * l * m * (l**2 - m**2) * Vdds + 2 * l * m * (m**2 - l**2) * Vddp + (l * m * (l**2 - m**2) / 2) * Vddd
        elif orbital2 == 'dxz':
            return (3 / 2) * n * l * (l**2 - m**2) * Vdds + n * l * (1 - 2 * (l**2 - m**2)) * Vddp - (n * l *(1- (l**2 - m**2) / 2)) * Vddd   
        elif orbital2 == 'dyz':
            return (3 / 2) * n * m * (l**2 - m**2) * Vdds - n * m * (1 + 2 * (l**2 - m**2)) * Vddp + (n * m *(1+ (l**2 - m**2) / 2)) * Vddd
        elif orbital2 == 'dx2_y2':
            return (3 / 4) * (l**2 - m**2)**2 * Vdds + (l**2 + m**2 - (l**2 - m**2)**2) * Vddp + (n**2+(l**2 - m**2)**2 / 4) * Vddd
        elif orbital2 == 'dr':
            return np.sqrt(3) * ((l**2 - m**2) *( n**2 - (l**2 + m**2) / 2) * Vdds/2.0 + (n**2 * (m**2 - l**2)) * Vddp + ((1 + n**2) * (l**2 - m**2) / 4) * Vddd)

    elif orbital1 == 'dr':
        if orbital2 == 's':
            return (n**2 - (l**2 + m**2) / 2) * Vdss
        elif orbital2 == 'px':
            return (-l) * (n**2 - (l**2 + m**2) / 2) * Vdps - np.sqrt(3) * (-l) * n**2 * Vdpp
        elif orbital2 == 'py':
            return (-m) * (n**2 - (l**2 + m**2) / 2) * Vdps - np.sqrt(3) * (-m) * n**2 * Vdpp
        elif orbital2 == 'pz':
            return (-n) * (n**2 - (l**2 + m**2) / 2) * Vdps + np.sqrt(3) * (-n) * (l**2+m**2) * Vdpp
         #añadimos los d (dxy, dxz, dyz, dx2_y2, d3z2_r29)
        elif orbital2 == 'dxy':
            return np.sqrt(3) * (l * m * (n**2 - (l**2 + m**2) / 2) * Vdds - 2 * l * m*n**2 * Vddp + (l * m * (1 + n**2) / 2) * Vddd)
        elif orbital2 == 'dxz':
            return np.sqrt(3) * (n * l * (n**2 - (l**2 + m**2) / 2) * Vdds + l * n * (l**2 + m**2 - n**2) * Vddp - (l * n * (l**2 + m**2) / 2) * Vddd)
        elif orbital2 == 'dyz':
            return np.sqrt(3) * (n * m * (n**2 - (l**2 + m**2) / 2) * Vdds + m * n * (l**2 + m**2 - n**2) * Vddp - (n * m * (l**2 + m**2) / 2) * Vddd)
        elif orbital2 == 'dx2_y2':
            return np.sqrt(3) * ((l**2 - m**2) *( n**2 - (l**2 + m**2) / 2) * Vdds/2.0 + (n**2 * (m**2 - l**2)) * Vddp + ((1 + n**2) * (l**2 - m**2) / 4) * Vddd)
        elif orbital2 == 'dr':
            return (n**2 - (l**2 + m**2) / 2)**2 * Vdds + 3 * n**2 * (l**2 + m**2) * Vddp + (3 / 4) * (l**2 + m**2)**2 * Vddd

                             
    else:
        return 0

#----------------FUNCION PARA DEVUELTA DE VALORES------------------
def ParametrosOnsite( symbolo, orbital, EnergiasOnsite):
    if symbolo == 'S':
        if orbital == 's':
            return EnergiasOnsite['E_s_S']
        elif orbital == 'pz' or orbital == 'px' or orbital == 'py':
            return EnergiasOnsite['E_p_S']
        else:
            return EnergiasOnsite['E_d_S']
        
    elif symbolo == 'Mo':
        if orbital == 's':
            return EnergiasOnsite['E_s_Mo']
        elif orbital == 'pz' or orbital == 'px' or orbital == 'py':
            return EnergiasOnsite['E_p_Mo']  
        else:
            return EnergiasOnsite['E_d_Mo']
    else:
        return 0    

def SacarterminoSK(simbolo1, simbolo2, acoplo, parametros):
    """
    Busca el término acoplo teniendo en cuenta la simetría Vorb1orb2_A_B = Vorb2orb1_B_A.
    Devuelve float (0.0 si no existe).
    parametros debe ser dict con claves tipo (A,B) -> dict(acopl->valor)
    """
    # forma swap: intercambiar las dos letras orbitales inmediatamente después de 'V'
    core = acoplo[1:3]
    rest = acoplo[3:]
    swapped = "V" + core[1] + core[0] + rest

    try:
        return parametros[(simbolo1, simbolo2)][acoplo]
    except KeyError:    
        try:
            return parametros[(simbolo2, simbolo1)][swapped]
        except KeyError:
            return 0.0     

def es_hermitica(H, tol=1e-8):
    return np.allclose(H, H.conj().T, atol=tol) 


def es_semidefinida_positiva_compleja(A, tol=1e-10):
    A = np.asarray(A)
    
    # 1. Debe ser cuadrada
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        return False
    
    # 2. Debe ser hermítica
    if not np.allclose(A, A.conj().T, atol=tol):
        return False
    
    # 3. Autovalores (reales si es hermítica)
    autovalores = np.linalg.eigvalsh(A)
    
    # 4. Condición semidefinida positiva
    return np.all(autovalores >= -tol)


#-----------------CODIGO---------------------------
#archivos de entrada
symbols, posiciones = leer_positions('Posiciones.txt')
posiciones = np.array(posiciones)
orbitales = leer_orbitales('orbitales.txt')
Red = leer_Red('Red.txt')
Delta, Onsite = leer_delta_y_Onsite('sk_params.txt')
TOL = 1
#print(Onsite)

#para acceder a Onsite['E_s_S']
SK = leer_SK('sk_params.txt')
Sol = leer_SK('solape.txt')


#ahora construirmos los vecinos cercanos y la red reciproca
Red_reciproca = reciprocal_lattice(Red)
Vecinos = generar_vecinos(Red)
#ahora se puede recorrer el conjunto de vecinos cercanos 
k = 0.0000000000*Red_reciproca[0]
#k = [0.24618196, 0.14213322, 0.        ]



# --- preparar offsets correctos ---
offsets = [0]*len(posiciones)
for idx in range(1, len(posiciones)):
    offsets[idx] = offsets[idx-1] + len(orbitales[symbols[idx-1]])

N = offsets[-1] + len(orbitales[symbols[-1]])

#Definimos el Hamiltoniano
#Definimos el Hamiltoniano
Ha = np.zeros((N,N), dtype=complex)
HL = np.zeros((N,N), dtype=complex)

SA = np.zeros((N,N), dtype=complex)
SL = np.zeros((N,N), dtype=complex)

def Hamiltoniano(POSICIONES, SYMBOLS, ORBITALES, TRASLACION, SKPARAMETROS, DELTA = Delta, TOLERANCIA = TOL , n = N):
    H = np.zeros((n,n), dtype=complex)
    for i in range(len(POSICIONES)):
        symb1 = SYMBOLS[i]
        ri = POSICIONES[i]
        lenorb1=len(ORBITALES[symb1])
        for j in range(len(POSICIONES)):
                symb2 = SYMBOLS[j]
                rj = POSICIONES[j]
                lenorb2=len(ORBITALES[symb2])
                R = ri - rj - TRASLACION
                dist = np.linalg.norm(R)
                if dist > (DELTA+TOLERANCIA):
                    continue       
                for i_orb, orbital1 in enumerate(ORBITALES[symb1]):
                    for j_orb, orbital2 in enumerate(ORBITALES[symb2]):
                        idx_i = offsets[i] + i_orb
                        idx_j = offsets[j] + j_orb
                        if dist == 0:
                            if orbital1 == orbital2:
                                H[idx_i, idx_j] += ParametrosOnsite(symb1, orbital1, Onsite)
                                    
                            else: 
                                continue
                        else:
                            Vss        = SacarterminoSK(symb1, symb2, 'Vss', SKPARAMETROS)
                            Vsp        = SacarterminoSK(symb1, symb2, 'Vsp', SKPARAMETROS) 
                            Vps        = SacarterminoSK(symb1, symb2, 'Vps', SKPARAMETROS)
                            Vpp_sigma  = SacarterminoSK(symb1, symb2, 'Vpp_sigma', SKPARAMETROS)
                            Vpp_pi     = SacarterminoSK(symb1, symb2, 'Vpp_pi', SKPARAMETROS)
                            Vdds       = SacarterminoSK(symb1, symb2, 'Vdds', SKPARAMETROS)
                            Vddp       = SacarterminoSK(symb1, symb2, 'Vddp', SKPARAMETROS)
                            Vddd       = SacarterminoSK(symb1, symb2, 'Vddd', SKPARAMETROS)
                            Vsds       = SacarterminoSK(symb1, symb2, 'Vsds', SKPARAMETROS)
                            Vdss       = SacarterminoSK(symb1, symb2, 'Vdss', SKPARAMETROS)
                            Vpdp       = SacarterminoSK(symb1, symb2, 'Vpdp', SKPARAMETROS)
                            Vpds       = SacarterminoSK(symb1, symb2, 'Vpds', SKPARAMETROS)
                            Vdps       = SacarterminoSK(symb1, symb2, 'Vdps', SKPARAMETROS)
                            Vdpp       = SacarterminoSK(symb1, symb2, 'Vdpp', SKPARAMETROS)

                            objhl= funcionslaterkoster(R,orbital1, orbital2, Vss, Vsp, Vps, Vsds,Vdss, Vpp_sigma ,Vpp_pi ,Vpds, Vdps , Vpdp, Vdpp ,Vdds ,Vddp ,Vddd)
                            
                            H[idx_i, idx_j] += objhl
    return H

def Solape(POSICIONES, SYMBOLS, ORBITALES, TRASLACION, SKPARAMETROS, DELTA = Delta, TOLERANCIA = TOL , n = N):
    H = np.zeros((n,n), dtype=complex)
    for i in range(len(POSICIONES)):
        symb1 = SYMBOLS[i]
        ri = POSICIONES[i]
        lenorb1=len(ORBITALES[symb1])
        for j in range(len(POSICIONES)):
                symb2 = SYMBOLS[j]
                rj = POSICIONES[j]
                lenorb2=len(ORBITALES[symb2])
                R = ri - rj - TRASLACION
                dist = np.linalg.norm(R)
                if dist > (DELTA+TOLERANCIA):
                    continue       
                for i_orb, orbital1 in enumerate(ORBITALES[symb1]):
                    for j_orb, orbital2 in enumerate(ORBITALES[symb2]):
                        idx_i = offsets[i] + i_orb
                        idx_j = offsets[j] + j_orb
                        if dist == 0:
                            if orbital1 == orbital2:
                                H[idx_i, idx_j] += 1
                                    
                            else: 
                                continue
                        else:
                            Vss        = SacarterminoSK(symb1, symb2, 'Vss', SKPARAMETROS)
                            Vsp        = SacarterminoSK(symb1, symb2, 'Vsp', SKPARAMETROS) 
                            Vps        = SacarterminoSK(symb1, symb2, 'Vps', SKPARAMETROS)
                            Vpp_sigma  = SacarterminoSK(symb1, symb2, 'Vpp_sigma', SKPARAMETROS)
                            Vpp_pi     = SacarterminoSK(symb1, symb2, 'Vpp_pi', SKPARAMETROS)
                            Vdds       = SacarterminoSK(symb1, symb2, 'Vdds', SKPARAMETROS)
                            Vddp       = SacarterminoSK(symb1, symb2, 'Vddp', SKPARAMETROS)
                            Vddd       = SacarterminoSK(symb1, symb2, 'Vddd', SKPARAMETROS)
                            Vsds       = SacarterminoSK(symb1, symb2, 'Vsds', SKPARAMETROS)
                            Vdss       = SacarterminoSK(symb1, symb2, 'Vdss', SKPARAMETROS)
                            Vpdp       = SacarterminoSK(symb1, symb2, 'Vpdp', SKPARAMETROS)
                            Vpds       = SacarterminoSK(symb1, symb2, 'Vpds', SKPARAMETROS)
                            Vdps       = SacarterminoSK(symb1, symb2, 'Vdps', SKPARAMETROS)
                            Vdpp       = SacarterminoSK(symb1, symb2, 'Vdpp', SKPARAMETROS)

                            objhl= funcionslaterkoster(R,orbital1, orbital2, Vss, Vsp, Vps, Vsds,Vdss, Vpp_sigma ,Vpp_pi ,Vpds, Vdps , Vpdp, Vdpp ,Vdds ,Vddp ,Vddd)
                            
                            H[idx_i, idx_j] += objhl
    return H
                            
""" k =   [ 0.24618196,  0.14213322 ,-0.        ]                      
for z in range(len(Vecinos)):
        
    Vec =np.array(Vecinos[z])
    phaseL = np.exp(1j * np.dot(k, Vec)) #version Latice
    HL = HL + phaseL*Hamiltoniano(posiciones, symbols, orbitales, Vec, SK)
    SL = SL + phaseL*Solape(posiciones, symbols, orbitales, Vec, Sol)
print(k, es_hermitica(SL))
eigenvaluesL, eigenvectorsL =  np.linalg.eigh(SL)     """
                        
k = 0.0000000000*Red_reciproca[0]
#print(SacarterminoSK('S', 'Mo', 'Vdss', SK))
for ik in range(20):
    k += Red_reciproca[0]*0.025
    HL = np.zeros((N,N), dtype=complex)
    SL = np.zeros((N,N), dtype=complex)
    for z in range(len(Vecinos)):
        
        Vec =np.array(Vecinos[z])
        phaseL = np.exp(1j * np.dot(k, Vec)) #version Latice
        HL = HL + phaseL*Hamiltoniano(posiciones, symbols, orbitales, Vec, SK)
        SL = SL + phaseL*Solape(posiciones, symbols, orbitales, Vec, Sol)
    print(k, es_hermitica(SL))
    eigenvaluesL, eigenvectorsL = eigh(SL)
    print(eigenvaluesL)


#print(SL)

eigenvaluesL, eigenvectorsL =  np.linalg.eigh(HL)
print(es_hermitica(SL))
#print(HL)
#HL = HL + np.triu(HL,1).conj().T
#SL = SL + np.triu(SL,1).conj().T
#SL = 0.5*(SL + SL.conj().T)
print("Determinante", np.linalg.det(HL))
print("NaN:", np.isnan(HL).any())
print("Inf:", np.isinf(HL).any())
print("Condición:", np.linalg.cond(HL))
print("Simétrica:", np.allclose(HL, HL.T.conj()))
#print(es_hermitica(SL))
#eigenvaluesL, eigenvectorsL = np.linalg.eigh(SL)
print(eigenvaluesL)

# Reemplaza la siguiente línea por la matriz que pegaste.




#print("Autovalor negativo:", evals[idx])
#print("Vector propio asociado:")
#print(evecs[:, idx])
