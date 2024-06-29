import matplotlib.pyplot as plt
import numpy as np
import matplotlib.tri as tri
import math
import numpy.random as rd
from mpl_toolkits.mplot3d import Axes3D
import time
import pyvista as pv

## Import du maillage : chemin d'accès A MODIFIER
vtk_mesh = pv.read(r"        ") ## REMPLIR CETTE LIGNE AVEC LE CHEMIN D'ACCES VERS L'UN DES MAILLAGES .vtk
X=vtk_mesh.points[:,0]
Y=vtk_mesh.points[:,1]
X=X.flatten()
Y=Y.flatten()
x_m = 0                         # x min du domaine sur lequel on cherche la solution
x_M = 1                         # x max du domaine sur lequel on cherche la solution, modifier cette valeur si le maillage choisi est défini su [0,5]x[0,5]
y_m = 0                         # y min du domaine sur lequel on cherche la solution
y_M = 1                         # y max du domaine sur lequel on cherche la solution, idem

triang = tri.Triangulation(X, Y)    # Maillage qui correspond à la triangulation de Delaunay à partir des points de coordonnées données par X[i] et Y[i]
NTri=np.shape(triang.triangles)[0] # Nombre de triangles du maillage
lsom = np.array([triang.triangles[j] for j in range(NTri)])                                   # Liste des sommets des triangles
lcs = np.array([[[triang.x[k], triang.y[k]] for k in lsom[j]] for j in range(NTri)])          # Liste des coordonnées des sommets des triangle
lcc = np.array([1/3*(lcs[j][0]+lcs[j][1]+lcs[j][2]) for j in range(NTri)])    ### Liste des coordonnées des centres des triangles

### Surfaces initiales : A MODIFIER selon les simulations souhaitées

def hmaillee(f):
    return f(lcc[:, 0], lcc[:, 1])

def planincline(x, y):
    a = 1                                           # Coefficient directeur selon x
    b = 1                                           # Coefficient directeur selon y
    c = 1                                           # Hauteur minimale
    return a*x+b*y+c

def gaussienne(x, y):
    a = 4                                            # Amplitude de la gaussienne
    l = 0.2                                          # Largeur typique de la gaussienne
    hmin = 1                                         # Hauteur minimale
    xc = 1/2*(x_M+x_m)                               # Abscisse du centre de la gaussienne
    yc = 1/2*(y_M+y_m)                               # Ordonnée du centre de la gaussienne
    return hmin + a*np.exp((-(x-xc)**2-(y-yc)**2)/l**2)

###  A MODIFIER Conditions initiales
tf = 2                                               # Temps final du calcul de la solution approchée
lhi = hmaillee(planincline)                          # Hauteur initiale, rentrée dans la fonction hmaillee planincline ou gaussienne
lvxi = np.zeros(NTri)                                # Liste des flux de vitesse initiale selon x
lvyi = np.zeros(NTri)                                # Liste des flux de vitesse initiale selon y

zmin=1                                               # Ordonnée minimale de la fenêtre d'affichage
zmax=3                                               # Ordonnée maximale de la fenêtre d'affichage


### Norme euclidienne
def ne(x):
    '''Retourne la norme euclidienne de x'''
    return np.sqrt(x[0]**2 + x[1]**2)

### Calculs sur le maillage : sommets, coordonnées, surface, rayon maximal

def surf(cs):
    '''Renvoie l'aire d'un triangle dont cs est la liste de ses coordonnées'''
    return 1/2*np.abs((cs[0][0]-cs[2][0])*(cs[1][1]-cs[2][1]) - (cs[0][1]-cs[2][1])*(cs[1][0]-cs[2][0]))

lsurf = np.array([surf(lcs[j]) for j in range(NTri)])                                         # Liste des surfaces des triangles

Smax = np.max(lsurf)

Lr=[] #Liste des périmètres des triangles
for N in range(NTri):
    sommets = triang.triangles[N]
    cs1 = np.array([triang.x[sommets[0]], triang.y[sommets[0]]])
    cs2 = np.array([triang.x[sommets[1]], triang.y[sommets[1]]])
    cs3 = np.array([triang.x[sommets[2]], triang.y[sommets[2]]])
    p = ne(cs1-cs2)+ne(cs2-cs3)+ne(cs3-cs1)
    s = 1/2*np.abs((cs1[0]-cs3[0])*(cs2[1]-cs3[1]) - (cs1[1]-cs3[1])*(cs2[0]-cs3[0]))
    r = 2*s/p
    Lr += [r]
rmin = np.min(Lr) #rayon minimal des triangles inscrits dans la grille

### Calculs sur le maillage : arêtes et triangles adjacents

lar = triang.edges                                                                  # Liste des arêtes, définies par le numéro de leurs sommets
lcssar = [[[triang.x[lar[na][l]], triang.y[lar[na][l]]] for l in range(2)] for na in range(len(lar))]       # Liste des coordonnées des sommets des arrêtes

def triang_adj(na):
    '''Renvoie les triangles qui ont pour côté l'arête d'indice na rangézs dans l'ordre croissant'''
    adj = []
    for i in range(NTri):
        if lar[na][0] in lsom[i] and lar[na][1] in lsom[i]:
            adj.append(i)
    return adj

lta = [triang_adj(j) for j in range(len(lar))]                            # Liste des triangles adjacents aux arêtes

### Calculs sur le maillage : longueur des arêtes
llar = np.array([ne([lcssar[na][1][0]-lcssar[na][0][0], lcssar[na][1][1]-lcssar[na][0][1]]) for na in range(len(lar))]) # Liste des longueurs des arêtes

### Calculs sur le maillage : normale aux arêtes

def n_ext(na):
    '''Renvoie la normale à l'arête na, orientée vers le triangle de plus petit indice'''
    som = lar[na]
    cs = lcssar[na]
    v = [cs[1][0] - cs[0][0], cs[1][1] - cs[0][1]]
    n = [-v[1], v[0]]
    t = max(lta[na])
    z1 = [cs[1][0] + n[0], cs[1][1] + n[1]]
    z2 = [cs[1][0] - n[0], cs[1][1] - n[1]]
    k = 0
    while lsom[t][k] in som:
        k += 1
    s3 = np.array(lcs[t][k])
    if ne(np.array(z1) - s3) < ne(np.array(z2) - s3):
        n = -np.array(n)
    n = n/ne(n)
    return n

lnar = np.array([n_ext(na) for na in range(len(lar))])                                        # Liste des normales aux arêtes

### Calculs sur le maillage : bords

bordtriang = [False for i in range(len(lar))]                                       # Liste qui détermine si les triangles sont au bord
for j in range(NTri):
    if -1 in triang.neighbors[j]:
        bordtriang[j] = True

bordar = [False for i in range(len(lar))]                                           # Liste qui détermine si les arêtes sont au bord

for na in range(len(lar)):
    if len(lta[na]) < 2:
        bordar[na] = True

### Calculs sur le maillage : centre des triangles
lcc = np.array([1/3*(lcs[j][0]+lcs[j][1]+lcs[j][2]) for j in range(NTri)])

### Calculs sur le maillage : liste des arrêtes et triangles au bord et non au bord
larint = []
larext = []
for na in range(len(lar)):
    if not bordar[na]:
        larint.append(na)
    else:
        larext.append(na)

lnarint = np.array([lnar[na] for na in larint])
lnarext = np.array([lnar[na] for na in larext])
llarint = np.array([llar[na] for na in larint])

ltint = []
ltext = []
for j in range(NTri):
    if bordtriang[j]:
        ltext.append(j)
    else:
        ltint.append(j)



### Fonctions pour l'approximation

g = 9.81

def ps1(lv, lu):
    '''Renvoie une liste de produit scalaire entre les colonnes de lv et les lignes de lu'''
    return np.array([lv[0][i]*lu[i][0] + lv[1][i]*lu[i][1] for i in range(len(lv[0]))])

def ps2(lm, lu):
    '''Renvoie une matrice de produit scalaire entre une les colonnes d'une liste de matrices et les lignes de lu (Cf cas d'utilisation dans la fonction flux)'''
    return np.array([[lm[j][k][0]*lu[j][0] + lm[j][k][1]*lu[j][1] for k in range(3)] for j in range(len(lm))])

def maxtat(l1, l2):
    return np.array([max(l1[i], l2[i]) for i in range(len(l1))])

def produit(l, m):
    '''Renvoie un produit entre une liste et une matrice, la ième ligne de la matrice est multipliée par le ième coefficient de la liste (Cf cas d'utilisation dans la fonction flux)'''
    return np.array([[l[i]*m[i][j] for j in range(len(m[0]))] for i in range(len(l))])

def F(h, qu, qv):
    return [[qu, qv], [qu**2/h+g*h**2/2, qu*qv/h], [qu*qv/h, qv**2/h+g*h**2/2]]

def ctecorr(lh, lqu, lqv, lnarint):
    return np.abs((1/lh)*ps1([lqu, lqv], lnarint)) + np.sqrt(g*lh)

def flux(lU0, lU1, lF0, lF1, lc, llarint, lnarint):
    '''Renvoie la liste des flux numériques traversant les arrêtes intérieures, suivant la normale du triangle d'indice le plus grand vers le triangle de plus petit indice'''
    return produit(llarint, 1/2*ps2(lF0+lF1, lnarint)) - produit(llarint, 1/2*produit(lc,(lU0-lU1)))

def mu_plus(U):
    if U[0] != 0:
        return U[1]/U[0]+np.sqrt(g*U[0])
    else :
        return 0

def mu_moins(U):
    if U[0] != 0:
        return U[1]/U[0]-np.sqrt(g*U[0])
    else :
        return 0

def nu_plus(U):
    if U[0] != 0:
        return U[2]/U[0]+np.sqrt(g*U[0])
    else :
        return 0

def nu_moins(U):
    if U[0] != 0:
        return U[2]/U[0]-np.sqrt(g*U[0])
    else :
        return 0

def lambda_plus(U):
    return ne([mu_plus(U), nu_plus(U)])

def lambda_moins(U):
    return ne([mu_moins(U), nu_moins(U)])


def sol_appr(tf, lUi):

    '''Cette fonction renvoie une valeur approchée de la solution jusqu'au temps tf avec un pas de temps dt et un maillage à n volumes'''

    lt = [0]

    # Initialisation de la matrice
    lU = [lUi]

    while lt[-1]<tf :
        dt = 0.9*rmin /np.max([max(lambda_plus(lU[-1][k]), lambda_moins(lU[-1][k])) for k in range(NTri)])

        lt.append(lt[-1] + dt)

        if len(lt)%200==0:
            print("solution calculée jusqu'au temps t = {} s".format(np.round(lt[-1],2)))


        # Calcul pour les arêtes non au bord
        lF = np.array(F(lU[-1][:, 0], lU[-1][:, 1], lU[-1][:, 2])) #Flux au centre de chaque cellule
        lU0 = np.array([lU[-1][lta[na][0]] for na in larint]) # par rapport à une arête intérieure:  U du triangle d'indice le plus petit
        lU1 = np.array([lU[-1][lta[na][1]] for na in larint]) # U d'indice le plus grand
        lc0 = ctecorr(lU0[:, 0], lU0[:, 1], lU0[:, 2], lnarint)
        lc1 = ctecorr(lU1[:, 0], lU1[:, 1], lU1[:, 2], lnarint)
        lc = maxtat(lc0, lc1)

        lF0 = np.array([lF[:,:,lta[na][0]] for na in larint])
        lF1 = np.array([lF[:,:,lta[na][1]] for na in larint])
        lflux = flux(lU0, lU1, lF0, lF1, lc, llarint, lnarint)

        lU_temp = np.copy(lU[-1][:][:])
        for j, na in enumerate(larint):
            t0 = lta[na][0] #triangle d'indice le plus petit à côté de na
            t1 = lta[na][1] #triangle d'indice le plus grand
            lU_temp[t1] = lU_temp[t1]-dt/lsurf[t1]*lflux[j] ### !!!
            lU_temp[t0] = lU_temp[t0]+dt/lsurf[t0]*lflux[j] ### !!!

        # Conditions de bord
        for j in ltext:
            lU_temp[j][1] = 0
            lU_temp[j][2] = 0
        lU.append(lU_temp)

    return lU, lt




lUi = np.array([[lhi[j], lvxi[j], lvyi[j]] for j in range(NTri)])
start = time.time()
lU, lt = sol_appr(tf, lUi)
end = time.time()
print('temps de calcul:')
print(end-start)


### Animation
from pylab import *

plt.ion()


fpt= 100
lk = np.arange(0,tf, 1/(fpt*tf))
i = 0
x = lcc[:, 0]
y = lcc[:, 1]

for k, t in enumerate(lk):
    while lt[i]<=t:
        i += 1

    fig = plt.figure(1)
    plt.clf()
    z = lU[i][:, 0]
    ax = fig.gca(projection='3d')
    ax.scatter(x,y,z, zdir='z',s=30,depthshade=True)
    ax.set_xlim3d([0, 1]) ; ax.set_xlabel('X en m')
    ax.set_ylim3d([0, 1]) ; ax.set_ylabel('Y en m')
    ax.set_zlim3d([zmin, zmax]) ; ax.set_zlabel('Z en m')
    plt.suptitle("t={} sec".format(round(lt[i],2)))
    pause(10**(-10))

