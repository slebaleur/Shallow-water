import matplotlib.pyplot as plt
import numpy as np
import matplotlib.tri as tri
import math
import numpy.random as rd
from mpl_toolkits.mplot3d import Axes3D
import time
import pyvista as pv


### Maillage gmsh
vtk_mesh = pv.read(r" ") ## REMPLIR CETTE LIGNE AVEC LE CHEMIN D'ACCES VERS L'UN DES MAILLAGES .vtk
X=vtk_mesh.points[:,0]
Y=vtk_mesh.points[:,1]
X=X.flatten()
Y=Y.flatten()
x_m = 0                # x min du domaine sur lequel on cherche la solution
x_M = 1                # x max du domaine sur lequel on cherche la solution, modifier cette valeur si le maillage choisi est défini sur [0,5]x[0,5]
y_m = 0                # y min du domaine sur lequel on cherche la solution
y_M = 1                # y max du domaine sur lequel on cherche la solution, modifier cette valeur si le maillage choisi est défini sur [0,5]x[0,5]

triang = tri.Triangulation(X, Y)    # Maillage qui correspond à la triangulation de Delaunay à partir des points de coordonnées données par X[i] et Y[i]
NTri=np.shape(triang.triangles)[0] # Nombre de triangles du maillage

lsom = np.array([triang.triangles[j] for j in range(NTri)])                             # Liste des sommets des triangles
lcs = np.array([[[triang.x[k], triang.y[k]] for k in lsom[j]] for j in range(NTri)])    # Liste des coordonnées des sommets des triangles
lcc = np.array([1/3*(lcs[j][0]+lcs[j][1]+lcs[j][2]) for j in range(NTri)])    ### Liste des coordonnées des centres des triangles

def fnmaillee(f):
    '''Renvoie la liste des valeurs de f au niveau des centres de chaque volume du maillage'''
    return np.array([f(lcc[j][0], lcc[j][1]) for j in range(len(lcc))])

### A MODIFIER Surfaces initiales et fond

def planincline_fond(x, y): # Plan incliné utilisé pour le fond
    return az*x+bz*y+cz

az = 0.4                 # Coefficient directeur selon x A MODIFIER
bz = 0.4                 # Coefficient directeur selon y A MODIFIER
cz = 0                   # Hauteur minimale
derzdy = bz*np.ones(NTri)
derzdx = az*np.ones(NTri)

def planincline_surface(x, y): # Plan incliné utilisé pour la surface
    a = -0.4                   # Coefficient directeur selon x A MODIFIER
    b = -0.4                   # Coefficient directeur selon y A MODIFIER
    c = 1                      # Hauteur minimale
    return a*x+b*y+c

def gaussienne(x, y):
    a = 2                                            # Amplitude de la gaussienne
    l = 0.1                                          # Largeur typique de la gaussienne
    hmin = 1.1                                       # Hauteur minimale
    xc = 2/5*(x_M+x_m)                               # Abscisse du centre de la gaussienne
    yc = 2/5*(y_M+y_m)                               # Ordonnée du centre de la gaussienne
    return hmin + a*np.exp((-(x-xc)**2-(y-yc)**2)/l**2)

def h(lz, ls):
    return [max(ls[i]-lz[i], 0) for i in range(NTri)]

###  A MODIFIER Conditions initiales

tf = 1                                          # Temps final
lsi = fnmaillee(planincline_surface)            # Profil initial de la hauteur
lz = fnmaillee(planincline_fond)                # Profil du fond
lhi = h(lz, lsi)                                # Hauteur initiale
lUi = [[lhi[j], 0, 0] for j in range(NTri)]     # Condition initiale
cfl=0.45                                        # Constante paramétrant le pas de temps, à diminuer si des instabilités apparaissent





# Norme euclidienne
def ne(x):
    '''Retourne la norme euclidienne de x'''
    return np.sqrt(x[0]**2 + x[1]**2)

# Calculs sur le maillage : sommets, coordonnées, surface, rayon minimal



def surf(cs):
    '''Renvoie l'aire d'un triangle dont cs est la liste de ses coordonnées'''
    return 1/2*np.abs((cs[0][0]-cs[2][0])*(cs[1][1]-cs[2][1]) - (cs[0][1]-cs[2][1])*(cs[1][0]-cs[2][0]))

lsurf = np.array([surf(lcs[j]) for j in range(NTri)])                     # Liste des surfaces des triangles

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

# Calculs sur le maillage : arêtes et triangles adjacents

lar = triang.edges                              # Liste des arêtes, définies par le numéro de leurs sommets
lcssar = [[[triang.x[lar[na][l]], triang.y[lar[na][l]]] for l in range(2)] for na in range(len(lar))]       # Liste des coordonnées des sommets des arrêtes

def triang_adj(na):
    '''Renvoie les triangles qui ont pour côté l'arête d'indice na rangés dans l'ordre croissant'''
    adj = []
    for i in range(NTri):
        if lar[na][0] in lsom[i] and lar[na][1] in lsom[i]:
            adj.append(i)
    return adj

lta = [triang_adj(j) for j in range(len(lar))]                   # Liste des triangles adjacents aux arêtes

# Calculs sur le maillage : longueur des arêtes
llar = np.array([ne([lcssar[na][1][0]-lcssar[na][0][0], lcssar[na][1][1]-lcssar[na][0][1]]) for na in range(len(lar))]) # Liste des longueurs des arêtes

# Calculs sur le maillage : normale aux arêtes

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

lnar = np.array([n_ext(na) for na in range(len(lar))])     # Liste des normales aux arêtes

# Calculs sur le maillage : bords

bordtriang = [False for i in range(len(lar))]           # Liste qui détermine si les triangles sont au bord
for j in range(NTri):
    if -1 in triang.neighbors[j]:
        bordtriang[j] = True

bordar = [False for i in range(len(lar))]               # Liste qui détermine si les arêtes sont au bord

for na in range(len(lar)):
    if len(lta[na]) < 2:
        bordar[na] = True



# Calculs sur le maillage : liste des arrêtes et triangles au bord et non au bord
larint = []                                         ### Liste des arêtes qui ne sont pas au bord
larext = []                                         ### Liste des arêtes qui sont au bord
for na in range(len(lar)):
    if not bordar[na]:
        larint.append(na)
    else:
        larext.append(na)

lnarint = np.array([lnar[na] for na in larint])    ### Liste des normales des arêtes qui ne sont pas au bord
llarint = np.array([llar[na] for na in larint])    ### Liste des longueurs des arêtes qui ne sont pas au bord

ltint = []                                         ### Liste des triangles qui ne sont pas au bord
ltext = []                                         ### Liste des triangles qui sont au bord
for j in range(NTri):
    if bordtriang[j]:
        ltext.append(j)
    else:
        ltint.append(j)

# Calculs sur le maillage : liste des triangles adjacents aux triangles intérieurs

ltat = []

NAr = len(lar)

for t in range(NTri):
    adj =  [] # liste triangles adjacents au triangle t
    for na in larint:
        if t == lta[na][0]:
            adj.append(lta[na][1])
        if t == lta[na][1]:
            adj.append(lta[na][0])
    ltat.append(adj)



# Fonctions pour l'approximation
g = 9.81     # Constante de pesanteur
def Fsc(U, n):
    ''' Renvoie le produit scalaire de F avec la normale n'''
    h = U[0]
    qu = U[1]
    qv = U[2]
    return np.array([np.dot([qu, qv], n), np.dot([qu**2/h+g*h**2/2, qu*qv/h], n), np.dot([qu*qv/h, qv**2/h+g*h**2/2], n)])

def ps1(lv, lu):
    '''Renvoie une liste de produit scalaire entre les colonnes de lv et les lignes de lu'''
    return np.array([lv[0][i]*lu[i][0] + lv[1][i]*lu[i][1] for i in range(len(lv[0]))])

def maxtat(l1, l2):
    '''Renvoie une liste qui correspond au maximum terme à terme des listes l1 et l2'''
    return np.array([max(l1[i], l2[i]) for i in range(len(l1))])

def mintat(l1, l2):
    '''Renvoie une liste qui correspond au minimum terme à terme des listes l1 et l2'''
    return np.array([min(l1[i], l2[i]) for i in range(len(l1))])

def lplus(nar, U0, U1):
    '''Renvoie la liste des maximum des valeurs propres au niveau de chaque arête intérieure'''
    if U0[0] == 0 and U1[0] == 0:
        return 0
    if U0[0] <= 0 and U1[0] <= 0:
        return 0
    elif U0[0] <= 0:
        return -1
    elif U1[0] <= 0:
        return 1
    else:
        p0 = 1/U0[0]*np.dot([U0[1], U0[2]], nar) + np.sqrt(g*U0[0])
        p1 = 1/U1[0]*np.dot([U1[1], U1[2]], nar) + np.sqrt(g*U1[0])
        return max(p0, p1)

def lmoins(nar, U0, U1):
    '''Renvoie la liste des minimum des valeurs propres au niveau de chaque arête intérieure'''
    if U0[0] == 0 and U1[0] == 0:
        return 0
    if U0[0] <= 0 and U1[0] <= 0:
        return 0
    elif U0[0] <= 0:
        return -1
    elif U1[0] <= 0:
        return 1
    else:
        m0 = 1/U0[0]*np.dot([U0[1], U0[2]], nar) - np.sqrt(g*U0[0])
        m1 = 1/U1[0]*np.dot([U1[1], U1[2]], nar) - np.sqrt(g*U1[0])
        return min(m0, m1)

def F_num(lm, lp, U0, U1, next):
    '''Renvoie le flux numérique HLL'''
    if lm > 0 :
        return Fsc(U0, next)
    elif lp < 0:
        return Fsc(U1, next)
    elif lp == lm == 0:
        return np.array([0,0,0])
    else :
        return (lp*Fsc(U1, next)-lm*Fsc(U0, next)+lp*lm*(U0-U1))/(lp-lm)

def s(lUp,dt):
    return [[0, -dt*lUp[j][0]*g*derzdx[j], -dt*lUp[j][0]*g*derzdy[j]] for j in range(NTri)]

def Usec(j, lU_prec):
    qu = 0
    qv = 0
    l = 0
    for k in ltat[j]:
        if lU_prec[k][0] > 10**(-5):
            nvk = lU_prec[k][1]**2/lU_prec[k][0]**2 + lU_prec[k][2]**2/lU_prec[k][0]**2
            if nvk + 2*g*(lz[k]- lz[j]) < 0 or nvk < 10**(-5):
                f = 0
            else:
                f = np.sqrt((nvk + 2*g*(lz[k] - lz[j]))/nvk)
            qu += f*lU_prec[k][1]
            qv += f*lU_prec[k][2]
            l += 1
    if l != 0:
        qu /= l
        qv /= l

    return qu, qv

def L(k,na, lU0, lU1):#lambda plus quand on se place du point de vue du triangle d'indice le plus grand comme triangle intérieur
    U0 = lU0[lta[na][0]]
    U1 = lU1[lta[na][1]]
    lambda1 = np.array([max(mu_plus(U0), mu_plus(U1)), max(nu_plus(U0), nu_plus(U1))])
    lambda2 = np.array([max(mu_plus(U0), mu_plus(U1)), min(nu_moins(U0), nu_moins(U1))])
    lambda3 = np.array([min(mu_moins(U0), mu_moins(U1)), max(nu_plus(U0), nu_plus(U1))])
    lambda4 = np.array([min(mu_moins(U0), mu_moins(U1)), min(nu_moins(U0), nu_moins(U1))])
    n = lnarint[k]
    return [max(np.dot(lambda1,n),np.dot(lambda2,n),np.dot(lambda3,n),np.dot(lambda4,n)), min(np.dot(lambda1,n),np.dot(lambda2,n),np.dot(lambda3,n),np.dot(lambda4,n))]

def mu_plus(U):
    if U[0] > 0:
        return U[1]/U[0]+np.sqrt(g*U[0])
    else :
        return 0

def mu_moins(U):
    if U[0] > 0:
        return U[1]/U[0]-np.sqrt(g*U[0])
    else :
        return 0

def nu_plus(U):
    if U[0] > 0:
        return U[2]/U[0]+np.sqrt(g*U[0])
    else :
        return 0

def nu_moins(U):
    if U[0] > 0:
        return U[2]/U[0]-np.sqrt(g*U[0])
    else :
        return 0

def lambda_plus(U):
    return ne([mu_plus(U), nu_plus(U)])

def lambda_moins(U):
    return ne([mu_moins(U), nu_moins(U)])

def sol_appr(tf, lUi, cfl):

    '''Cette fonction renvoie une valeur approchée de la solution jusqu'au temps tf'''
    #initialisation de la liste des temps et de la liste des vecteurs U
    lt = [0]
    lU = [lUi]

    while lt[-1]<tf :
        dt = cfl*rmin /np.max([max(lambda_plus(lU[-1][k]), lambda_moins(lU[-1][k])) for k in range(NTri)])
        lt.append(lt[-1] + dt)
        if len(lt) % 150 == 0:
            print("solution calculée jusqu'au temps t={}s".format(np.round(lt[-1], 4)))             # Affichage du temps où le programme est en train de calculer la solution approchée
        # Listes permettant d'obtenir U pour les triangles adjacents aux arêtes
        lUp = [lU[-1][j] for j in range(NTri)]
        lU0 = np.array([lU[-1][lta[na][0]] for na in larint])
        lU1 = np.array([lU[-1][lta[na][1]] for na in larint])

        # Listes des plus petites et plus grandes valeurs propres liées à chaque arête intérieure
        lLm = [lmoins(lnarint[k], lU0[k], lU1[k]) for k in range(len(larint))]
        lLp = [lplus(lnarint[k], lU0[k], lU1[k]) for k in range(len(larint))]

        # Listes des flux numériques par rapport aux arêtes intérieures
        lF_num = np.zeros((len(larint), 3))   # Initialisation de la liste
        for k in range(len(larint)):
            lF_num[k] = F_num(lLm[k], lLp[k], lU0[k], lU1[k], lnarint[k])

        # Calcul des nouvelles valeurs de U dans chaque triangle
        lU_temp = np.copy(lU[-1])         # On initialise avec les valeurs du temps précédent
        for k,na in enumerate(larint):
            t0 = lta[na][0] # Indice du triangle adjacent à na d'indice le plus petit
            t1 = lta[na][1] # Indice du triangle adjacent à na d'indice le plus grand
            lU_temp[t1] -= dt*llar[na]/lsurf[t1]*lF_num[k]
            lU_temp[t0] += dt*llar[na]/lsurf[t0]*lF_num[k]

        ls = s(lUp,dt)

        lU_temp += s(lUp,dt)

        lU_temp = np.round(lU_temp, 7)

        for j in range(NTri):
            if lU_temp[j][0] >= 10**(-4) and lU[-1][j][0] == 0:
                lU_temp[j][1], lU_temp[j][2] = Usec(j, lU[-1])[:]
            if lU_temp[j][0] < 10**(-4):
                lU_temp[j][0] = 0

        # Conditions de bord
        for j in ltext:
            lU_temp[j][1] = 0
            lU_temp[j][2] = 0
        lU.append(lU_temp)

    return lU, lt


start = time.time()                             # Heure de départ du programme
lU,lt = sol_appr(tf, lUi, cfl)
end = time.time()                               # Heure de fin du programme
print('temps de calcul:', end-start)            # Affichage du temps du programme

# Réécriture des listes : les mailles vides sont effacées

fpt = 300

lk = np.arange(0,tf, 1/(fpt*tf))
li = [] #liste qui va contenir les indices correspondant à la subdivision régulière de temps définie par lk

i=0

for t in lk:
    while lt[i]<t:
        i+=1
    li.append(i)


mx = [[] for i in range(len(li))]
my = [[] for i in range(len(li))]
mh = [[] for i in range(len(li))]
mz = [[] for i in range(len(li))]

for i, j in enumerate(li):
    for t in range(NTri):
        if lU[j][t][0] > 10**(-5):
            mx[i].append(lcc[t][0])
            my[i].append(lcc[t][1])
            mh[i].append(lU[j][t][0])
            mz[i].append(lz[t])

# Affichage de la hauteur pour un temps donné avec affichage spécifique zone sèche

from pylab import *
plt.ion()
i = 0


lx = lcc[:, 0]
ly = lcc[:, 1]

lzi = mz[i]
lhi =[mh[i][j] + lzi[j] for j in range(len(mh[i]))]
lxi = mx[i]
lyi = my[i]


fig = plt.figure(1)

plt.show()

# Animation avec affichage spécifique zone sèche
from pylab import *

plt.ion()

x = lcc[:, 0]
y = lcc[:, 1]

lx = lcc[:, 0]
ly = lcc[:, 1]


plt.clf()
ax = fig.gca(projection='3d')
ax.scatter(lx,ly,lz, zdir='z',s=40, c='y',depthshade=True)
ax.scatter(lxi,lyi,lhi, zdir='z',s=40,depthshade=True)

ax.set_xlim3d([0, 1]) ; ax.set_xlabel('X en m')
ax.set_ylim3d([0, 1]) ; ax.set_ylabel('Y en m')
ax.set_zlim3d([0.0, 3]) ; ax.set_zlabel('Z en m')
plt.suptitle("t={}s".format(lt[i]))
for i, j in enumerate(li):
    lzk = mz[i]
    lhk = np.array(mh[i]) + np.array(lzk)
    lxk = mx[i]
    lyk = my[i]
    fig = plt.figure(1)
    plt.clf()
    ax = fig.gca(projection='3d')
    ax.scatter(lxk,lyk,lhk, zdir='z',s=40,depthshade=True)
    ax.scatter(x,y,lz, zdir='z',s=40, c='y',depthshade=True)
    ax.set_xlim3d([0, 1]) ; ax.set_xlabel('X en m')
    ax.set_ylim3d([0, 1]) ; ax.set_ylabel('Y en m')
    ax.set_zlim3d([0, 2]) ; ax.set_zlabel('Z en m')
    plt.suptitle("t={} s".format(np.round(lt[j],2)))
    pause(10**(-6))