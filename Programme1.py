import numpy as np
import math
import matplotlib.pylab as plt
from pylab import *
plt.ion()

### FONCTIONS POUR LES CONDITIONS INITIALES
def barrage(hg, hd, xl, n, lx):
    lh = [hd for j in range(n)]
    for j in range(n):
        if lx[j] < xl:
            lh[j] = hg
    return np.array(lh)

def bosse(hmax, hmin, xc, L, lx,n):
    return [hmax*np.exp(-(lx[j]-xc)**2/L) + hmin for j in range(n)]

### FONCTION POUR LE PROFIL DU FOND
def pente(hg, hd, n):
    return np.array([hg + (hd-hg)/(n)*(j+1/2) for j in range(n+2)])


### A MODIFIER
# Domaine d'espace et de temps
xmin = - 50                                  # Limite gauche du domaine considéré
xmax = 40                                    # Limite droite du domaine considéré
tf = 10                                       # Temps final
n = 1000                                     # Nombre de volumes du maillage pour le calcul numérique

L = xmax-xmin                                # Longueur du domaine considéré
dx   = L/n                                   # Largeur des volumes
lx = [xmin+dx/2 + k*dx for k in range(n)]    # Liste des abscisses du barycentre de chaque volume


# Condition initiale et fond, possibilité de les choisir dans les fonctions déjà écrites ci-dessus
# lsi = barrage(2,1,0,n,lx)
lsi = bosse(0.3,1.6,-15,6,lx, n)             #surface initiale
# lui = np.array([0 for j in range(n)])
lui = bosse(1,0,-15,9,lx, n)                 #vitesse initiale
lzb = pente(0, 3,n)                          #profil du fond




###CALCUL DE LA SOLUTION

g = 9.81                                # Constante de pesanteur

def vap_min(ug, ud, hg, hd):
    '''Retourne la valeur propre minimale de la jacobienne'''
    return min((ug-np.sqrt(g*hg)), (ud-np.sqrt(g*hd)))

def vap_max(ug, ud, hg, hd):
    '''Retourne la valeur propre maximale de la jacobienne'''
    return max((ug+np.sqrt(g*hg)), (ud+np.sqrt(g*hd)))

def f2(h, u):
    '''Retourne la deuxième composante du vecteur F(U)'''
    return u**2*h + (g * h**2)/2

def h(lzb, ls,n):
    '''Renvoie les hauteurs d'eau à partir du graphe de la surface et du graphe du fond'''
    return [max(ls[i]-lzb[i+1], 0) for i in range(n)]    # Rem : La surface de leau peut être en dessous du fond, dans ce cas on consière que la hauteur deau est nulle

def fh(ug, ud, hg, hd, v_min, v_max):
    '''Renvoie le flux numérique (HLL) de hauteur'''
    if v_min > 0:
        return hg*ug
    elif v_max < 0:
        return hd*ud
    elif v_max == 0 and v_min == 0:
        return 0
    else:
        return (v_max * hg *ug - v_min * hd * ud + v_min * v_max*(hd-hg))/(v_max - v_min)

def fq(ug, ud, hg, hd, v_min, v_max):
    '''Renvoie le flux numérique (HLL) de quantité de mouvement'''
    if v_min > 0 :
        return f2(hg, ug)
    elif v_max < 0:
        return f2(hd, ud)
    elif v_max == 0 and v_min == 0:
        return 0
    else:
        return (v_max * f2(hg, ug) - v_min * f2(hd, ud)+ v_min * v_max*(ud*hd-ug*hg))/(v_max - v_min)

def s(h, zbg, zbd):
    '''Renvoie la valeur du terme source pour l'équation de quantité de mouvement'''
    return -g*h/(2*dx)*(zbd-zbg)

def usec(j, lh, lu):
    '''Calcule la vitesse de leau de la cellule j nouvellement remplie'''
    if j == 0 or j == n-1:          # Cas des bords
        return 0
    u = 0
    l = 0
    if lh[-1][j+1] > 0 and lu[-1][j+1] > 0:
        if lu[-1][j+1]**2 + 2*g*(lzb[j+1] + lh[-1][j+1]/2 - lzb[j] - lh[-1][j]/2) < 0:
            c = 0
        else:
            c = np.sqrt(lu[-1][j+1]**2 + 2*g*(lzb[j+1] + lh[-1][j+1]/2 - lzb[j] - lh[-1][j]/2))/lu[-1][j+1]
        u += c*lu[-1][j+1]
        l += 1
    if lh[-1][j-1] > 0 and lu[-1][j-1] > 0:
        if lu[-1][j-1]**2 + 2*g*(lzb[j-1] + lh[-1][j-1]/2 - lzb[j] - lh[-1][j]/2) < 0:
            c = 0
        else:
            c = np.sqrt(lu[-1][j-1]**2 + 2*g*(lzb[j-1] + lh[-1][j-1]/2 - lzb[j] - lh[-1][j]/2))/lu[-1][j-1]
        u += c*lu[-1][j-1]
        l += 1
    if l != 0:
        u /= l
    return u


def sol_appr(tf, n, lhi, lui, lzb, lx) :

    '''Cette fonction renvoie une valeur approchée de la solution jusqu'au temps tf avec un pas de temps dt et un maillage à n volumes'''

    # Initialisation des matrices

    lt = [0]                            # Initialisation de la liste de temps qui vérifiera la condition CFL
    lh = [lhi]                          # Initialisation de la matrice des hauteurs, chaque lignes correspond au temps de même indice de la liste lt
    lu = [lui]                          # Initialisation de la  matrice des vitesses, chaque lignes correspond au temps de même indice de la liste lt
    lfh = np.zeros(n+1)                 # Création de la liste des flux de hauteur pour le temps considéré
    lfq = np.zeros(n+1)                 # Création de la liste des flux de quantité de mouevement pour le temps considéré

    while lt[-1] < tf:

        dt = 0.9*dx/max(max([abs(vap_max(lu[-1][i], lu[-1][i+1], lh[-1][i], lh[-1][i+1])) for i in range(n-1)]), max([abs(vap_min(lu[-1][i], lu[-1][i+1], lh[-1][i], lh[-1][i+1])) for i in range(n-1) if lh[-1][i] != 0]))         # Calcul du pas de temps basé sur la condition CFL
        lt.append(lt[-1] + dt)          # Ajout du prochain temps dans la liste des temps

        # Calcul des flux à chaque extrémité des segaments
        for j in range(1,n):
            v_min = vap_min(lu[-1][j-1], lu[-1][j], lh[-1][j-1], lh[-1][j])
            v_max = vap_max(lu[-1][j-1], lu[-1][j], lh[-1][j-1], lh[-1][j])
            lfh[j] = fh(lu[-1][j-1], lu[-1][j], lh[-1][j-1], lh[-1][j], v_min, v_max)
            lfq[j] = fq(lu[-1][j-1], lu[-1][j], lh[-1][j-1], lh[-1][j], v_min, v_max)
        # Rem : Le fait de garder lfh[0] = lfh[-1] = 0 permet de respecter la condition de bord concernant la hauteur

        # Création de listes temporaires pour calculer la hauteur et la vitesse
        lut = [0 for j in range(n)]
        lht = [0 for j in range(n)]

        # Calcul de la hauteur et de la vitesse à l'intérieur du domaine
        for j in range(0,n):
            hj = lh[-1][j] - dt*(lfh[j+1]-lfh[j])/dx
            if hj>10**(-5) and lh[-1][j] > 10**(-5):
                q = lh[-1][j]*lu[-1][j] - dt*(lfq[j+1]-lfq[j])/dx + dt*s(lh[-1][j], lzb[j-1], lzb[j+1])
                lut[j] = q/hj
                lht[j] = hj
            elif hj <= 10**(-5):               # Cas des zones sèches
                lut[j] = 0
                lht[j] = 0
            else:
                lht[j] = hj
                lut[j] = usec(j, lh, lu)

        # Conditions de bord pour la vitesse
        lut[0] = 0
        lut[-1] = 0

        lu.append(lut)
        lh.append(lht)

    return lt, lh, lu



lhi = h(lzb, lsi, n)
lt, lh, lu = sol_appr(tf, n, lhi, lui, lzb, lx)

### ANIMATION

i=0
fpt=20
lk = np.arange(0,tf, 1/(fpt*tf))
hmax = np.max(np.array(np.max(l+np.array(lzb[1:-1])) for l in lh))

for t in lk:
    while lt[i]<t:
        i+=1
    plt.figure(2)
    plt.clf()
    plt.plot(lx, np.array(lzb[1:-1]) + lh[i])
    plt.plot(lx, np.array(lzb[1:-1]), c="k")
    plt.ylim([0,3])
    plt.xlim([xmin,xmax])
    plt.ylabel("Hauteur de l'eau")
    plt.title("t={}".format(np.round(lt[i], 2)))
    pause(10**(-14))


### Extraction de N temps
plt.figure(2)
plt.clf()
N=6
lk = [k*tf /(N-1) for k in range(N)]
i=0
for j, t in enumerate(lk):
    while lt[i]<t:
        i+=1
    plt.subplot(N//2 + 1, 2, j+1)
    plt.plot(lx, lzb[1:-1] + np.array(lh[i]), c="b")
    plt.plot(lx, lzb[1:-1], c="k")
    plt.ylim([1,2.5])
    plt.xlim([-20,15])
    plt.ylabel("Hauteur de l'eau")
    plt.title("t={}".format(np.round(lt[i],1)))



