import numpy as np
import math
import matplotlib.pylab as plt
from pylab import *
plt.ion()

### FONCTIONS POUR LES CONDITIONS INITIALES

def h(lzb, ls,n):
    '''Cette fonction permet de calculer les hauteurs effectives dans le cas où la surface initiale est en-dessous du fond'''
    return [max(ls[i]-lzb[i], 0) for i in range(n)]

def barrage(hg, hd, xl, n):
    '''Cette fonction renvoie une liste de taille n qui discrétise un créneau de hauteur à gauche (resp droite) hg (resp hd) au niveau des points de lx, dont la discontinuité se situe en xl'''
    lh = [hd for j in range(n)]
    for j in range(n):
        if lx[j] < xl:
            lh[j] = hg
    return np.array(lh)

def bosse(hmax, hmin, xc, L, n):
    '''Cette fonction renvoie une liste de taille n qui discrétise une bosse de hauteur maximale (resp minimale) hmax (resp hmin) au niveau des points de lx, dont le centre de la bosse se situe en xc'''
    return [hmax*np.exp(-(lx[j]-xc)**2/L) + hmin for j in range(n)]

def pente(hg, a, n):
    return np.array([hg + np.tan(a)*dx*(j+1/2) for j in range(n)])

### CONDITIONS DE TRACE A MODIFIER

# Paramètres
xmin = -15                                   # Limite gauche du domaine considéré
xmax = 30                                    # Limite droite du domaine considéré
n = 2500                                     # Nombre de volumes du maillage pour le calcul numérique
tf = 3                                       # Temps final
d = 15                                       # Angle de la pente en degré


# /!\ Ne pas modifier les quatre lignes qui suivent
L = xmax-xmin                                # Longueur du domaine considéré
dx   = L/n                                   # Largeur des volumes
lx = [xmin+dx/2 + k*dx for k in range(n)]    # Liste des abscisses du barycentre de chaque volume
a = d*pi/180                                 # Angle de la pente en radians


lsi = pente(0.5, a, n)                       # Surface d'eau initiale (on considèrera par la suite que si la surface de l'eau est en dessous du sol, alors il n'y a pas d'eau)
lui = [0]*n                                  # Vitesse initile
xminaff = 0                                  # Limite gauche du domaine d'affichage (doit être supérieure à xmin)
xmaxaff = 30                                 # Limite droite du domaine d'affichage (doit être inférieure à xmax)

## Calcul de la solution

lzbp = pente(0,a,n)                          # Fond
lhi = h(lzbp, lsi, n)                        # Hauteur initiale

g = 9.81                                     # Constante de pesanteur

def vap_minp(ug, ud, hg, hd):
    '''Renvoie la valeur propre minimale au niveau d'une arête étant donné les valeurs de u et h des mailles ajdacentes'''
    return np.cos(a)*min((ug-np.sqrt(g*hg)*np.cos(a)), (ud-np.sqrt(g*hd)*np.cos(a)))

def vap_maxp(ug, ud, hg, hd):
    '''Renvoie la valeur propre maximale au niveau d'une arête étant donné les valeurs de u et h des mailles ajdacentes'''
    return np.cos(a)*max((ug+np.sqrt(g*hg)*np.cos(a)), (ud+np.sqrt(g*hd)*cos(a)))

def f2p(h, u):
    '''Renvoie la deuxième coordonnée de F'''
    return np.cos(a)*(u**2*h + np.cos(a)**2*(g * h**2)/2)

def fhp(ug, ud, hg, hd, vap_min, vap_max):
    '''Renvoie le flux HLL de hauteur numérique'''
    if vap_min > 0:
        return np.cos(a)*hg*ug
    elif vap_max < 0:
        return np.cos(a)*hd*ud
    elif vap_max == 0 and vap_min == 0:
        return 0
    else:
        return (np.cos(a)*vap_max * hg *ug - np.cos(a)*vap_min * hd * ud + vap_min * vap_max*(hd-hg))/(vap_max - vap_min)


def fqp(ug, ud, hg, hd, vap_min, vap_max):
    '''Renvoie le flux HLL de quantité de mouvement numérique'''
    if vap_min > 0 :
        return f2p(hg, ug)
    elif vap_max < 0:
        return f2p(hd, ud)
    elif vap_max == 0 and vap_min == 0:
        return 0
    else:
        return (vap_max * f2p(hg, ug) - vap_min * f2p(hd, ud)+ vap_min * vap_max*(ud*hd-ug*hg))/(vap_max - vap_min)

def sp(h):
    '''Renvoie la valeur du terme source pour l'équation de quantité de mouvement'''
    return -g*h*np.sin(a)

def usecp(j, lh, lu):
    '''Calcule la vitesse de leau de la cellule j nouvellement remplie'''
    if j == 0 or j == n-1:          # Cas des bords
        return 0
    u = 0
    l = 0
    if lh[-1][j+1] > 0 and lu[-1][j+1] > 0:
        if lu[-1][j+1]**2 + 2*g*(lzbp[j+1] + lh[-1][j+1]/2 - lzbp[j] - lh[-1][j]/2) < 0:
            c = 0
        else:
            c = np.sqrt(lu[-1][j+1]**2 + 2*g*(lzbp[j+1] + lh[-1][j+1]/2 - lzbp[j] - lh[-1][j]/2))/lu[-1][j+1]
        u += c*lu[-1][j+1]
        l += 1
    if lh[-1][j-1] > 0 and lu[-1][j-1] > 0:
        if lu[-1][j-1]**2 + 2*g*(lzbp[j-1] + lh[-1][j-1]/2 - lzbp[j] - lh[-1][j]/2) < 0:
            c = 0
        else:
            c = np.sqrt(lu[-1][j-1]**2 + 2*g*(lzbp[j-1] + lh[-1][j-1]/2 - lzbp[j] - lh[-1][j]/2))/lu[-1][j-1]
        u += c*lu[-1][j-1]
        l += 1
    if l != 0:
        u /= l
    return u


def sol_plage(tf, n, lhi, lui, lzb, lx) :

    '''Cette fonction renvoie une valeur approchée de la solution jusqu'au temps tf avec un pas de temps variable et un maillage à n volumes'''

    # Initialisation des matrices

    lt = [0]                            # Liste de temps qui vérifient la condition CFL
    lh = [lhi]
    lu = [lui]
    lfh = np.zeros(n+1)                 # Les 1ers et derniers termes de lfh sont toujours nuls
    lfq = np.zeros(n+1)

    while lt[-1] < tf:
        if len(lt)%30 == 0:
            print('Solution calculée jusque t =', np.round(lt[-1], 2))

        dt = 0.9*dx/max(max([abs(vap_maxp(lu[-1][i], lu[-1][i+1], lh[-1][i], lh[-1][i+1])) for i in range(n-1)]), max([abs(vap_minp(lu[-1][i], lu[-1][i+1], lh[-1][i], lh[-1][i+1])) for i in range(n-1)]))
        lt.append(lt[-1] + dt)

        # Calcul des flux à chaque position à l'intérieur
        for j in range(1,n):
            v_min = vap_minp(lu[-1][j-1], lu[-1][j], lh[-1][j-1], lh[-1][j])
            v_max = vap_maxp(lu[-1][j-1], lu[-1][j], lh[-1][j-1], lh[-1][j])
            lfh[j] = fhp(lu[-1][j-1], lu[-1][j], lh[-1][j-1], lh[-1][j], v_min, v_max)
            lfq[j] = fqp(lu[-1][j-1], lu[-1][j], lh[-1][j-1], lh[-1][j], v_min, v_max)

        lut = [0 for j in range(n)]
        lht = [0 for j in range(n)]


        # Calcul de la solution approchée à l'intérieur du domaine
        for j in range(0,n):
            hj = lh[-1][j] - dt*(lfh[j+1]-lfh[j])/dx
            if hj>10**-3:
                q = lh[-1][j]*lu[-1][j] - dt*(lfq[j+1]-lfq[j])/dx + dt*sp(lh[-1][j])
                lut[j] = q/hj
                lht[j] = hj
            else:
                lut[j] = usecp(j, lh, lu)
                lht[j] = 0

        lut[0] = 0
        lut[-1] = 0

        lu.append(lut)
        lh.append(lht)

    return lt, lh, lu

ltp, lhp, lup = sol_plage(tf, n, lhi, lui, lzbp, lx)


## ANIMATION
ymin = tan(a)*xminaff-xmin*tan(a) - 1
ymax = tan(a)*xmaxaff-xmin*tan(a) +1

plt.ion()

fpt = 50
plt.figure(0)
plt.clf()
lk = np.arange(0,tf, 1/fpt)
ip = 0
for t in lk:
    while ltp[ip]<=t :
        ip += 1

    plt.figure(0)
    plt.clf()
    plt.plot(lx, np.array(lzbp) + lhp[ip], c='blue', label='selon la pente')
    plt.plot(lx, np.array(lzbp), c="k")
    plt.xlim([xminaff,xmaxaff])
    plt.ylabel("Hauteur de l'eau en m")
    plt.xlabel('x en m')
    plt.ylim([ymin, ymax])
    plt.title("t={} sec".format(format(t, '.2g')))
    plt.pause(10**(-10))

