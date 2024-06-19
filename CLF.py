import numpy as np
import math
import matplotlib.pylab as plt
from pylab import *
import time
from scipy.integrate import quad


### Approximation de la solution

g = 9.81                                # Constante de pesanteur

vap_min = lambda ug, ud, hg, hd : min((ug-np.sqrt(g*hg)), (ud-np.sqrt(g*hd)))
vap_max  = lambda ug, ud, hg, hd : max((ug+np.sqrt(g*hg)), (ud+np.sqrt(g*hd)))

f2 = lambda h, u : (u**2*h + (g * h**2)/2)

def h(lzb, ls,n):
    return np.array([max(ls[i]-lzb[i+1], 0) for i in range(n)])

def fh(ug, ud, hg, hd, vap_min, vap_max):
    if vap_min > 0:
        return hg*ug
    elif vap_max < 0:
        return hd*ud
    elif vap_max == 0 and vap_min == 0:
        return 0
    else:
        return (vap_max * hg *ug - vap_min * hd * ud + vap_min * vap_max*(hd-hg))/(vap_max - vap_min)

def fq(ug, ud, hg, hd, vap_min, vap_max):
    if vap_min > 0 :
        return f2(hg, ug)
    elif vap_max < 0:
        return f2(hd, ud)
    elif vap_max == 0 and vap_min == 0:
        return 0
    else:
        return (vap_max * f2(hg, ug) - vap_min * f2(hd, ud)+ vap_min * vap_max*(ud*hd-ug*hg))/(vap_max - vap_min)

def s(h, u, zbg, zbd):
    '''Renvoie la valeur du terme source pour l'équation de quantité de mouvement'''
    return -g*h/(2*dx)*(zbd-zbg)

def s2(h, u, zbg, zbd):
    '''Renvoie la valeur du terme source pour l'équation de quantité de mouvement'''
    return -g*h/(2*dx)*(zbd-zbg) - (g*u**2) /400


def sol_appr(tf, n, lhi, lui, lzb, lx) :

    '''Cette fonction renvoie une valeur approchée de la solution jusqu'au temps tf avec un pas de temps dt et un maillage à n volumes'''

    # Initialisation des matrices

    lt = [0]                            # Liste de temps qui vérifient la condition CFL
    lh = [lhi]
    lu = [lui]
    lfh = np.zeros(n+1)                 # Les 1ers et derniers termes de lfh sont toujours nuls
    lfq = np.zeros(n+1)

    while lt[-1] < tf:

        dt = 0.9*dx/max(max([abs(vap_max(lu[-1][i], lu[-1][i+1], lh[-1][i], lh[-1][i+1])) for i in range(n-1)]), max([abs(vap_min(lu[-1][i], lu[-1][i+1], lh[-1][i], lh[-1][i+1])) for i in range(n-1)]))
        lt.append(lt[-1] + dt)

        if len(lt)%50 == 0 :
            print(lt[-1])


        # Calcul des flux à chaque position à l'intérieur
        for j in range(1,n):
            v_min = vap_min(lu[-1][j-1], lu[-1][j], lh[-1][j-1], lh[-1][j])
            v_max = vap_max(lu[-1][j-1], lu[-1][j], lh[-1][j-1], lh[-1][j])
            lfh[j] = fh(lu[-1][j-1], lu[-1][j], lh[-1][j-1], lh[-1][j], v_min, v_max)
            lfq[j] = fq(lu[-1][j-1], lu[-1][j], lh[-1][j-1], lh[-1][j], v_min, v_max)

        lut = [0 for j in range(n)]
        lht = [0 for j in range(n)]

        # Calcul de la solution approchée à l'intérieur du domaine
        for j in range(0,n):
            hj = lh[-1][j] - dt*(lfh[j+1]-lfh[j])/dx
            if(hj>10**(-3)):
                q = lh[-1][j]*lu[-1][j] - dt*(lfq[j+1]-lfq[j])/dx + dt*s(lh[-1][j],lu[-1][j], lzb[j-1], lzb[j+1])
                lut[j] = q/hj
                lht[j] = hj
            else:
                #print('erreur', i, j, hj)
                lut[j] = lu[-1][j]
                lht[j] = 0

        lut[0] = 0
        lut[-1] = 0
        # lut[0] = u0(zeta(lx[0],lt[-1])[0])
        # lut[-1] = u0(zeta(lx[-1],lt[-1])[0])
        # lht[0] = lut[0]**2/(4*g)
        # lht[-1] = lut[-1]**2/(4*g)


        lu.append(lut)
        lh.append(lht)

    return lt, lh, lu

### Bibliothèque de surfaces initiales
def barrage(hg, hd, xl, n, lx):
    lh = [hd for j in range(n)]
    for j in range(n):
        if lx[j] < xl:
            lh[j] = hg
    return np.array(lh)

def bosse(hmax, hmin, xc, L, lx,n):
    return np.array([hmax*np.exp(-(lx[j]-xc)**2/L) + hmin for j in range(n)])

def pente_s(hg, hd, n):
    return np.array([hg + (hd-hg)/(n)*(j+1/2) for j in range(n)])

### Bibliothèque de fonds

def pente(hg, hd, n):
    return np.array([hg + (hd-hg)/(n)*(j+1/2) for j in range(n+2)])


marche_centree = lambda hg, hd : [hg for j in range((n+2)//2)] + [hd for j in range((n+2)//2)]

plat_pente_plat = lambda hg, hd : [hg for j in range((n+2)//3)]+ [hg + (hd-hg)/((n+2)//3)*(j+1/2) for j in range((n+2)//3+1)] + [hd for j in range((n+2)//3)]

#lzb = 0.1*np.ones(n+2)
#lzb =1/992 * (np.arange(-8-dx,8+dx, dx))**4

def pente_angle(hg, a, n):
    return np.array([hg + np.tan(a)*dx*(j+1/2) for j in range(n+2)])

def bosse_VI(hmax, hmin, xc, L, vmax, Lv, lx,n):
    """ Retourne une bosse avec une liste de vitesse initiale où la vitesse est nulle sur la partie horizontale de la surface, linéaire croissante de la base gauche de la bosse jusqu'au sommet où elle atteint vi, puis décroissante du sommet vers la base à droite de la bosse"""
    lhi = [hmax*np.exp(-(lx[j]-xc)**2/L) + hmin for j in range(n)]
    lui = [vmax*np.exp(-(lx[j]-xc)**2/Lv)  for j in range(n)]

    return lhi, lui

### Conditions du tracé

# Paramètres
xmin = -20                                 # Limite gauche du domaine considéré
xmax = 20                                    # Limite droite du domaine considéré
L = xmax-xmin                                # Longueur du domaine considéré
n    = 1000                                  # Nombre de volumes du maillage pour le calcul numérique
dx   = L/n                                   # Largeur des volumes
lx = [xmin+dx/2 + k*dx for k in range(n)]    # Liste des abscisses du barycentre de chaque volume
tf = 2                                     # Temps final
N = 8
a = 0.5                                     # Nombre de graphiques affichés
#lzb = pente_angle(0,a, n)
lzb = np.zeros(n+2)
#Condition initiale

#lsi, lui = bosse_VI(0.3, 1.5,-0, 6, 0, 12, lx, n)
# lsi = pente_s(1,1.5,n)


#lsi = bosse(3, -xmin*tan(a), -10, 4, lx, n)
# lhi = h(lzb, lsi, n)
# lui = 4*np.sqrt(g*lhi)
#lui = np.array([0 for j in range(n)])

lui = [u0(x) for x in lx]
lhi = [j**2/(4*g) for j in lui]
lt, lh, lu = sol_appr(tf, n, lhi, lui, lzb, lx)


## Calcul solution analytique sur le temps et l'espace lt lh

lua = [[u0(zeta(x, t)[0]) for x in lx] for t in lt]
lha = [[lu[j][i]**2/(4*g) for i in range(len(lu[0]))] for j in range(len(lu))]


### Animation
plt.ion()  # interactive on

# ymin = -20*tan(a)+60*tan(a)-5
# ymax = 20*tan(a)+60*tan(a)+5

ymin = 0.96
ymax = 1.07
xminaff = -2
xmaxaff = 40

ymin2 = 0.9
ymax2 = 1.05
xminaff2 = -2
xmaxaff2 = 60


for k in range(int(len(lt)//5)):
    plt.figure(2)
    plt.clf()
    # plt.subplot(121)
    plt.plot(lx, np.array(lzb[1:-1]) + lh[5*k])
    # plt.plot(lx, lha[5*k])
    plt.plot(lx, np.array(lzb[1:-1]), c="k")
    plt.ylim([ymin,ymax])
    plt.xlim([xminaff,xmaxaff])
    plt.ylabel("Hauteur de l'eau")
    plt.title("t={}".format(lt[5*k]*np.sqrt(g)))
    pause(10**(-10))
    # plt.subplot(122)
    # plt.plot(lx, np.array(lzb[1:-1]) + lh[5*k])
    # plt.plot(lx, np.array(lzb[1:-1]), c="k")
    # plt.ylim([ymin2,ymax2])
    # plt.xlim([xminaff2,xmaxaff2])
    # plt.ylabel("Hauteur de l'eau")
    # plt.title("t={}".format(lt[5*k]*sqrt(g)))
    # pause(10**(-10))

##
ymin = 0
ymax = 1.5
xminaff = -20
xmaxaff = 20

plt.figure(8)
plt.clf()
plt.plot(lx, np.array(lzb[1:-1])+lh[50])
plt.ylim([ymin,ymax])
plt.xlim([xminaff,xmaxaff])


### Tests

ymin = -0.5
ymax = 0.5
xminaff = -5
xmaxaff = 0.98


for k in range(int(len(lt)//5)):
    plt.figure(2)
    plt.clf()
    plt.plot(lx, np.array(np.array(lu[5*k]) - 4*np.sqrt(g* np.array(lh[5*k]))))
    plt.ylim([ymin,ymax])
    plt.xlim([xminaff,xmaxaff])
    plt.ylabel("Hauteur de l'eau")
    plt.title("t={}".format(lt[5*k]))
    pause(10**(-14))

## Enregistrement séparé des images pour créer l'animation sous beamer

plt.ioff()

fpt=30
ymin = 0
ymax = 0.4
xminaff = -20
xmaxaff = 20

# tf = 8
plt.figure(0)
plt.clf()
lk = np.arange(0,8, 1/(fpt*tf))

i = 0
# i1 = 0
# i2 = 0
#
# plt.figure(1)
# plt.plot(lx, lzb[1:-1], c='blue')
# plt.plot(lx, lzb1[1:-1], c='red')
# plt.plot(lx, lzb2[1:-1], c='green')
# plt.show()

for k, t in enumerate(lk):
    while lt[i]<=t:
        i += 1
    # while lt1[i1]<=t:
    #     i1 += 1
    # while lt2[i2]<=t:
    #     i2 += 1

    plt.figure(0)
    plt.clf()
    plt.plot(lx, np.array(lzb[1:-1]) + lh[i], c='blue',label='numérique')
    plt.plot(lx, lha[i], c='red', label='analytique')
    # plt.plot(lx, np.array(lzb1[1:-1]) + lh1[i1], c='red',label='n=5')
    # plt.plot(lx, np.array(lzb2[1:-1]) + lh2[i2], c='green',label='n=10')
    plt.plot(lx, lzb[1:-1], c='black')
    # plt.plot(lx, lzb1[1:-1], c='red')
    # plt.plot(lx, lzb2[1:-1], c='green')
    plt.legend(title='solution')
    # plt.plot(lx, np.array(lzb) + lhc[ic], c='red', label='horizontale')
    plt.plot(lx, np.array(lzb[1:-1]), c="k")
    plt.ylim([ymin, ymax])
    plt.xlim([xminaff,xmaxaff])
    plt.ylabel("Hauteur de l'eau")
    plt.title("t={}".format(t))
#
#     plt.suptitle("Alpha="+str(a)+", vitesse crete="+str(vi))
    plt.savefig(r'C:\Users\lbs\Documents\006- ENS\A1\Stage\Illustrations\Animations\compsola\c{}.png'.format(k))
    plt.close()

## Comparaison au benchmark

#fonctions

from scipy.integrate import quad

gamma = lambda H, d : np.sqrt(3*(H/d)/4)
sech =  lambda x : 2/(np.exp(x)+np.exp(-x))
arccosh = lambda x : np.log(x+np.sqrt(x**2-1))
eta = lambda x, H, Xs,d : H*sech(gamma(H,d)*(x-Xs))**2
L = lambda H : 1/gamma(H, d) *arccosh(np.sqrt(20))
v = lambda h, x : (-np.sqrt(g*h)/2)*sech(np.sqrt(np.sqrt(g*h))*x/2)**2

def pente_plat(lx, X0, beta):
    lzb =  np.zeros(n+2)
    for i,X in enumerate(lx):
        x= X-dx/2
        if x<X0:
            lzb[i]= -np.tan(beta)*(x-X0)
        if x>= X0:
            lzb[i]=0
    return lzb

# a = quad(f, -1, 1)[0]

def G(x):
    return quad(f, -1, x)[0]/a

def f(x):
    if x <= X0:
        return -np.tan(beta)*(x-X0)
    else:
        return 0

def psi(x):
    if x < 1 and x > -1:
        return np.exp(-1/(1-x**2))
    else:
        return 0

c = quad(psi, - np.inf, np.inf)[0]

def noyau(x, n):
    return n*1/c*psi(n*x)

def convolution(x, nc):
    def h(w):
        return noyau(w, nc)*f(x-w)
    return quad(h, - nc, nc)[0]

def pente_plat_Cinf(lx, X0, beta, nc):
    lzb =  np.zeros(n+2)
    for i,X in enumerate(lx):
        x = X-dx/2
        lzb[i+1]= convolution(x, nc)
    lzb[0] = convolution(xmin-dx/2, nc)
    lzb[-1] = convolution(xmax + dx/2, nc)
    return lzb

eps2 = 0.1



#Paramètres

X0 = 19.85
beta = 0.05033528012620885  #en radians
Hsd = 0.0185
d = np.tan(beta)*X0
H = Hsd*d
Xs = X0+L(H)


g=9.81
xmin = -6                               # Limite gauche du domaine considéré
xmax = 140                                # Limite droite du domaine considéré
L = xmax-xmin                                # Longueur du domaine considéré
n    = 10000                                  # Nombre de volumes du maillage pour le calcul numérique
dx   = L/n                                   # Largeur des volumes
lx = [xmin+dx/2 + k*dx for k in range(n)]

lx = lx+lx1+lx2+lx3+lx4+lx5
lx = sorted(set(lx))




##
n=len(lx)

letainit = np.array([eta(x, H, Xs,d) for x in lx]) *d
lsinit = letainit + d*np.ones(n)

# L2 = np.sqrt(4*d**3/(H*g))
# lhi = d+H*sech((lx-Xs)/L2)**2

# lzb= np.zeros(n+2)

# lui = np.sqrt(g*lhi)


nc= 10
lzb = pente_plat_Cinf(lx, X0, beta, nc)
lhi = h(lzb, lsinit, n)
# lh0 = np.ones(n)-np.array(lzb[1:-1])

# lui = [v(lh0[j],lx[j]) for j in range(n)]
# lui = - letainit/3

c = -np.sqrt(g*(d))

lui = np.zeros(n)
for k,eta in enumerate(letainit):
    if eta <= 10**(-4):

        lui[k] = 0
    else:
        lui[k] = c*(1-(d/(1+eta)))

tf = 26.5
start = time.time()
lt, lh, lu = sol_appr(tf, n, lhi, lui, lzb, lx)
end = time.time()
print('temps de calcul:')
print(end-start)

# nc1 = 5
# lzb1 = pente_plat_Cinf(lx, X0, beta, nc1)
# lhi = h(lzb1, lsinit, n)
# start = time.time()
# lt1, lh1, lu1 = sol_appr(tf, n, lhi, lui, lzb1, lx)
# end = time.time()
# print('temps de calcul:')
# print(end-start)
#
# nc2= 10
# lzb2 = pente_plat_Cinf(lx, X0, beta, nc2)
# lhi = h(lzb2, lsinit, n)
# start = time.time()
# lt2, lh2, lu2 = sol_appr(tf, n, lhi, lui, lzb2, lx)
# end = time.time()
# print('temps de calcul:')
# print(end-start)






# def pente_plat_C1(lx, X0, beta, eps):
#     lzb =  np.zeros(n+2)
#
#     for i,x in enumerate(lx):
#         if x<X0-eps:
#             lzb[i]= -np.tan(beta)*(x-X0)
#         if x>= X0-eps and x<= X0+eps:
#             lzb[i]= -tan(beta)*(x-X0)*g((x/eps-X0))
#         else:
#             lzb[i] = 0
#     return lzb

# b = lambda eps, X0, beta :  np.tan(beta) / (4*eps)
# c = lambda eps, X0, beta : np.tan(beta)*(-eps-X0)/(2*eps)
# d = lambda eps, X0, beta : np.tan(beta) * (eps**2 + 2 * eps * X0 + X0**2)/(4 * eps)
#
#
# def pente_plat_C1(lx, X0, beta, eps):
#
#     lzb =  np.zeros(n+2)
#     for i,X in enumerate(lx):
#         x= X-dx/2
#         if x<X0-eps:
#             lzb[i+1]= -np.tan(beta)*(x-X0)
#         elif x>= X0-eps and x<= X0+eps:
#             lzb[i+1]= b(eps, X0, beta)*x**2 + c(eps, X0, beta)*x + d(eps, X0, beta)
#         else:
#             lzb[i+1] = 0
#     lzb[0] = -np.tan(beta)*(xmin-dx/2-X0)
#     lzb[-1] = 0
#     return lzb




# # lzb = pente_plat_C1(lx, X0, beta, eps)
# plt.figure(0)
# plt.clf()
# # plt.plot(lx, lzb[1:-1])

# lx2 = [xmin-dx/2 + k*dx for k in range(n+2)]
# lzb = np.array([-tan(beta)*(x-X0)*G(-(x/eps-X0)) for x in lx2])
# lzb = np.zeros(n+2)
# lzb = pente_plat_C1(lx, X0, 0.05, 4)
# lzb = pente_plat_C1(lx, X0, beta, eps)
# plt.figure(0)
# plt.clf()
# # ly = [b(eps, X0, beta)*x**2 + c(eps, X0, beta)*x + d(eps, X0, beta) for x in lx2]
# # plt.plot(lx2, ly)
# plt.plot(lx2, lzb)
# plt.ylim([-1,1])




#lzb = pente_plat_Cinf(lx, X0, beta, eps2)
# plt.figure(0)
# plt.clf()
# plt.plot(lx, lzb[1:-1])
# plt.show()
#

##

ymin2 = 1-0.03
ymax2 = 1 + 0.06
xminaff2 = -1.5
xmaxaff2 = 20
i = 0
plt.figure(3)
plt.clf()
plt.suptitle('Vitesse initiale soliton, convolution n=10, dx=0.015')
k=30
while lt[i]*sqrt(g)< k :
    i += 1
plt.subplot(231)
plt.plot(lx, np.array(lzb[1:-1]+lh[i]))
plt.plot(lx, np.array(lzb[1:-1]), 'k')
plt.scatter(lx1, np.array(la1)+1, s=10, color='red')
plt.title("t={}".format(np.round(lt[i]*sqrt(g), 1)))
plt.ylim([ymin2,ymax2])
plt.xlim([xminaff2,xmaxaff2])

k=40
while lt[i]*sqrt(g)< k :
    i += 1
plt.subplot(232)
plt.plot(lx, np.array(lzb[1:-1]+lh[i]))
plt.plot(lx, np.array(lzb[1:-1]), 'k')
plt.scatter(lx2, np.array(la2)+1, s=10, color='red')
plt.title("t={}".format(np.round(lt[i]*sqrt(g), 1)))
plt.ylim([ymin2,ymax2])
plt.xlim([xminaff2,xmaxaff2])

k=50
while lt[i]*sqrt(g)< k :
    i += 1
plt.subplot(233)
plt.plot(lx, np.array(lzb[1:-1]+lh[i]))
plt.plot(lx, np.array(lzb[1:-1]), 'k')
plt.scatter(lx3, np.array(la3)+1, s=10, color='red')
plt.title("t={}".format(np.round(lt[i]*sqrt(g), 1)))
plt.ylim([ymin2,ymax2])
plt.xlim([xminaff2,xmaxaff2])

k=60
while lt[i]*sqrt(g)< k :
    i += 1
plt.subplot(234)
plt.plot(lx, np.array(lzb[1:-1]+lh[i]))
plt.plot(lx, np.array(lzb[1:-1]), 'k')
plt.scatter(lx4, np.array(la4)+1, s=10, color='red')
plt.title("t={}".format(np.round(lt[i]*sqrt(g), 1)))
plt.ylim([ymin2,ymax2])
plt.xlim([xminaff2,xmaxaff2])

k=70
while lt[i]*sqrt(g)< k :
    i += 1
plt.subplot(235)
plt.plot(lx, np.array(lzb[1:-1]+lh[i]))
plt.plot(lx, np.array(lzb[1:-1]), 'k')
plt.scatter(lx5, np.array(la5)+1, s=10, color='red')
plt.title("t={}".format(np.round(lt[i]*sqrt(g), 1)))
plt.ylim([ymin2,ymax2])
plt.xlim([xminaff2,xmaxaff2])



##

ymin2 = 1-0.03
ymax2 = 1 + 0.3
xminaff2 = -1.5
xmaxaff2 = 20
i = 0
plt.figure(3)
plt.clf()
# plt.suptitle('Vitesse initiale u=sqrt(g*eta) et frottements')
k=15
while lt[i]*sqrt(g)< k :
    i += 1
plt.subplot(221)
plt.plot(lx, np.array(lzb[1:-1]+lh[i]))
plt.scatter(lx1b, np.array(la1b)+1)
plt.title("t={}".format(np.round(lt[i]*sqrt(g), 2)))
plt.ylim([ymin2,ymax2])
plt.xlim([xminaff2,xmaxaff2])

k=20
while lt[i]*sqrt(g)< k :
    i += 1
plt.subplot(222)
plt.plot(lx, np.array(lzb[1:-1]+lh[i]))
plt.scatter(lx2b, np.array(la2b)+1)
plt.title("t={}".format(np.round(lt[i]*sqrt(g), 2)))
plt.ylim([ymin2,ymax2])
plt.xlim([xminaff2,xmaxaff2])

k=25
while lt[i]*sqrt(g)< k :
    i += 1
plt.subplot(223)
plt.plot(lx, np.array(lzb[1:-1]+lh[i]))
plt.scatter(lx3b, np.array(la3b)+1)
plt.title("t={}".format(np.round(lt[i]*sqrt(g), 2)))
plt.ylim([ymin2,ymax2])
plt.xlim([xminaff2,xmaxaff2])

k=30
while lt[i]*sqrt(g)< k :
    i += 1
plt.subplot(224)
plt.plot(lx, np.array(lzb[1:-1]+lh[i]))
plt.scatter(lx4b, np.array(la4b)+1)
plt.title("t={}".format(np.round(lt[i]*sqrt(g), 2)))
plt.ylim([ymin2,ymax2])
plt.xlim([xminaff2,xmaxaff2])

# k=70
# while lt[i]*sqrt(g)< k :
#     i += 1
# plt.subplot(235)
# plt.plot(lx, np.array(lzb[1:-1]+lh[i]))
# plt.scatter(lx5, np.array(la5)+1)
# plt.title("t={}".format(np.round(lt[i]*sqrt(g), 2)))
# plt.ylim([ymin2,ymax2])
# plt.xlim([xminaff2,xmaxaff2])

##

import csv

donnees = lt, lh, lu

chemin_fichier = r'C:\Users\lbs\Documents\006- ENS\A1\Stage\Programmes\données enregistrées\soliton.csv'
with open(chemin_fichier, mode='w', newline='') as fichier_csv:
    writer = csv.writer(fichier_csv, delimiter=',')
    writer.writerows(donnees)

print("Données enregistrées avec succès dans", chemin_fichier)

### Open graphs

# Chemin vers le fichier CSV contenant les données
chemin_fichier = r'C:\Users\lbs\Documents\006- ENS\A1\Cours\Système_Complexes\Données enregistrées\3.csv'

# Liste pour stocker les données chargées
donnees_chargees = []

# Chargement des données depuis le fichier CSV
with open(chemin_fichier, mode='r') as fichier_csv:
    reader = csv.reader(fichier_csv, delimiter=',')
    for ligne in reader:
        donnees_chargees.append(ligne)

str_T = donnees_chargees[0]
str_count_susceptible = donnees_chargees[1]
str_count_infectious = donnees_chargees[2]
str_count_recovered = donnees_chargees[3]
