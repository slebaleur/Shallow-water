import numpy as np
from math import *
import matplotlib.pylab as plt
from pylab import *
from tkinter import *
from matplotlib.animation import FuncAnimation
from ttkthemes import ThemedTk


### Approximation de la solution

g = 9.81 # Constante de pesanteur

vap_min = lambda ug, ud, hg, hd: min((ug-np.sqrt(g*hg)), (ud-np.sqrt(g*hd))) #plus petite valeur propre
vap_max = lambda ug, ud, hg, hd: max((ug+np.sqrt(g*hg)), (ud+np.sqrt(g*hd)))#plus grfande valeur propre

f2 = lambda h, u: (u**2*h + (g * h**2)/2) #deuxieme composante du flux

def h(lzb, ls, n):
    return [max(ls[i]-lzb[i+1], 0) for i in range(n)]

def fh(ug, ud, hg, hd, vap_min, vap_max):#calcul du flux pour h
    if vap_min > 0:
        return hg*ug
    elif vap_max < 0:
        return hd*ud
    elif vap_max == 0 and vap_min == 0:
        return 0
    else:
        return (vap_max * hg *ug - vap_min * hd * ud + vap_min * vap_max*(hd-hg))/(vap_max - vap_min)

def fq(ug, ud, hg, hd, vap_min, vap_max):#calcul du flux pour q
    if vap_min > 0:
        return f2(hg, ug)
    elif vap_max < 0:
        return f2(hd, ud)
    elif vap_max == 0 and vap_min == 0:
        return 0
    else:
        return (vap_max * f2(hg, ug) - vap_min * f2(hd, ud)+ vap_min * vap_max*(ud*hd-ug*hg))/(vap_max - vap_min)

def s(h, zbg, zbd):
    '''Renvoie la valeur du terme source pour l'équation de quantité de mouvement'''
    return -g*h/(2*dx)*(zbd-zbg)

def sol_appr(tf, n, lhi, lui, lzb, lx):
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

        # Calcul des flux à chaque position à l'intérieur
        for j in range(1, n):
            v_min = vap_min(lu[-1][j-1], lu[-1][j], lh[-1][j-1], lh[-1][j])
            v_max = vap_max(lu[-1][j-1], lu[-1][j], lh[-1][j-1], lh[-1][j])
            lfh[j] = fh(lu[-1][j-1], lu[-1][j], lh[-1][j-1], lh[-1][j], v_min, v_max)
            lfq[j] = fq(lu[-1][j-1], lu[-1][j], lh[-1][j-1], lh[-1][j], v_min, v_max)

        lut = [0 for j in range(n)]
        lht = [0 for j in range(n)]

        # Calcul de la solution approchée à l'intérieur du domaine
        for j in range(0, n):
            hj = lh[-1][j] - dt*(lfh[j+1]-lfh[j])/dx
            if(hj > 10**(-3)):
                q = lh[-1][j]*lu[-1][j] - dt*(lfq[j+1]-lfq[j])/dx + dt*s(lh[-1][j], lzb[j-1], lzb[j+1])
                lut[j] = q/hj
                lht[j] = hj
            else:
                lut[j] = lu[-1][j]
                lht[j] = 0

        lut[0] = 0
        lut[-1] = 0

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

def bosse(hmax, hmin, xc, L, lx, n):
    return [hmax*np.exp(-(lx[j]-xc)**2/L) + hmin for j in range(n)]

### Bibliothèque de fonds
def pente(hg, hd, n):
    return np.array([hg + (hd-hg)/(n)*(j+1/2) for j in range(n+2)])

marche_centree = lambda hg, hd: [hg for j in range((n+2)//2)] + [hd for j in range((n+2)//2)]

plat_pente_plat = lambda hg, hd: [hg for j in range((n+2)//3)] + [hg + (hd-hg)/((n+2)//3)*(j+1/2) for j in range((n+2)//3+1)] + [hd for j in range((n+2)//3)]

def pente_angle(hg, a, n):
    return np.array([hg + np.tan(a)*dx*(j+1/2) for j in range(n+2)])

def bosse_VI(hmax, hmin, xc, L, vmax, Lv, lx, n):
    """ Retourne une bosse avec une liste de vitesse initiale où la vitesse est nulle sur la partie horizontale de la surface, linéaire croissante de la base gauche de la bosse jusqu'au sommet où elle atteint vi, puis décroissante du sommet vers la base à droite de la bosse"""
    lhi = [hmax*np.exp(-(lx[j]-xc)**2/L) + hmin for j in range(n)]
    lui = [vmax*np.exp(-(lx[j]-xc)**2/Lv)  for j in range(n)]
    return lhi, lui

### Conditions du tracé

# Paramètres
xmin = -20                                 # Limite gauche du domaine considéré
xmax = 20                                  # Limite droite du domaine considéré
ymin = 0
ymax = 10
L = xmax - xmin                            # Longueur du domaine considéré
n = 1000                                   # Nombre de volumes du maillage pour le calcul numérique
dx = L/n                                   # Largeur des volumes
lx = [xmin+dx/2 + k*dx for k in range(n)]  # Liste des abscisses du barycentre de chaque volume
tf = 3                                    # Temps final (en secondes)4
N = 8                                      # Nombre de graphiques affichés
a = 0.5                                    # Nombre de graphiques affichés
lzb = np.zeros(n+2)

# Condition initiale
lsi = barrage(3, 2, 0, n, lx)
lui = np.zeros(n)
lhi = h(lzb, lsi, n)
lt, lh, lu = sol_appr(tf, n, lhi, lui, lzb, lx)
### Interface graphique avec Tkinter

class Application(Frame):
    def __init__(self, master=None):
        super().__init__(master)
        self.master = master
        self.master.title("Interface Graphique")
        self.master.geometry("600x400")
        self.master.resizable(True, True)
        self.pack()
        self.create_widgets()

    def create_widgets(self):
        """création des widgets de l'interface"""
        self.tf_label = Label(self, text="Temps final : ",font=('Times New Roman',12))
        self.tf_label.pack()
        self.tf_entry = Entry(self)
        self.tf_entry.pack()

        self.xmin_label = Label(self, text="X Min : ",font=('Times New Roman',12))
        self.xmin_label.pack()
        self.xmin_entry = Entry(self)
        self.xmin_entry.pack()

        self.xmax_label = Label(self, text="X Max : ",font=('Times New Roman',12))
        self.xmax_label.pack()
        self.xmax_entry = Entry(self)
        self.xmax_entry.pack()

        self.ymin_label = Label(self, text="Y Min : ",font=('Times New Roman',12))
        self.ymin_label.pack()
        self.ymin_entry = Entry(self)
        self.ymin_entry.pack()

        self.ymax_label = Label(self, text="Y Max : ",font=('Times New Roman',12))
        self.ymax_label.pack()
        self.ymax_entry = Entry(self)
        self.ymax_entry.pack()

        self.valider_button = Button(self)
        self.valider_button["text"] = "Valider"
        self.valider_button["command"] = self.validate_parameters
        self.valider_button.pack(pady=10)

        self.quit = Button(self, text="Quitter", fg="red", command=self.master.destroy)
        self.quit.pack(side="bottom", padx=10, pady=10)

    def validate_parameters(self): #récupération des parametres
        self.tf = float(self.tf_entry.get())
        self.xmin = float(self.xmin_entry.get())
        self.xmax = float(self.xmax_entry.get())
        self.ymin = float(self.ymin_entry.get())
        self.ymax = float(self.ymax_entry.get())
        self.show_choices()

    def show_choices(self):#création des boutons pour les données initiales
        self.clear_widgets()

        self.barrage_button = Button(self)
        self.barrage_button["text"] = "Barrage"
        self.barrage_button["command"] = self.choose_barrage
        self.barrage_button.pack(side="top", padx=10, pady=30)

        self.bosse_button = Button(self)
        self.bosse_button["text"] = "Bosse"
        self.bosse_button["command"] = self.choose_bosse
        self.bosse_button.pack(side="top", padx=10, pady=30)

        self.function_button = Button(self)
        self.function_button["text"] = "Profil initial (fonction)"
        self.function_button["command"] = self.choose_function
        self.function_button.pack(side="top", padx=10, pady=30)






#Boutons de retours des différentes fenetres
    def go_back_barrage(self):
        self.barrage_button.pack_forget()
        self.hg_label.pack_forget()
        self.hg_entry.pack_forget()
        self.ud_label.pack_forget()
        self.ug_entry.pack_forget()
        self.ud_entry.pack_forget()
        self.ug_label.pack_forget()
        self.hd_label.pack_forget()
        self.hd_entry.pack_forget()
        self.xl_label.pack_forget()
        self.xl_entry.pack_forget()
        self.angle_label.pack_forget()
        self.angle_entry.pack_forget()
        self.hmin_pente_entry.pack_forget()
        self.hmin_pente_label.pack_forget()
        self.quit_button.pack_forget()
        self.valider_button.pack_forget()
        self.create_widgets()
    def go_back_bosse(self):
        self.bosse_button.pack_forget()
        self.hmax_label.pack_forget()
        self.hmax_entry.pack_forget()
        self.hmin_label.pack_forget()
        self.hmin_entry.pack_forget()
        self.vitessebosse_label.pack_forget()
        self.vitessebosse_entry.pack_forget()
        self.xc_label.pack_forget()
        self.xc_entry.pack_forget()
        self.L_label.pack_forget()
        self.L_entry.pack_forget()
        self.angle_label.pack_forget()
        self.angle_entry.pack_forget()
        self.hmin_pente_entry.pack_forget()
        self.hmin_pente_label.pack_forget()
        self.quit_button.pack_forget()
        self.valider_button.pack_forget()
        self.create_widgets()
    def go_back_profil(self):
        self.function_button.pack_forget()
        self.function_label.pack_forget()
        self.function_entry.pack_forget()
        self.angle_label.pack_forget()
        self.angle_entry.pack_forget()
        self.hmin_pente_entry.pack_forget()
        self.hmin_pente_label.pack_forget()
        self.quit_button.pack_forget()
        self.valider_button.pack_forget()
        self.create_widgets()






    """Choix des données initiales"""
    def choose_barrage(self):
        self.clear_widgets()

        self.hg_label = Label(self, text="Hauteur gauche : ",font=('Times New Roman',12))
        self.hg_label.pack()
        self.hg_entry = Entry(self)
        self.hg_entry.pack()

        self.hd_label = Label(self, text="Hauteur droite : ",font=('Times New Roman',12))
        self.hd_label.pack()
        self.hd_entry = Entry(self)
        self.hd_entry.pack()

        self.ug_label = Label(self, text="Vitesse gauche : ",font=('Times New Roman',12))
        self.ug_label.pack()
        self.ug_entry = Entry(self)
        self.ug_entry.pack()

        self.ud_label = Label(self, text="Vitesse droite : ",font=('Times New Roman',12))
        self.ud_label.pack()
        self.ud_entry = Entry(self)
        self.ud_entry.pack()

        self.xl_label = Label(self, text="Position du changement : ",font=('Times New Roman',12))
        self.xl_label.pack()
        self.xl_entry = Entry(self)
        self.xl_entry.pack()

        self.angle_label = Label(self, text="Angle de la pente (en radians) : ",font=('Times New Roman',12))
        self.angle_label.pack()
        self.angle_entry = Entry(self)
        self.angle_entry.pack()

        self.hmin_pente_label = Label(self, text="Hauteur minimale de la pente : ",font=('Times New Roman',12))
        self.hmin_pente_label.pack()
        self.hmin_pente_entry = Entry(self)
        self.hmin_pente_entry.pack()

        self.valider_button = Button(self)
        self.valider_button["text"] = "Valider"
        self.valider_button["command"] = self.validate_barrage
        self.valider_button.pack()

        self.quit_button = Button(self, text="Retour",fg='red',command=self.go_back_barrage,)
        self.quit_button.pack()

    def validate_barrage(self):
        hg = float(self.hg_entry.get())
        hd = float(self.hd_entry.get())
        ug = float(self.ug_entry.get())
        ud = float(self.ud_entry.get())
        xl = float(self.xl_entry.get())
        angle = float(self.angle_entry.get())
        hmin_pente = float(self.hmin_pente_entry.get())
        self.lsi = barrage(hg, hd, xl, n, lx)
        self.lui = barrage(ug,ud,xl,n,lx)
        self.lzb = pente_angle(hmin_pente, angle, n)
        self.plot_solution()

    def choose_bosse(self):
        self.clear_widgets()

        self.hmax_label = Label(self, text="Hauteur maximale : ",font=('Times New Roman',12))
        self.hmax_label.pack()
        self.hmax_entry = Entry(self)
        self.hmax_entry.pack()

        self.hmin_label = Label(self, text="Hauteur minimale : ",font=('Times New Roman',12))
        self.hmin_label.pack()
        self.hmin_entry = Entry(self)
        self.hmin_entry.pack()

        self.vitessebosse_label = Label(self, text="Fonction de vitesse initiale :",font=('Times New Roman',12))
        self.vitessebosse_label.pack()
        self.vitessebosse_entry = Entry(self)
        self.vitessebosse_entry.pack()

        self.xc_label = Label(self, text="Position du centre : ",font=('Times New Roman',12))
        self.xc_label.pack()
        self.xc_entry = Entry(self)
        self.xc_entry.pack()

        self.L_label = Label(self, text="Largeur : ",font=('Times New Roman',12))
        self.L_label.pack()
        self.L_entry = Entry(self)
        self.L_entry.pack()

        self.angle_label = Label(self, text="Angle de la pente (en radians) : ",font=('Times New Roman',12))
        self.angle_label.pack()
        self.angle_entry = Entry(self)
        self.angle_entry.pack()

        self.hmin_pente_label = Label(self, text="Hauteur minimale de la pente : ",font=('Times New Roman',12))
        self.hmin_pente_label.pack()
        self.hmin_pente_entry = Entry(self)
        self.hmin_pente_entry.pack()

        self.quit_button = Button(self, text="Retour",fg='red', command=self.go_back_bosse)
        self.quit_button.pack()

        self.valider_button = Button(self)
        self.valider_button["text"] = "Valider"
        self.valider_button["command"] = self.validate_bosse
        self.valider_button.pack()



    def validate_bosse(self):
        hmax = float(self.hmax_entry.get())
        hmin = float(self.hmin_entry.get())
        xc = float(self.xc_entry.get())
        L = float(self.L_entry.get())
        vitessebosse_str=self.vitessebosse_entry.get()
        angle = float(self.angle_entry.get())
        hmin_pente = float(self.hmin_pente_entry.get())
        self.lsi = bosse(hmax, hmin, xc, L, lx, n)
        self.lui = self.evaluate_function(vitessebosse_str)
        self.lzb = pente_angle(hmin_pente, angle, n)
        self.plot_solution()

    def choose_function(self):
        self.clear_widgets()

        self.function_label = Label(self, text="Fonction de profil initial : ",font=('Times New Roman',12))
        self.function_label.pack()
        self.function_entry = Entry(self)
        self.function_entry.pack()

        self.vitesse_label = Label(self, text="Fonction de vitesse initiale : ",font=('Times New Roman',12))
        self.vitesse_label.pack()
        self.vitesse_entry = Entry(self)
        self.vitesse_entry.pack()


        self.angle_label = Label(self, text="Angle de la pente (en radians) : ",font=('Times New Roman',12))
        self.angle_label.pack()
        self.angle_entry = Entry(self)
        self.angle_entry.pack()

        self.hmin_pente_label = Label(self, text="Hauteur minimale de la pente : ",font=('Times New Roman',12))
        self.hmin_pente_label.pack()
        self.hmin_pente_entry = Entry(self)
        self.hmin_pente_entry.pack()

        self.quit_button = Button(self, text="Retour",fg='red', command=self.go_back_profil)
        self.quit_button.pack()

        self.valider_button = Button(self)
        self.valider_button["text"] = "Valider"
        self.valider_button["command"] = self.validate_function
        self.valider_button.pack()

    def validate_function(self):
        function_str = self.function_entry.get()
        vitesse_str=self.vitesse_entry.get()
        angle = float(self.angle_entry.get())
        hmin_pente = float(self.hmin_pente_entry.get())
        self.lsi = self.evaluate_function(function_str)
        self.lui = self.evaluate_function(vitesse_str)
        self.lzb = pente_angle(hmin_pente, angle, n)
        self.plot_solution()

    def evaluate_function(self, function_str):
        """Évalue la fonction entrée par l'utilisateur."""
        func = lambda x: eval(function_str)
        return [func(x) for x in lx]
    def clear_widgets(self):
        """Supprime les widgets de saisie existants."""
        for widget in self.winfo_children():
            widget.destroy()




    """Affichage de la solution"""
    def plot_solution(self):
        lt, lh, lu = sol_appr(self.tf, n, self.lsi, self.lui,self.lzb, lx)

        fig, ax = plt.subplots()
        ax.set_xlim([self.xmin, self.xmax])
        ax.set_ylim([self.ymin, self.ymax])
        ax.set_ylabel("Hauteur de l'eau")
        ax.set_title("Solution numérique aux équations de Saint Venant")

        line1, = ax.plot([], [], label="Hauteur d'eau")
        line2, = ax.plot([], [], 'k', label="Fond")
        time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes, fontsize=10)
        def init():
            line1.set_data([], [])
            line2.set_data([], [])
            time_text.set_text('')
            return line1, line2, time_text
        """Changement d'image pour l'animation"""
        def update(frame):
            line1.set_data(lx, np.array(self.lzb[1:-1]) + lh[frame])
            line2.set_data(lx, np.array(self.lzb[1:-1]))
            time_text.set_text('Temps : {:.2f}s'.format(lt[frame]))
            return line1, line2, time_text

        ani = FuncAnimation(fig, update, frames=len(lt), init_func=init, blit=True, interval=10)
        plt.legend()
        plt.show()
"""lancement de la fenetre"""
root = ThemedTk(theme="breeze")
app = Application(master=root)
app.mainloop()
