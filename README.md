# Shallow-water
Codes 1D et 2D simulant des écoulements d'eau à partir des équations de Saint-Venant

Les entrées à modifier par l'utilisateur sont signaler par ## A MODIFIER

Liste des programmes et de leurs caractéristiques :

PROGRAMME 1 / 1D, flux HLL, projection horizontale

PROGRAMME 2 / 1D, flux HLL, projection tangente, Utilisable uniquement pour des fonds formant une pente constante avec l’horizontale

PROGRAMME 3 / 2D, flux de Rusanov, projection horizontale, Utilisable uniquement pour un fond plat, ne traite pas les zones sèches

PROGRAMME 4 / 2D, flux HLL, projection horizontale

PROGRAMME 5 / 1D, flux HLL, interface graphique

Pour les codes 2D, il faut au préalable importer la bibliothèque de maillages générés avec gmsh disponible sur le dépôt GitHub et installer le module python pyvista pour pouvoir les ouvrir sous Python.
