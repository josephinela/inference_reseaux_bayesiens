from Parsing import *
import copy
import numpy as np
from LikelihoodWeightingInference import likelihoodWeighting
from RejectionMethod import methode_rejet
import time


data = ""
path = "./bn.bif"
with open(path, "r+") as file:
    data = file.read()


data_reseaux  =  ["network"+r for r in data.split("network")][1:]


choix_reseaux = 4


#print(reseaux[choix_reseaux])
reseau = ReseauBayesien(data_reseaux[choix_reseaux])

Q = "N2"
observations = {"N7" : 1, "N3": 0 }
res = [0.9722035837848854, 0.027796416215114632]

"""
time1 = time.time()
print(likelihoodWeighting(reseau=reseau, Q=Q, observations=observations, nbrEchantillon=100000, resultat=res))
time2 = time.time()
print(time2-time1)
"""


print('REJET :')
time3= time.time()
print(methode_rejet(reseau=reseau, Q=Q, observations=observations, nbrEchantillon=100000, resultat = res))
time4=time.time()
print(time4-time3)


