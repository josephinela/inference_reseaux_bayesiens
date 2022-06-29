
import copy
import numpy as np
from Parsing import ReseauBayesien
from LikelihoodWeightingInference import parcours_variables



def methode_rejet(reseau: ReseauBayesien, Q, observations, nbrEchantillon, resultat):
    """
    Estimation d'une requete de la forme P(Q | X1 =v1, ... ,Xk = vk)
    avec la methode du rejet

    Args :
        reseau  (class ReseauBayesien):  l'objet qui représente le réseau bayésien
        Q (string) : le nom de la variable dans la requete
        observations (Dict) : un dictionnaire potentiellement vide avec les observations du réseau de la forme
                              {Var1 : 1 , Var2 : 0 }
                              1 : true  ,     0 : false

    Returns :
        list , de la forme [ProbabiliteDuVrai, ProbabiliteDuFaux]

    """

    parents = copy.deepcopy(reseau.parents_variables)
    ordre_du_parcours = parcours_variables(reseau)
    ensemble_echantillon = []
    proba = reseau.tables_TCP
    cpt = 0
    count_proba = 0
    test= True
    while cpt<= nbrEchantillon :
        test=True
        echantillon = {}
        for elt in ordre_du_parcours :
            list_proba = proba[elt][0]
            if len(parents[elt])>0:
                for par in parents[elt] :
                    if echantillon[par]==1:
                        list_proba=list_proba[:len(list_proba)//2]
                    else :
                        list_proba = list_proba[len(list_proba) // 2:]
            list_proba= np.append(list_proba, 1 - list_proba[0])
            list_proba = list_proba[::-1]
            echantillon[elt] = np.random.choice(2, p=list_proba)


        for elt, value in observations.items() :
            if (value == 0 and echantillon[elt] == 1) or (value == 1 and echantillon[elt] == 0):
                test=False
                cpt += 1

        if test == False :
            continue
        ensemble_echantillon.append(echantillon)
        cpt+=1
    if len(ensemble_echantillon) == 0:
        return 'no sample'
    for elt in ensemble_echantillon :
        if elt[Q]==1:
            count_proba+=1

    print(len(ensemble_echantillon))
    return [abs(resultat[0] - (1-count_proba/len(ensemble_echantillon)))]


