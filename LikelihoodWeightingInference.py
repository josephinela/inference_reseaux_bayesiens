from Parsing import ReseauBayesien
import copy
import numpy as np

def parcours_variables (reseau):
    ordre = []
    parents = copy.deepcopy(reseau.parents_variables)
    num_var = len(parents.keys())
    while len(ordre) != num_var:
        for variable, par in parents.copy().items() : 
            if len(par) == 0 :
                ordre.append(variable)
                for vv in parents : 
                    if variable in parents[vv]:
                        parents[vv].remove(variable)
                del parents[variable]
    return ordre
        

            


def likelihoodWeighting(reseau : ReseauBayesien, Q, observations, nbrEchantillon, resultat):
    """
    Estimation d'une requete de la forme P(Q | X1 =v1, ... ,Xk = vk) 
    avec la methode du Likelihood Weighting

    Args :
        reseau  (class ReseauBayesien):  l'objet qui représente le réseau bayésien
        Q (string) : le nom de la variable dans la requete
        observations (Dict) : un dictionnaire potentiellement cide avec les observations du réseau de la forme
                              {Var1 : 1 , Var2 : 0 }  
                              1 : true  ,     0 : false

    Returns : 
        list , de la forme [ProbabiliteDuVrai, ProbabiliteDuFaux]
    
    """

    
    var_observees = list(observations.keys())
    ensemble_echantillons = []
    parents = copy.deepcopy(reseau.parents_variables)
    ordre_du_parcours = parcours_variables(reseau)
    poids_vrai = 0
    somme_des_poids = 0
    for ii in range(nbrEchantillon):
        echantillon = {}
        poids = 1
        for variable in ordre_du_parcours:
            nbr_parents = len(parents[variable])
            #print(variable)

            if variable in var_observees : 

                echantillon[variable] = observations[variable]

                #P(variable | parents(variable))
                ind = 0
                for i in range(nbr_parents) :
                    ind += int(not echantillon[parents[variable][i]])* 2**(nbr_parents-1-i)

                poids = poids * reseau.tables_TCP[variable][int(not observations[variable])][ind]

                #print(reseau.tables_TCP[variable][int(not observations[variable])])
                #print(reseau.tables_TCP[variable][int(not observations[variable])][ind])

            else:
                ind = 0
                for i in range(nbr_parents) :
                    ind += int(not echantillon[parents[variable][i]])* 2**(nbr_parents-1-i)
                pVraie= reseau.tables_TCP[variable][0][ind]
                pFaux=1-pVraie

                echantillon[variable] = np.random.choice([1,0], p=[pVraie, pFaux])

                #print(pVraie)
            #print(poids)
            
        #print(echantillon)
        ensemble_echantillons.append((echantillon, poids))
        if echantillon[Q] == 1 :  
            poids_vrai +=  poids
        somme_des_poids += poids

    somme_des_poids = sum([e[1] for e in ensemble_echantillons] )
    estimationPVrai = poids_vrai/somme_des_poids 
    print([1-estimationPVrai, estimationPVrai])
    return abs(resultat[0] - (1-estimationPVrai))


