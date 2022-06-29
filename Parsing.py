import re
from copy import copy
from itertools import product
import numpy as np

from joblib import Parallel, delayed
from pyparsing import (
    CharsNotIn,
    Group,
    OneOrMore,
    Optional,
    Suppress,
    Word,
    ZeroOrMore,
    alphanums,
    nums,
    printables,
)


def get_variable_grammar():
        """
        A method that returns variable grammar
        """
        # Defining a expression for valid word
        word_expr = Word(alphanums + "_" + "-")
        word_expr2 = Word(initChars=printables, excludeChars=["{", "}", ",", " "])
        name_expr = Suppress("variable") + word_expr + Suppress("{")
        state_expr = ZeroOrMore(word_expr2 + Optional(Suppress(",")))
        # Defining a variable state expression
        variable_state_expr = (
            Suppress("type")
            + Suppress(word_expr)
            + Suppress("[")
            + Suppress(Word(nums))
            + Suppress("]")
            + Suppress("{")
            + Group(state_expr)
            + Suppress("}")
            + Suppress(";")
        )
        # variable states is of the form type description [args] { val1, val2 }; (comma may or may not be present)

        property_expr = (
            Suppress("property") + CharsNotIn(";") + Suppress(";")
        )  # Creating a expr to find property

        return name_expr, variable_state_expr, property_expr




def get_probability_grammar():
        """
        A method that returns probability grammar
        """
        # Creating valid word expression for probability, it is of the format
        # wor1 | var2 , var3 or var1 var2 var3 or simply var
        word_expr = (
            Word(alphanums + "-" + "_")
            + Suppress(Optional("|"))
            + Suppress(Optional(","))
        )
        word_expr2 = Word(
            initChars=printables, excludeChars=[",", ")", " ", "("]
        ) + Suppress(Optional(","))
        # creating an expression for valid numbers, of the format
        # 1.00 or 1 or 1.00. 0.00 or 9.8e-5 etc
        num_expr = Word(nums + "-" + "+" + "e" + "E" + ".") + Suppress(Optional(","))
        probability_expr = (
            Suppress("probability")
            + Suppress("(")
            + OneOrMore(word_expr)
            + Suppress(")")
        )
        optional_expr = Suppress("(") + OneOrMore(word_expr2) + Suppress(")")
        probab_attributes = optional_expr | Suppress("table")
        cpd_expr = probab_attributes + OneOrMore(num_expr)

        return probability_expr, cpd_expr


def variable_block(data):
    start = re.finditer("variable", data)
    for index in start:
        end = data.find("}\n", index.start())
        yield data[index.start() : end]

def probability_block(data):
    start = re.finditer("probability", data)
    for index in start:
        end = data.find("}\n", index.start())
        yield data[index.start() : end]

def get_network_name(data):
        """
        Returns the name of the network
        """
        start = data.find("network")
        end = data.find("}\n", start)
        # Creating a network attribute
        network_attribute = Suppress("network") + Word(alphanums + "_" + "-") + "{"
        network_name = network_attribute.searchString(data[start:end])[0][0]

        return network_name


def get_variables(data):
        """
        Returns list of variables of the network
        """
        variable_names = []
        for block in variable_block(data):
            name = name_expr.searchString(block)[0][0]
            variable_names.append(name)

        return variable_names


def get_states(data):
        """
        Returns the states of variables present in the network

        """
        variable_states = {}
        for block in variable_block(data):
            name = name_expr.searchString(block)[0][0]
            variable_states[name] = list(state_expr.searchString(block)[0][0])

        return variable_states

def get_parents(data):
        """
        Returns the parents of the variables present in the network
        """
        variable_parents = {}
        for block in probability_block(data):
            names = probability_expr.searchString(block.split("\n")[0])[0]
            variable_parents[names[0]] = names[1:]
        return variable_parents


def _get_values_from_block( block, variable_states):
    names = probability_expr.searchString(block)
    var_name, parents = names[0][0], names[0][1:]
    cpds = cpd_expr.searchString(block)
    
    # Check if the block is a table.
    if bool(re.search(".*\\n[ ]*table .*\n.*", block)):
        arr = np.array([float(j) for i in cpds for j in i])
        arr = arr.reshape(
            (
                len(variable_states[var_name]),
                arr.size // len(variable_states[var_name]),
            )
        )
    else:
        arr_length = np.prod([len(variable_states[var]) for var in parents])
        arr = np.zeros((len(variable_states[var_name]), int(arr_length)))
        values_dict = {}
        for prob_line in cpds:
            states = prob_line[: len(parents)]
            vals = [float(i) for i in prob_line[len(parents) :]]
            values_dict[tuple(states)] = vals
        for index, combination in enumerate(
            product(*[variable_states[var] for var in parents])
        ):
            arr[:, index] = values_dict[combination]
    return var_name, arr

def get_values(data):
        """
        Returns the CPD of the variables present in the network

        """
        variable_states = get_states(data)
        cpd_values = Parallel(n_jobs=1)(
            delayed(_get_values_from_block)(block, variable_states)
            for block in probability_block(data)
        )

        variable_cpds = {}
        for var_name, arr in cpd_values:
            variable_cpds[var_name] = arr

        return variable_cpds






name_expr, state_expr, property_expr = get_variable_grammar()
probability_expr, cpd_expr = get_probability_grammar()



class ReseauBayesien : 
    """
    Classe qui représente un réseau bayesien
    """

    def __init__(self, data) -> None:
        """
        args:
            data (string) : la chaine de caractères qui represente le reseau
        """
        self.data = data
        self.nom_reseau = get_network_name(data)
        self.tables_TCP = get_values(data)
        self.parents_variables = get_parents(data)

    def get_edges(self):
        """
        Returns the edges of the network
        """
        variable_parents = self.parents_variables
        edges = [
            [value, key]
            for key in variable_parents.keys()
            for value in variable_parents[key]
        ]
        return edges