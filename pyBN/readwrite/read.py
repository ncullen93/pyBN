"""
*************
Read BayesNet
from File
*************

Read a BayesNet object from a file. These
functions all currently work, but support
for more formats should be added in the future.

"""

__author__ = """Nicholas Cullen <ncullen.th@dartouth.edu>"""

import json
import numpy as np

from pyBN.classes.bayesnet import BayesNet
from pyBN.classes.factor import Factor


def read_bn(path):
    """
    Wrapper function for reading BayesNet objects
    from various file types.

    Arguments
    ---------
    *path* : a string
        The path (relative or absolute) - MUST
        include extension -> that's how we know
        which file reader to call.

    Returns
    -------
    *bn* : a BayesNet object

    Effects
    -------
    None

    Notes
    -----

    """
    if '.bif' in path:
        return read_bif(path)
    elif '.bn' in path:
        return read_json(path)
    else:
        print "Path Extension not recognized"

def read_bif(path):
    """
    This function reads a .bif file into a
    BayesNet object. It's probably not the 
    fastest or prettiest but it gets the job
    done.

    Arguments
    ---------
    *path* : a string
        The path

    Returns
    -------
    *bn* : a BayesNet object

    Effects
    -------
    None

    Notes
    -----
    Arguments for a factor:
        scope (list)
        cpt (list)
        vals (dict, key=rv, val=list of values)

    """
    _scope = {} # key = vertex, value = list of vertices in the scope (includind itself)
    _cpt = {} # key = vertex, value = list (then numpy array)
    _vals = {} # key = vertex, value = dict where key=vertex, val=list of values

    with open(path, 'r') as f:
        while True:
            line = f.readline()
            if 'variable' in line:
                new_vertex = line.split()[1]

                _scope[new_vertex] = [new_vertex]
                _cpt[new_vertex] = []
                _vals[new_vertex] = []

                new_line = f.readline()
                new_vals = new_line.replace(',', ' ').split()[6:-1] # list of vals
                _vals[new_vertex] = new_vals
                num_outcomes = len(new_vals)
            elif 'probability' in line:
                line = line.replace(',', ' ')
                child_rv = line.split()[2]
                parent_rvs = line.split()[4:-2]

                if len(parent_rvs) == 0: # prior
                    new_line = f.readline().replace(';', ' ').replace(',',' ').split()
                    prob_values = new_line[1:]
                    _cpt[child_rv] = map(float,prob_values)
                else: # not a prior
                    _scope[child_rv].extend(list(parent_rvs))
                    while True:
                        new_line = f.readline()
                        if '}' in new_line:
                            break
                        new_line = new_line.replace(',',' ').replace(';',' ').replace('(', ' ').replace(')', ' ').split()
                        prob_values = new_line[-(len(_vals[new_vertex])):]
                        prob_values = map(float,prob_values)
                        _cpt[child_rv].append(prob_values)
            if line == '':
                break

    # CREATE FACTORS
    factors = {}
    for rv in _scope.keys():
        _scopevals = dict((rv,vals) for rv, vals in _vals.items() if rv in _scope[rv])
        f = Factor(_scope[rv],np.array(_cpt[rv]),_scopevals)
        factors[rv] = f
    bn = BayesNet(factors)
    #bn.factors = factors

    return bn

def read_json(path):
    """
    Read a BayesNet object from the json format. This
    format has the ".bn" extension and is completely
    unique to pyBN.

    Arguments
    ---------
    *path* : a string
        The file path

    Returns
    -------
    None

    Effects
    -------
    - Instantiates and sets a new BayesNet object

    Notes
    -----
    
    This function reads in a libpgm-style format into a bn object

    File Format:
        {
            "V": ["Letter", "Grade", "Intelligence", "SAT", "Difficulty"],
            "E": [["Intelligence", "Grade"],
                ["Difficulty", "Grade"],
                ["Intelligence", "SAT"],
                ["Grade", "Letter"]],
            "Vdata": {
                "Letter": {
                    "ord": 4,
                    "numoutcomes": 2,
                    "vals": ["weak", "strong"],
                    "parents": ["Grade"],
                    "children": None,
                    "cprob": [[.1, .9],[.4, .6],[.99, .01]]
                },
                ...
        }


    """
    def byteify(input):
        if isinstance(input, dict):
            return {byteify(key):byteify(value) for key,value in input.iteritems()}
        elif isinstance(input, list):
            return [byteify(element) for element in input]
        elif isinstance(input, unicode):
            return input.encode('utf-8')
        else:
            return input

    bn = BayesNet()
    V = []
    E = []
    data = {}
    f = open(path,'r')
    ftxt = f.read()

    success=False
    try:
        data = byteify(json.loads(ftxt))
        bn.V = data['V']
        bn.E = data['E']
        bn.data = data['Vdata']
        success = True
    except ValueError:
        print "Could not read file - check format"

    return bn
