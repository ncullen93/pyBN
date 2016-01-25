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
from pyBN.classes.bayesnet import BayesNet


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
    bn = BayesNet()
    V = []
    E = []
    data = {}
    _scope = {} # key = vertex
    _cpt = {} # 
    _val = {}

    with open(path, 'r') as f:
        while True:
            line = f.readline()
            if 'variable' in line:
                new_vertex = line.split()[1]
                V.append(new_vertex) # add to bn.V
                data[new_vertex] = {} # add empty dict to bn.data
                new_line = f.readline()
                new_vals = new_line.replace(',', ' ').split()[6:-1] # list of vals
                data[new_vertex]['vals'] = new_vals # add vals to bn.data dict
                data[new_vertex]['numoutcomes'] = len(new_vals)
                data[new_vertex]['children'] = []
                data[new_vertex]['cprob'] = []
            elif 'probability' in line:
                line = line.replace(',', ' ')
                child_rv = line.split()[2]
                parent_rvs = line.split()[4:-2]
                num_tail = len(parent_rvs)
                if num_tail == 0: # prior
                    data[child_rv]['parents'] = []
                    new_line = f.readline().replace(';', ' ').replace(',',' ').split()
                    prob_values = new_line[1:]
                    data[child_rv]['cprob'] = map(float,prob_values)
                else: # not a prior
                    data[child_rv]['parents'] = list(parent_rvs)
                    for parent in parent_rvs:
                        E.append([parent,child_rv])
                        data[parent]['children'].append(child_rv)
                    while True:
                        new_line = f.readline()
                        if '}' in new_line:
                            break
                        new_line = new_line.replace(',',' ').replace(';',' ').replace('(', ' ').replace(')', ' ').split()
                        prob_values = new_line[-(data[new_vertex]['numoutcomes']):]
                        prob_values = map(float,prob_values)
                        data[child_rv]['cprob'].append(prob_values)
            if line == '':
                break

    #bn.V = V
    #bn.E = E
    #bn.data = data

    # ADDDED
    factors = {}
    for rv in V:
        f = Factor(_scope[rv],_cpt[rv],_val[rv])
        factors[rv] = f
    bn.factors = factors

    return bn

def read_json(path):
    """
    Overview
    --------


    Parameters
    ----------


    Returns
    -------


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
