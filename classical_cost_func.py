"""
\********************************************************************************
* Copyright (c) 2024 the Qrisp authors
*
* This program and the accompanying materials are made available under the
* terms of the Eclipse Public License 2.0 which is available at
* http://www.eclipse.org/legal/epl-2.0.
*
* This Source Code may also be made available under the following Secondary
* Licenses when the conditions for such availability set forth in the Eclipse
* Public License, v. 2.0 are satisfied: GNU General Public License, version 2
* with the GNU Classpath Exception which is
* available at https://www.gnu.org/software/classpath/license.html.
*
* SPDX-License-Identifier: EPL-2.0 OR GPL-2.0 WITH Classpath-exception-2.0
********************************************************************************/
"""

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from sympy import Symbol
from qrisp import *
from qrisp.quantum_backtracking import OHQInt
from state_preparation_parametrized import *
from scipy.optimize import minimize


def format_coeffs(coeff_dict, N, set_same_site_coeffs = False):
    tot_coeff = {}
    for k,v in coeff_dict.items():
        tot_coeff[k]={}
        for rs,c in v.items():
            i,j=rs[0],rs[1]
            if i > N-1 or j > N-1:
                continue
            tot_coeff[k][(i,j)]=c
            tot_coeff[k][(j,i)]=c
            #avoid movements from a destination to the same
            if set_same_site_coeffs: 
                tot_coeff[k][(i,i)]=1000 
                tot_coeff[k][(j,j)]=1000
            else:
                tot_coeff[k][(i,i)]=0 
                tot_coeff[k][(j,j)]=0

    return tot_coeff


def cost_symp(tot_coeff, n_parts, n_sites, G):
    """
    This methods returs the classical cost function for a given PBS problem as SymPy polynomial and a dictionary of symbols.

    Parameters
    ----------
    tot_coeff : dict(dict)
        A dictionary of dictionaries, containing the coefficents of the PBS problem objective function.
    n_parts : int
        The number of parts.
    n_sites : int
        The number of sites.
    G : networkx.DiGraph
        The directed graph representing the PBS.
    values : list, optional
        A list for storing intermediate results.

    Returns
    -------
    cost : SymPy expression
        The classical cost function of the problem.
    x : list[Symbol]
        A dictionary of symbols.

    """

    combinations=list(tot_coeff[1].keys())

    x = {str(r)+str(i): Symbol(f"x{r}{i}") for r in range(n_parts) for i in range(n_sites)}
    
    cost=sum([ sum([ tot_coeff[r][(i,j)]*x[str(r)+str(i)]*x[str(s)+str(j)] for (i,j) in combinations]) for r,s in G.edges()])
    
    return cost, x


def cost_function(tot_coeff, G, values=None):
    """
    This methods returns the classical cost function for a given PBS problem.
    This function is evaluated on a dictionary of measurement results.

    Parameters
    ----------
    tot_coeff : dict(dict)
        A dictionary of dictionaries, containing the coefficents of the PBS problem objective function.
    G : networkx.DiGraph
        The directed graph representing the PBS.
    values : list, optional
        A list for storing intermediate results.

    Returns
    -------
    cl_cost_function : function
        The classical cost function of the problem.

    """

    def cl_cost_function(res_dic):
        cost=0
        for sites,v in res_dic.items():
            for edge in G.edges():
                cost += v*tot_coeff[edge[0]][(sites[edge[0]],sites[edge[1]])]

        if values is not None:
            values.append(cost)

        return cost
    
    return cl_cost_function


