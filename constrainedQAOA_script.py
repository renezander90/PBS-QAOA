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

from state_preparation_parametrized import *
from qrisp.quantum_backtracking import OHQInt
from classical_cost_func import cost_function, format_coeffs, cost_symp
import numpy as np
import matplotlib.pyplot as plt
from encode_PBS_problem import PBS_graph, N, M, cost_coeff

tot_coeff = format_coeffs(cost_coeff, N)

####################
# Compute optimal solution via unstructured search (brute force method)
####################

qtype = OHQInt(N)
q_array_2 = QuantumArray(qtype = qtype, shape = (M))

uniform_state = prepare_pbs_state(PBS_graph, 0, N, q_array_2)
meas_res = uniform_state.get_measurement()

cl_cost = cost_function(tot_coeff,PBS_graph)

solutions = {}
for k,v in meas_res.items():
    c = cl_cost({k:1})  
    solutions[k] = c
sorted_solutions = sorted(solutions.items(), key=lambda item: item[1])
min_cost = sorted_solutions[0][1]
optimal_solution = sorted_solutions[0][0]

print('###  Best assignment (brute force): ###')
print('   Cost:  ',sorted_solutions[0][1])
print('   State: ',sorted_solutions[0][0])
print('   Number of admissible states: ',len(sorted_solutions))

####################
####################

qtype = OHQInt(N)
q_array = QuantumArray(qtype = qtype, shape = (M))

# Cost function as SymPy polynomial
C,S = cost_symp(tot_coeff,M,N,PBS_graph)
ord_symbs=list(S.values())

def cost_op(q_array, gamma):
    app_sb_phase_polynomial(q_array, C, ord_symbs, t=-gamma)

# Initialization function
init_func = pbs_state_init(PBS_graph, 0 , N)

# Inverse state preparation operator
def inv_init_func(q_array):
    with invert():
        init_func(q_array)

# Define mixer operator
def mixer_op(q_array, beta):
    with conjugate(inv_init_func)(q_array):
        for i in range(len(q_array)):
            mcp(beta, q_array[i], ctrl_state = 0)

# Classical cost function
#values = []
cl_cost = cost_function(tot_coeff,PBS_graph)

# Define QAOA problem
from qrisp.qaoa import *
qaoaPBS = QAOAProblem(cost_operator=cost_op ,mixer=mixer_op, cl_cost_function=cl_cost, init_type='tqa')
qaoaPBS.set_init_function(init_func)
depth = 3
res = qaoaPBS.run(q_array, depth,  max_iter = 100)

#
# Evaluations
#
best_result= list(res.keys())[0]
print('###  Best assignment: ###')
print('   Cost:  ',cl_cost({best_result:1.0}))
print('   State: ',best_result)
print('   Prob:  ',list(res.values())[0])
