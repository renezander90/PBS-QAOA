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

import networkx as nx

# Define the problem instance 

# Number of sites
N = 4

# create the graph
Phi = [(1,0),(2,0),(3,1)]

PBS_graph = nx.DiGraph()
PBS_graph.add_edges_from(Phi)
#nx.draw(PBS_graph, with_labels = True)
#plt.show()

# Number of parts
M = PBS_graph.number_of_nodes() 
max_parts = 10


# Define the table of cost_matrix c_{i,j}^x
cost_matrix_data = [
    [2, 1, 2, 1.64], [4, 1, 2, 8.06], [6, 1, 2, 7.31], [8, 1, 2, 2.73], [10, 1, 2, 4.9],
    [2, 1, 3, 1.05], [4, 1, 3, 5.14], [6, 1, 3, 4.66], [8, 1, 3, 1.74], [10, 1, 3, 3.12],
    [2, 1, 4, 1.09], [4, 1, 4, 5.35], [6, 1, 4, 4.85], [8, 1, 4, 1.81], [10, 1, 4, 3.25],
    [2, 1, 5, 1.43], [4, 1, 5, 7.03], [6, 1, 5, 6.38], [8, 1, 5, 2.39], [10, 1, 5, 4.27],
    [2, 1, 6, 0.91], [4, 1, 6, 4.47], [6, 1, 6, 4.06], [8, 1, 6, 1.52], [10, 1, 6, 2.72],
    [2, 1, 7, 1.7], [4, 1, 7, 8.32], [6, 1, 7, 7.55], [8, 1, 7, 2.82], [10, 1, 7, 5.05],
    [2, 2, 3, 0.59], [4, 2, 3, 2.88], [6, 2, 3, 2.61], [8, 2, 3, 0.98], [10, 2, 3, 1.75],
    [2, 2, 4, 0.37], [4, 2, 4, 1.8], [6, 2, 4, 1.63], [8, 2, 4, 0.61], [10, 2, 4, 1.09],
    [2, 2, 5, 1.12], [4, 2, 5, 5.49], [6, 2, 5, 4.98], [8, 2, 5, 1.86], [10, 2, 5, 3.34],
    [2, 2, 6, 0.24], [4, 2, 6, 1.18], [6, 2, 6, 1.07], [8, 2, 6, 0.4], [10, 2, 6, 0.72],
    [2, 2, 7, 1.79], [4, 2, 7, 8.76], [6, 2, 7, 7.95], [8, 2, 7, 2.97], [10, 2, 7, 5.32],
    [2, 3, 4, 0.93], [4, 3, 4, 4.58], [6, 3, 4, 4.15], [8, 3, 4, 1.55], [10, 3, 4, 2.78],
    [2, 3, 5, 1.02], [4, 3, 5, 4.98], [6, 3, 5, 4.52], [8, 3, 5, 1.69], [10, 3, 5, 3.02],
    [2, 3, 6, 1.35], [4, 3, 6, 6.62], [6, 3, 6, 6.01], [8, 3, 6, 2.25], [10, 3, 6, 4.02],
    [2, 3, 7, 1.65], [4, 3, 7, 8.07], [6, 3, 7, 7.32], [8, 3, 7, 2.74], [10, 3, 7, 4.9],
    [2, 4, 5, 1.65], [4, 4, 5, 8.12], [6, 4, 5, 7.36], [8, 4, 5, 2.75], [10, 4, 5, 4.93],
    [2, 4, 6, 0.19], [4, 4, 6, 0.94], [6, 4, 6, 0.85], [8, 4, 6, 0.32], [10, 4, 6, 0.57],
    [2, 4, 7, 1.52], [4, 4, 7, 7.45], [6, 4, 7, 6.76], [8, 4, 7, 2.53], [10, 4, 7, 4.52],
    [2, 5, 6, 1.04], [4, 5, 6, 5.1], [6, 5, 6, 4.63], [8, 5, 6, 1.73], [10, 5, 6, 3.1],
    [2, 5, 7, 0.7], [4, 5, 7, 3.41], [6, 5, 7, 3.1], [8, 5, 7, 1.16], [10, 5, 7, 2.07],
    [2, 6, 7, 1.57], [4, 6, 7, 7.68], [6, 6, 7, 6.97], [8, 6, 7, 2.61], [10, 6, 7, 4.67],
    [3, 1, 2, 5.56], [5, 1, 2, 7.66], [7, 1, 2, 1.0], [9, 1, 2, 1.49],
    [3, 1, 3, 3.54], [5, 1, 3, 4.88], [7, 1, 3, 0.64], [9, 1, 3, 0.95],
    [3, 1, 4, 3.68], [5, 1, 4, 5.08], [7, 1, 4, 0.66], [9, 1, 4, 0.99],
    [3, 1, 5, 4.85], [5, 1, 5, 6.68], [7, 1, 5, 0.87], [9, 1, 5, 1.3],
    [3, 1, 6, 3.08], [5, 1, 6, 4.25], [7, 1, 6, 0.56], [9, 1, 6, 0.83],
    [3, 1, 7, 5.73], [5, 1, 7, 7.9], [7, 1, 7, 1.03], [9, 1, 7, 1.54],
    [3, 2, 3, 1.98], [5, 2, 3, 2.73], [7, 2, 3, 0.36], [9, 2, 3, 0.53],
    [3, 2, 4, 1.24], [5, 2, 4, 1.71], [7, 2, 4, 0.22], [9, 2, 4, 0.33],
    [3, 2, 5, 3.79], [5, 2, 5, 5.22], [7, 2, 5, 0.68], [9, 2, 5, 1.02],
    [3, 2, 6, 0.82], [5, 2, 6, 1.12], [7, 2, 6, 0.15], [9, 2, 6, 0.22],
    [3, 2, 7, 6.04], [5, 2, 7, 8.32], [7, 2, 7, 1.09], [9, 2, 7, 1.62],
    [3, 3, 4, 3.15], [5, 3, 4, 4.35], [7, 3, 4, 0.57], [9, 3, 4, 0.85],
    [3, 3, 5, 3.43], [5, 3, 5, 4.73], [7, 3, 5, 0.62], [9, 3, 5, 0.92],
    [3, 3, 6, 4.56], [5, 3, 6, 6.29], [7, 3, 6, 0.82], [9, 3, 6, 1.23],
    [3, 3, 7, 5.56], [5, 3, 7, 7.66], [7, 3, 7, 1.0], [9, 3, 7, 1.5],
    [3, 4, 5, 5.59], [5, 4, 5, 7.71], [7, 4, 5, 1.01], [9, 4, 5, 1.5],
    [3, 4, 6, 0.64], [5, 4, 6, 0.89], [7, 4, 6, 0.12], [9, 4, 6, 0.17],
    [3, 4, 7, 5.13], [5, 4, 7, 7.07], [7, 4, 7, 0.93], [9, 4, 7, 1.38],
    [3, 5, 6, 3.51], [5, 5, 6, 4.84], [7, 5, 6, 0.63], [9, 5, 6, 0.95],
    [3, 5, 7, 2.35], [5, 5, 7, 3.24], [7, 5, 7, 0.42], [9, 5, 7, 0.63],
    [3, 6, 7, 5.3], [5, 6, 7, 7.3], [7, 6, 7, 0.96], [9, 6, 7, 1.42]
] 

for i1 in range(len(cost_matrix_data)):
    cost_matrix_data[i1][0] -= 1
    cost_matrix_data[i1][1] -= 1
    cost_matrix_data[i1][2] -= 1

cost_coeff={}
for i2 in range(1, max_parts):
    cost_coeff[i2]={}

for item in cost_matrix_data:
    #print(item)
    cost_coeff[item[0]].setdefault((item[1],item[2]), item[3])

