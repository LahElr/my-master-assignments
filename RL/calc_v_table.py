import numpy
from Q_learing import QTable
import os

exp_path = r"./box_goal_dist_off_q"
q_table_path = os.path.join(exp_path,"best.npy")
epsilon = 0.01

q_table = numpy.load(q_table_path)
v_table = numpy.zeros(q_table.shape[:4],dtype = q_table.dtype)

filter_o = numpy.array([epsilon/3]*4,dtype = q_table.dtype)

q_table = q_table.view(QTable)

for i in range(v_table.shape[0]):
    for j in range(v_table.shape[1]):
        for k in range(v_table.shape[2]):
            for v in range(v_table.shape[3]):
                filter = filter_o.copy()
                filter[numpy.argmax(q_table[i,j,k,v])] = 1 - epsilon
                v_table[i,j,k,v] = numpy.sum(filter*q_table[i,j,k,v])

grid_table = numpy.mean(v_table,axis=(2,3))
agent_q_table = numpy.mean(q_table,axis=(2,3))
policy_table = numpy.zeros(agent_q_table.shape[:2],dtype=numpy.int8)

for i in range(policy_table.shape[0]):
    for j in range(policy_table.shape[1]):
        policy_table[i,j] = numpy.argmax(agent_q_table[i,j])

numpy.save(os.path.join(exp_path,"v_table.npy"),v_table)
numpy.save(os.path.join(exp_path,"grid_table.npy"),grid_table)
numpy.save(os.path.join(exp_path,"policy_table.npy"),policy_table)

print(grid_table)
print(policy_table)