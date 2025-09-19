'''Module is a first attempt for a brute force search method for optimal CX gate numbers in CX circuits. Currently only works for small number of gates and qubits.'''


import numpy as np
from copy import deepcopy
import random
from anytree import Node, RenderTree
import pyzx as zx

def plus(m1,m2):
    return np.logical_xor(m1,m2)

def mat2int(m):
    return m.astype('int')

def j_to_i(m,j,i):
    mc=deepcopy(m)
    mc[i,:]=plus(mc[i,:],mc[j,:])
    return mc

def equals(m1,m2):
    return np.array_equal(m1,m2)

def is_identity(m):
    e=np.eye(m.shape[0],dtype='bool')
    return equals(m,e)


def random_matrix(dimension, steps):
    m=np.eye(dimension,dtype='bool')
    for _ in range(steps):
        i=random.randint(0, dimension-1)
        while True:
            j=random.randint(0, dimension-1)
            if not i==j:
                break
        m=j_to_i(m,j,i)
    return m
            
    

def test_equ(m1,m2):
    return np.sum(m1)==np.sum(m2)
 

def expand(tree):
    for leaf in tree.leaves:
        d=leaf.data.shape[0]
        for i in range(d):
            for j in range(d):
                if not i==j:
                    new_m=j_to_i(leaf.data,j,i)
                    node=Node(str(j)+'to'+str(i), parent=leaf, data=new_m)
                    if np.sum(new_m)==d:
                        return (True, node)
    return (False, None)

   

def print_tree(tree):
    for pre, _, node in RenderTree(tree):
        print(f"{pre}{node.name}")


        
def circuit_to_CX_matrix(c):
    n_qubit=c.qubits
    m=np.eye(n_qubit,dtype='bool')
    for gate in c.gates:
        assert isinstance(gate,zx.circuit.gates.CNOT), 'circuit must be pure CX circuit'
        target, control=gate.target, gate.control
        m=j_to_i(m,control,target)
    return m
            
      
def optimal_sequence(m,print_depths=True):
    tree = Node("Root", data=m)
    success=False
    i=0
    if print_depths:
        print('search depths:')
    while not success:
        if print_depths:
            print(i)
        i+=1
        success, succ_node=expand(tree)
    return (succ_node,tree)


def sequence_to_circuit(succ_node, n_qubit):
    c=zx.Circuit(n_qubit)
    sequence=[s.name for s in succ_node.ancestors if not s.name=='Root']
    if not succ_node.name=='Root':
        sequence.append(succ_node.name)
    sequence=[(int(s.split('to')[0]),int(s.split('to')[1])) for s in sequence]
    
    
    for s in reversed(sequence):
        c.add_gate(zx.circuit.gates.CNOT(s[0],s[1]))
    return c


def optimize_CX_circuit(c,print_depths=True):
    '''Finds minimal CX gate sequences to match a CX circuit.
       Attention, the function works modulo SWAP gates. This is implemented in the
       function expand where np.sum(new_m)==d instead of new_m=identity is checked. '''
    n_qubit=c.qubits
    m=circuit_to_CX_matrix(c)
    succ_node,tree=optimal_sequence(m,print_depths=print_depths)
    c_new=sequence_to_circuit(succ_node, n_qubit)
    return c_new





    
    
    