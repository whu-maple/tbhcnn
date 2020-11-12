from model import Variation2
import tensorflow as tf 
import numpy as np

# we take the example of the 13-AGNR 

# Define the tight-binding representation by selecting the real-space Hamiltonian considered
"""
Here, to ensure that the Hamiltonians with the opposite lattice vectors are always taken into consideration simultaneously, and they tranpose each other
only one of each pair of the opposite lattice vectors should be included in the rvectors array, the other one will be handled automatically
e.g., for this system, both np.array([[0,0,0],[0,0,1],]) and np.array([[0,0,0],[0,0,-1],]) represent the lattice vector set np.array([[0,0,0],[0,0,1],[0,0,-1]])
and in the end we shall get the 3 real-space Hamiltonian matrices we actually used to build the tight-binding model
"""
rvectors_without_opposite = np.array([[0,0,0],[0,0,1],], dtype=np.int32) # in units of[a, b, c] (a, b, and c are the real-space basis vectors; [l, n, m] means the lattice vector l*a+n*b+m*c)

# Energy bands data as references
references = np.load("./data/input/variation2 13AGNR/13AGNR-references.npy")
kvectors = np.load("./data/input/variation2 13AGNR/13AGNR-kpoints.npy")  # in units of 1/2pi*[ak, bk, ck] (ak, bk, and ck are the corresponding k-space basis vectors; [l, n, m] means the k-vector (l/2pi)*ak+(n/2pi)*bk+(m/2pi)*ck)
indices = [8, 18] #point out the indices of the reference bands within the whole tight-binding band structure

# Coordinate file and lattice information
xyz_file = "./data/input/variation2 13AGNR/13AGNR.xyz"
lattice = np.array([[15.0,0.0,0.0],
                    [0.0,32.0,0.0],
                    [0.0,0.0,4.26],])

# Hyperparameters
Optimizer = tf.train.AdamOptimizer(0.001)
max_steps = 20000 # the max training steps, when surpassing it, end the training anyway

sess = tf.Session()

def fitting(Optimizer, max_training_steps, variation2, bands_indices):
    reproduces = variation2.compute_bands(bands_indices) 
    loss = tf.reduce_mean(tf.square(reproduces-variation2.references))
    train = Optimizer.minimize(loss)
    init = tf.global_variables_initializer()
    sess.run (init)
    train_steps = 0
    print("initial loss value: %.8f" % (sess.run(loss)))
    while train_steps<=max_steps:
        sess.run(train)
        train_steps += 1
        if train_steps%1000 == 0:
            print("steps:%s, loss:%.8f" % (train_steps, sess.run(loss)))
    return variation2.HR
    
def main():
    variation2 = Variation2()
    variation2.read_geometry(xyz_file, lattice, rvectors_without_opposite)
    variation2.read_training_set(references, kvectors)
    variation2.initialize()
    
    # output the optimized real-space Hamiltonians, their corresponding lattice vectors, their computed 
    # bandstructure, and their reproduction of the reference bands

    Resulting_Hamiltonian = fitting(Optimizer, max_steps, variation2, indices)
    Resulting_Hamiltonian = sess.run(tf.cast(Resulting_Hamiltonian, tf.float64))
    Rvectors_of_the_resulting_hamiltonian = sess.run(tf.cast(variation2.R, tf.int32))
    Reproduced_TB_bandstructure = sess.run(variation2.wholebandstructure)
    Reproduced_TB_bands = sess.run(variation2.reproduces)
            
    np.save("resulting_real_space_hamiltonians.npy", Resulting_Hamiltonian)
    np.save("rvectors_of_the_resulting_hamiltonian.npy", Rvectors_of_the_resulting_hamiltonian)
    np.save("TB_bandstructure.npy", Reproduced_TB_bandstructure)
    np.save("reproduced_TB_bands.npy", Reproduced_TB_bands)
    
    print("final mean square error for band reproduction : %.8f" % sess.run(tf.reduce_mean(tf.square(variation2.reproduces-variation2.references))))
    sess.close()
    
if __name__ == '__main__':
    main()
