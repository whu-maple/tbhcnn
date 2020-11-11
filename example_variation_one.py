from model import Variation1
import tensorflow as tf 
import numpy as np

# we take the example of the monolayer black phosphorus


# Energy bands data as references and unoptimized Hamiltonians as templates
references = np.load("./data/input/variation1 Monolayer BP/MBP-references.npy")
kvectors = np.load("./data/input/variation1 Monolayer BP/MBP-kpoints.npy")  # in units of[1/a, 1/b, 1/c] (a, b, and c are lattice constants)
indices = [4,12] #point out the indices of the references bands within the whole band structure

# Here, to ensure that the Hamiltonians with the opposite lattice vectors are always taken into consideration simultaneously, and they tranpose each other
# only one of them with the corresponding lattice vector should be included in the provided templates and rvectors, the other one will be handled automatically
# e.g. for this system, just provide the rvectors =  np.array([[0,0,0],[0,0,1],[0,1,0],[0,1,1],[0,1,-1]]) and the templates Hamiltonians with the same order
# and in the end we shall get the 9 Hamiltonian matrices we actually used to build the tight-binding model
templates = np.load("./data/input/variation1 Monolayer BP/MBP-templates.npy")
rvectors = np.load("./data/input/variation1 Monolayer BP/MBP-latticevectors_of_templates.npy")  # in units of[a, b, c] (a, b, and c are lattice constants)

#Hyperparameters
Optimizer = tf.train.AdamOptimizer(0.001)
lamda = 1/1000
threshold = 1e-6
max_steps = 10000 # the max training steps, when surpassing it, end the training anyway

sess = tf.Session()

def fitting(Optimizer, loss_threshold, max_training_steps, variation1, bands_indices, lamda):
    reproduces = variation1.compute_bands(bands_indices) 
    loss1 = tf.reduce_mean(tf.square(reproduces-variation1.references))
    loss2 = tf.cast(tf.reduce_mean(tf.square(variation1.HR_init - variation1.HR)), tf.float64)
    loss = loss1 + lamda * loss2
    train = Optimizer.minimize(loss)
    init = tf.global_variables_initializer()
    sess.run (init)
    train_steps = 0
    print("initial loss value: %.8f" % (sess.run(loss)))
    while (sess.run(loss)>loss_threshold) and (train_steps<=max_steps):
        sess.run(train)
        train_steps += 1
        if train_steps%1000 == 0:
            print("steps:%s, loss:%.8f" % (train_steps, sess.run(loss)))
    return variation1.HR
    
def main():
    variation1 = Variation1()
    variation1.read_training_set(references, kvectors, templates, rvectors)
    variation1.initialize()
    
    # output the optimized real-space Hamiltonians, their corresponding lattice vectors, their computed 
    # bandstructure, and their reproduction of the reference bands

    Resulting_Hamiltonian = fitting(Optimizer, threshold, max_steps, variation1, indices, lamda)
    Resulting_Hamiltonian = sess.run(tf.cast(Resulting_Hamiltonian, tf.float64))
    Rvectors_of_the_resulting_hamiltonian = sess.run(tf.cast(variation1.R, tf.int32))
    Reproduced_TB_bandstructure = sess.run(variation1.wholebandstructure)
    Reproduced_TB_bands = sess.run(variation1.reproduces)
            
    np.save("./data/output/variation1 Monolayer BP/resulting_real_space_hamiltonians.npy", Resulting_Hamiltonian)
    np.save("./data/output/variation1 Monolayer BP/rvectors_of_the_resulting_hamiltonian.npy", Rvectors_of_the_resulting_hamiltonian)
    np.save("./data/output/variation1 Monolayer BP/TB_bandstructure.npy", Reproduced_TB_bandstructure)
    np.save("./data/output/variation1 Monolayer BP/reproduced_TB_bands.npy", Reproduced_TB_bands)
    
    print("final mean square error for band reproduction : %.8f" % sess.run(tf.reduce_mean(tf.square(variation1.reproduces-variation1.references))))
    print("final mean absolute deviation for Hamiltonian elements : %.8f" % sess.run(tf.cast(tf.reduce_mean(tf.abs(variation1.HR_init - variation1.HR)), tf.float64)))

    sess.close()
    
if __name__ == '__main__':
    main()
