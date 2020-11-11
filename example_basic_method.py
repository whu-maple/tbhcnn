from model import TBHCNN
import tensorflow as tf 
import numpy as np

# we take the example of the 13-atom-wide InSe


# Energy bands data as references 
references = np.load("./data/input/InSe Nanoribbon/InSe-references.npy")
kvectors = np.load("./data/input/InSe Nanoribbon/InSe-kpoints.npy") # in units of[1/a, 1/b, 1/c] (a, b, and c are lattice constants)

# Here, to ensure that the Hamiltonians with the opposite lattice vectors are always taken into consideration simultaneously, and they tranpose each other
# only one of each pair of the opposite lattice vectors should be included in the rvectors array, the other one will be handled automatically
# e.g., for this system, both np.array([[0,0,0],[0,0,1],]) and np.array([[0,0,0],[0,0,-1],]) represent the lattice vector set np.array([[0,0,0],[0,0,1],[0,0,-1]])
# and in the end we shall get the 3 real-space Hamiltonian matrices we actually used to build the tight-binding model
rvectors_without_opposite = np.array([[0,0,0],[0,0,1],], dtype=np.int32) # in units of[a, b, c] (a, b, and c are lattice constants)

# Hyperparameters
Optimizer = tf.train.AdamOptimizer(0.001)
threshold = 1e-5
max_training_steps = 10000
basis_added_step = 2

sess = tf.Session()

def fitting(Optimizer, loss_threshold, max_train_steps, tbhcnn):
    reproduces = tbhcnn.compute_bands() 
    loss = tf.reduce_mean(tf.square(reproduces-tbhcnn.references))
    train = Optimizer.minimize(loss)
    init = tf.global_variables_initializer()
    sess.run (init)
    train_steps = 0
    print("initial loss value: %.8f" % (sess.run(loss)))
    while (sess.run(loss)>loss_threshold) and (train_steps < max_train_steps):
        sess.run(train)
        train_steps += 1
        if train_steps%1000 == 0:
            print("steps:%s, loss:%.8f" % (train_steps, sess.run(loss)))
    if sess.run(loss)>loss_threshold:
        return False
    else:
        return tbhcnn.HR
     
def main():
    tbhcnn = TBHCNN()
    tbhcnn.read_training_set(references, kvectors)
    tbhcnn.define_TB_representation(rvectors_without_opposite)
    tbhcnn.reinitialize()
    
    finished = fitting(Optimizer, threshold , max_training_steps, tbhcnn)
    
    while finished == False:
        tbhcnn.H_size_added += basis_added_step
        tbhcnn.reinitialize()
        finished = fitting(Optimizer, threshold , max_training_steps, tbhcnn)
        
    # output the trained real-space Hamiltonians, their corresponding lattice vectors, their computed 
    # bandstructure, and their reproduction of the reference bands
    Resulting_Hamiltonian = sess.run(tf.cast(finished, tf.float64))
    Rvectors_of_the_resulting_hamiltonian = sess.run(tf.cast(tbhcnn.R, tf.int32))
    Reproduced_TB_bandstructure = sess.run(tbhcnn.wholebandstructure)
    Reproduced_TB_bands = sess.run(tbhcnn.reproduces)
            
    np.save("./data/output/InSe Nanoribbon/resulting_real_space_hamiltonians.npy", Resulting_Hamiltonian)
    np.save("./data/output/InSe Nanoribbon/rvectors_of_the_resulting_hamiltonian.npy", Rvectors_of_the_resulting_hamiltonian)
    np.save("./data/output/InSe Nanoribbon/TB_bandstructure.npy", Reproduced_TB_bandstructure)
    np.save("./data/output/InSe Nanoribbon/reproduced_TB_bands.npy", Reproduced_TB_bands)
    
    print("final loss value: %.8f" % sess.run(tf.reduce_mean(tf.square(tbhcnn.reproduces-tbhcnn.references))))

    sess.close()
    
if __name__ == '__main__':
    main()
