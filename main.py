from model import TBHCNN
import tensorflow as tf 
import numpy as np

# we take the example of the 13-atom-wide InSe


# Energy bands data as references 
references = np.load("./data/input/InSe Nanoribbon/InSe-references.npy")
kvectors = np.load("./data/input/InSe Nanoribbon/InSe-kpoints.npy")

# The lattice vector of the real-space Hamiltonian considered 
# The opposite vectors cannot appear at same time but should only choose one
# e.g. both np.array([[0,0,0],[0,0,1],]) and np.array([[0,0,0],[0,0,-1],]) correspond to H(R=[0,0,0]), H(R=[0,0,1]) and H(R=[0,0,-1]) 
# while np.array([[0,0,0],[0,0,1],[0,0,-1]]) are not supported
rvectors_without_opposite = np.array([[0,0,0],[0,0,1],])

#Hyperparameters
Optimizer = tf.train.AdamOptimizer(0.001)
thrould = 1e-5
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
    while (sess.run(loss)>loss_threshold) and (train_steps <= max_train_steps):
        sess.run(train)
        train_steps += 1
        if train_steps%1000 == 1:
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
    
    finished = fitting(Optimizer, thrould , max_training_steps, tbhcnn)
    
    while not finished:
        tbhcnn.H_size_added += basis_added_step
        tbhcnn.reinitialize()
        finished = fitting(Optimizer, thrould , max_training_steps, tbhcnn)
        
    Resulting_Hamiltonian = []
    for i in finished:
        Resulting_Hamiltonian.append(sess.run(i))
    Resulting_Hamiltonian = np.array(Resulting_Hamiltonian).real
    
    np.save("./data/output/InSe Nanoribbon/resulting_real_space_hamiltonians.npy", Resulting_Hamiltonian)
    np.save("./data/output/InSe Nanoribbon/rvectors_of_the_resulting_hamiltonian.npy", sess.run(tbhcnn.R))
    np.save("./data/output/InSe Nanoribbon/reproduced_TB_bands.npy", sess.run(tbhcnn.reproduces))
    np.save("./data/output/InSe Nanoribbon/TB_bandstructure.npy", sess.run(tbhcnn.wholebandstructure))
    
    print("final loss value: %.8f" % sess.run(tf.reduce_mean(tf.square(tbhcnn.reproduces-tbhcnn.references))))

if __name__ == '__main__':
    main()