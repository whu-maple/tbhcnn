from model import TBHCNN
import tensorflow as tf 
import numpy as np

# we take the example of the 13-atom-wide InSe


# Energy bands data as references 
references = np.load("./data/input/InSe Nanoribbon/InSe-references.npy")
kvectors = np.load("./data/input/InSe Nanoribbon/InSe-kpoints.npy")
# Here, to ensure that the Hamiltonians with the opposite lattice vectors tranpose each other
# only one of them with the corresponding lattice vector should be included in the templates and rvectors, the other one will be handled automatically
# e.g. both np.array([[0,0,0],[0,0,1],]) and np.array([[0,0,0],[0,0,-1],]) represent the lattice vector set np.array([[0,0,0],[0,0,1],[0,0,-1]])
rvectors_without_opposite = np.array([[0,0,0],[0,0,1],])

#Hyperparameters
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
        
    Resulting_Hamiltonian = sess.run(tf.cast(finished, tf.float64))
            
    np.save("./data/output/InSe Nanoribbon/1resulting_real_space_hamiltonians.npy", Resulting_Hamiltonian)
    np.save("./data/output/InSe Nanoribbon/1rvectors_of_the_resulting_hamiltonian.npy", sess.run(tbhcnn.R))
    np.save("./data/output/InSe Nanoribbon/1reproduced_TB_bands.npy", sess.run(tbhcnn.reproduces))
    np.save("./data/output/InSe Nanoribbon/1TB_bandstructure.npy", sess.run(tbhcnn.wholebandstructure))
    
    print("final loss value: %.8f" % sess.run(tf.reduce_mean(tf.square(tbhcnn.reproduces-tbhcnn.references))))

    sess.close()
    
if __name__ == '__main__':
    main()
