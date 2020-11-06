import tensorflow as tf
import numpy as np

class TBHCNN():
    """
    Tight-binding Hamiltonian construction neural network
    """
    def __init__(self):
        """
        initialize TBHCNN class
        """
        super(TBHCNN, self).__init__()
        self.H_size_added = 0
        
    def read_training_set(self, references, kvectors):
        """
        read ab-inirio reference band data (energies and k-vectors)
        
        numk: number of the k points per referebce band
        numb: number of the reference bands
        
        Parameters
        ----------

        references: Variable(np.array) shape (numk, numb)
          energy bands to be fitted
        kvectors: Variable(np.array) shape (numk, 3)
          k vectors of the band data
        """
        
        numk, numb = references.shape
        
        self.numb = numb
        self.numk = numk
        self.H_size_init = numb
        self.H_size = numb
        self.references = tf.constant(references, tf.float64)
        self.K = tf.constant(kvectors, tf.complex64)
    
    def define_TB_representation(self, rvectors_without_opposite):
        """
        define the TB model desired by select the considered lattice vectors
        
        numr: number of the real-space Hamiltonians desired to be include 
        in the TB model in practice 
        
        Parameters
        ----------
        rvectors_without_opposite: Variable(int np.array) shape ((numr+1)/2, 3)
          lattice vectors of the real-space Hamiltonians without the opposite 
          vectores appearing at the same time.
          
        """
        len_rvector = rvectors_without_opposite.shape[0]
        numr = 2 * len_rvector - 1
        self.R = np.array([])
        
        for i in rvectors_without_opposite:
            self.R = np.append(self.R, i)
            if np.sum(np.abs(i)) != 0:
                self.R = np.append(self.R, -1*i)
                
        self.numr = numr        
        self.R = self.R.reshape(-1,3)
        self.R = tf.cast(self.R, tf.complex64)
        self.R_without_opposite = rvectors_without_opposite

    def reinitialize(self):
        """
        initialize/reinitialize the real-space Hamiltonian matrices
    
        """
        self.HR = []
        name = locals()
        self.H_size = self.H_size_init + self.H_size_added
        
        for i in range(self.R_without_opposite.shape[0]):
            R = ""
            R_opposite = ""
            for j in range(3):
                R += str(self.R_without_opposite[i,j])
                R_opposite += str(-1 * self.R_without_opposite[i,j])
            R = R.replace("-","m")
            R_opposite = R_opposite.replace("-","m")
            name['H%s'%R] = tf.cast(tf.Variable(tf.truncated_normal([self.H_size, self.H_size], mean=0.0, stddev=1.0,dtype=tf.float64)), dtype=tf.complex64)
            
            name['H%s'%R_opposite] = tf.cast(tf.transpose(name['H%s'%R]), dtype=tf.complex64)
            if R =="000":
                self.HR.append(name['H%s'%R] + tf.transpose(name['H%s'%R]))
            else:
                self.HR.append(name['H%s'%R])
                self.HR.append(name['H%s'%R_opposite])
                
    def compute_bands(self):
        """
        Using the real-space Hamiltonians considered to compute the k-space 
        Hamiltonians to compute the energy bands
        
        """
        reproduces = tf.zeros([self.numk,self.H_size], dtype=tf.float64)
        for i in range(self.numk):
            HK = tf.zeros([self.H_size, self.H_size], dtype = tf.complex64)
            for j in range(self.numr):
                HK += tf.scalar_mul(tf.exp(1j*tf.reduce_sum(self.K[i]*self.R[j])),self.HR[j])
                
            e = tf.self_adjoint_eigvals(HK)
            e = tf.cast(e, tf.float64)
            e = tf.reshape(e,[1,self.H_size])
            reproduces = reproduces + tf.scatter_nd([[i]], e, [self.numk,self.H_size])
        
        # the added bands will be placed above and/or below the references and 
        # will not be used for loss function computation
        
        self.wholebandstructure = reproduces
        
        reproduces = reproduces[:, int(self.H_size_added/2):int(self.H_size_added//2)+self.numb]
         
        self.reproduces = reproduces
        
        return reproduces

class Variation1():
    """
    Variation1 of the tight-binding Hamiltonian construction neural network
    """
    def __init__(self):
        """
        initialize 
        """
        super(Variation1, self).__init__()
        self.H_size_added = 0
        
    def read_training_set(self, references, kvectors, template_hamiltonians, rvectors_without_opposite):
        """
        read ab-inirio reference band data (energies and k-vectors) and the real-space 
        TB Hamiltonian matrices to be optimized
        
        numk: number of the k points per referebce band
        numb: number of the reference bands
        numh: size of the template real-space Hamiltonians
        numr: number of the lattice vector considered without opposite
        
        Parameters
        ----------

        references: Variable(np.array) shape (numk, numb)
          energy bands to be fitted
        kvectors: Variable(np.array) shape (numk, 3)
          k vectors of the band data
        template_hamiltonians: Variable(np.array) shape ((numr+1)/2, numh, numh)
          TB Hamiltonians to be optimized
        rvectors_without_opposite: Variable(int np.array) shape ((numr+1)/2, 3)
          lattice vectors of the real-space Hamiltonians without the opposite 
          vectores appearing at the same time.
        """
        
        numk, numb = references.shape
        numr = 2 * rvectors_without_opposite.shape[0] - 1
        
        self.numr = numr  
        self.numb = numb
        self.numk = numk
        self.H_size = template_hamiltonians.shape[1]
        self.references = tf.constant(references, tf.float64)
        self.K = tf.constant(kvectors, tf.complex64)
        self.template = template_hamiltonians
        
        self.R = np.array([])
        for i in rvectors_without_opposite:
            self.R = np.append(self.R, i)
            if np.sum(np.abs(i)) != 0:
                self.R = np.append(self.R, -1*i)
                
        self.R = self.R.reshape(-1,3)
        self.R = tf.cast(self.R, tf.complex64)
        self.R_without_opposite = rvectors_without_opposite

    def initialize(self):
        """
        initializethe real-space Hamiltonian matrices with the provided Hamiltonians
    
        """
        self.HR = []
        name = locals()
        
        for i in range(self.R_without_opposite.shape[0]):
            R = ""
            R_opposite = ""
            for j in range(3):
                R += str(self.R_without_opposite[i,j])
                R_opposite += str(-1 * self.R_without_opposite[i,j])
            R = R.replace("-","m")
            R_opposite = R_opposite.replace("-","m")
            name['H%s'%R] = tf.cast(tf.Variable(self.template[i,:,:]), dtype=tf.complex64)
            
            name['H%s'%R_opposite] = tf.cast(tf.transpose(name['H%s'%R]), dtype=tf.complex64)
            if R =="000":
                self.HR.append(tf.linalg.band_part(name['H%s'%R],-1,0) + tf.transpose(tf.linalg.band_part(name['H%s'%R],-1,0)) - tf.linalg.tensor_diag_part(name['H%s'%R]))
            else:
                self.HR.append(name['H%s'%R])
                self.HR.append(name['H%s'%R_opposite])
                
    def compute_bands(self, indices):
        """
        Using the real-space Hamiltonians considered to compute the k-space 
        Hamiltonians to compute the energy bands
        
        Parameters
        ----------

        indices: Variable(list) shape (1,2)
          point out the band indices of the references bands in the whole band structure
        
        """
        reproduces = tf.zeros([self.numk,self.H_size], dtype=tf.float64)
        for i in range(self.numk):
            HK = tf.zeros([self.H_size, self.H_size], dtype = tf.complex64)
            for j in range(self.numr):
                HK += tf.scalar_mul(tf.exp(1j*tf.reduce_sum(self.K[i]*self.R[j])),self.HR[j])
                
            e = tf.self_adjoint_eigvals(HK)
            e = tf.cast(e, tf.float64)
            e = tf.reshape(e,[1,self.H_size])
            reproduces = reproduces + tf.scatter_nd([[i]], e, [self.numk,self.H_size])
        
        # the added bands will be placed above and/or below the references and 
        # will not be used for loss function computation
        
        reproduces = reproduces[:, indices[0]:indices[1]]
         
        self.reproduces = reproduces
        
        return reproduces