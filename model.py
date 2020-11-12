import tensorflow as tf
import numpy as np
import os 
import re

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
        define the TB model desired by selecting the considered lattice vectors
        
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
        self.H_size = self.H_size_init + self.H_size_added
        
        for i in self.R_without_opposite:
            H_tmp = tf.cast(tf.Variable(tf.truncated_normal([self.H_size, self.H_size], mean=0.0, stddev=1.0,dtype=tf.float64)), dtype=tf.complex64)
            
            # Once we take into consideration the real-space Hamiltonian H(R=Ri), 
            # we must consider H(R=-Ri) also, which is the tranposed matrix of 
            # H(R=Ri), to ensure that calculated reciprocal-space Hamiltonian maintains Hermitian
            if np.sum(np.abs(i)) != 0:
                self.HR.append(H_tmp)
                self.HR.append(tf.transpose(H_tmp))
            else:
                self.HR.append(H_tmp + tf.transpose(H_tmp)) #ensure that H(R=[0,0,0]) maintains symmetric  
                
        self.HR = tf.cast(self.HR, tf.complex64)
                
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
         
        self.wholebandstructure = reproduces
        
        # the added bands will be placed above and/or below the references, 
        # and they will not be used for loss function computation
        reproduces = reproduces[:, int(self.H_size_added/2):int(self.H_size_added//2)+self.numb]
         
        self.reproduces = reproduces
        
        return reproduces

class Variation1():
    """
    Variation1 of the tight-binding Hamiltonian construction neural network
    for optimizing a given TB model
    """
    def __init__(self):
        """
        initialize 
        """
        super(Variation1, self).__init__()
        
    def read_training_set(self, references, kvectors, template_hamiltonians, rvectors_without_opposite):
        """
        read ab-inirio reference band data (energies and k-vectors) and the real-space 
        TB Hamiltonian matrices to be optimized
        
        numk: number of the k points per referebce band
        numb: number of the reference bands
        numh: size of the template real-space Hamiltonians
        numr: number of the real-space Hamiltonians desired to be include 
        in the TB model in practice 
        
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
        self.templates = template_hamiltonians
        
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
        initializethe real-space Hamiltonian matrices with the template Hamiltonians
    
        """
        self.HR = []
        self.HR_init = []
        
        for i in range(self.R_without_opposite.shape[0]):
            H_trainable = tf.cast(tf.Variable(self.templates[i,:,:]), dtype=tf.complex64)
            H_template = tf.cast(tf.constant(self.templates[i,:,:]), dtype=tf.complex64)
            
            # Once we take into consideration the real-space Hamiltonian H(R=Ri), 
            # we must consider H(R=-Ri) also, which is the tranposed matrix of 
            # H(R=Ri), to ensure that calculated reciprocal-space Hamiltonian maintains Hermitian 
            if np.sum(np.abs(self.R_without_opposite[i])) != 0:
                self.HR.append(H_trainable)
                self.HR_init.append(H_template)
                self.HR.append(tf.transpose(H_trainable))
                self.HR_init.append(tf.transpose(H_template))
            else:
                
                # ensure that H(R=[0,0,0]) is symmetric and initilized with the matrix element values of the template 
                self.HR.append(tf.linalg.band_part(H_trainable,-1,0) + tf.linalg.band_part(H_trainable,0,-1) - tf.linalg.band_part(H_trainable,0,0))
                self.HR_init.append(H_template)
    
        self.HR = tf.cast(self.HR, tf.complex64)
        self.HR_init = tf.cast(self.HR_init, tf.complex64)
            
                
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
        
        self.wholebandstructure = reproduces
        
        reproduces = reproduces[:, indices[0]:indices[1]]
         
        self.reproduces = reproduces
        
        return reproduces

class Variation2():
    """
    Variation2 of the tight-binding Hamiltonian construction neural network
    for constructing the tight-binding Hamiltonians of 13-AGNR (consider pz
    orbitals of C atoms only )in Slater-Koster form
    
    Note that this class now is suitble exclusively for 13-AGNR, because it 
    directly uses the fixed fitting formula for 13-AGNR. There will be
    updates later to generalize this class
    
    """
    def __init__(self):
        """
        initialize 
        """
        super(Variation2, self).__init__()
        
    def read_geometry(self, xyz_file, lattice, rvectors_without_opposite):
        """
        define the TB model desired by selecting the considered lattice vectors
        of the real-space Hamiltonians and use the coordinate information of the
        orbitals within these lattice to obtain the distances between them  
         
        numr: number of the real-space Hamiltonians desired to be include 
        in the TB model in practice
        
        numa: number of the C atoms within a unit cell
           
        Parameters
        ----------
        xyz_file: 
          The standard xyz file of the studied system consisting of the 
          coordinate information
        
        lattice: Variable(np.array) shape (3, 3)
          point out the lattice parameters of the studied system
           
        rvectors_without_opposite: Variable(int np.array) shape ((numr+1)/2, 3)
          lattice vectors of the real-space Hamiltonians without the opposite 
          vectores appearing at the same time.
          
        """
        open_file = open(xyz_file, 'r')
        data = (open_file.readlines())[2:]
        open_file.close()
        
        coordinates = []
        
        for line in data:
            element = line[0]
            coordinate = np.array(list(map(float,re.findall(r"\-?\d+\.?\d*e?\-?\+?\d*",line))))
            if element == "C":
                coordinates.append(coordinate)
                
        self.coordinates = np.array(coordinates)
        numa = self.coordinates.shape[0]
        
        self.numa = int(numa)

        len_rvector = rvectors_without_opposite.shape[0]
        numr = 2 * len_rvector - 1
        self.R = np.array([])
        for i in rvectors_without_opposite:
            self.R = np.append(self.R, i)
            if np.sum(np.abs(i)) != 0:
                self.R = np.append(self.R, -1*i)
                
        self.numr = numr        
        self.R = self.R.reshape(-1,3)
        self.R_without_opposite = rvectors_without_opposite
        
        distance_matrices = np.zeros((numr, numa, numa), np.float64)
        for i in range(numr):
            rvector_tmp = self.R[i]
            coordinate_list_tmp = self.coordinates + np.sum(rvector_tmp * lattice,0)
            for j in range(numa):
                coordinate_tmp = coordinate_list_tmp[j]
                for k in range(numa):
                    coordinate = self.coordinates[k]
                    distance = np.linalg.norm(coordinate_tmp - coordinate)
                    distance_matrices[i,j,k] = distance
                    
        self.distance_matrices = distance_matrices
        
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
                       
        
    def initialize(self):
        """
        initializethe real-space Hamiltonian matrices with the fixed fitting 
        form, which is the function of distances between two C atoms
        
        fitting form: 
        
        Vpp(r) = a1 * r^(-a2) * exp(-a3 *  r^(a4))
    
        """
        
        #initialize a1 a2 a3 and a4
        a1 = tf.Variable(tf.truncated_normal([1], mean=-1.0, stddev=0.1, dtype=tf.float64))
        a2 = tf.Variable(tf.truncated_normal([1], mean=1.0, stddev=0.1, dtype=tf.float64))
        a3 = tf.Variable(tf.truncated_normal([1], mean=1.0, stddev=0.1, dtype=tf.float64))
        a4 = tf.Variable(tf.truncated_normal([1], mean=1.0, stddev=0.1, dtype=tf.float64))
        
        HR = []
        self.H_size = self.numa
        
        for i in range(self.R.shape[0]):
            r = tf.constant(self.distance_matrices[i], tf.float64)
            if np.sum(np.abs(i)) != 0:  
                HR_tmp = a1 * tf.pow(r, a2) * tf.exp(-1 * a3 * tf.pow(r, a4))
            else:
                r = r + tf.eye(self.numa, dtype = tf.float64)
                onsites =  tf.Variable(tf.truncated_normal([1], mean=0.0, stddev=0.0, dtype=tf.float64)) + 0.2 * tf.tanh(tf.Variable(tf.truncated_normal([7], mean=0.0, stddev=0.0, dtype=tf.float64)))
                o1,o2,o3,o4,o5,o6,o7 = onsites[0],onsites[1], onsites[2], onsites[3], onsites[4], onsites[5], onsites[6]
                onsite_matrix = tf.matrix_diag([o1,o2,o1,o3,o4,o2,o3,o5,o6,o4,o5,o7,o6,o6,o7,o5,o4,o6,o5,o3,o2,o4,o3,o1,o2,o1])
                HR_tmp = a1 * tf.pow(r, a2) * tf.exp(-1 * a3 * tf.pow(r, a4))
                HR_tmp = HR_tmp - tf.linalg.band_part(HR_tmp,0,0) + onsite_matrix
            
            HR.append(HR_tmp)
            
        self.HR = tf.cast(HR, tf.complex64)

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
        self.R = tf.cast(self.R, dtype = tf.complex64)
        for i in range(self.numk):
            HK = tf.zeros([self.H_size, self.H_size], dtype = tf.complex64)
            for j in range(self.numr):
                HK += tf.scalar_mul(tf.exp(1j*tf.reduce_sum(self.K[i]*self.R[j])),self.HR[j])
                
            e = tf.self_adjoint_eigvals(HK)
            e = tf.cast(e, tf.float64)
            e = tf.reshape(e,[1,self.H_size])
            reproduces = reproduces + tf.scatter_nd([[i]], e, [self.numk,self.H_size])
        
        self.wholebandstructure = reproduces
        
        reproduces = reproduces[:, indices[0]:indices[1]]
         
        self.reproduces = reproduces
        
        return reproduces
