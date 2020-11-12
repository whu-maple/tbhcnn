# tbhcnn
Prototype code of the tight-binding hamiltonian construction neural network model. Note that it is not necessarily high-performance and well-optimized since it is a prototype code for research work, however, 

there will be continuous updates later to add more examples and optimize the code.

# requirements
tensorflow1.x （1.15）
numpy （1.16.0）

# example
Check the example_basic_method.py to construct a TB model for the InSe nanoribbon. (basic TBHCNN model in the manuscrippt)

Check the example_variation_one.py to optimize a given Wannier TB model for 2-D black phosphorus. (Variation1 in the manuscrippt)

Check the example_variation_two.py to construct a TB model in Slater-Koster form for 13-atom-wide armchair graphene nanoribbon (13-AGNR). (Variation2 in the manuscrippt)

[Note that the Variation2 code in model.py now is suitble specifically for the 13-AGNR system because it uses directly the fixed fitting formula for 13-AGNR. There will be updates later to generalize this 

class.]

These scripts are self-explanatory and can be easily understood with the manuscript. With the provided default parameter settings, they can be run directly to obtain the corresponding results.

# data
in ./data/input file folder we provide the reference bands data for training the TBHCNN for InSe nanoribbon, Si of the diamond structure, and GaN of the wurtzite structure. Also, we provide the reference 

bands data and unoptimized Wannier TB model as a template for training the Variation1 of TBHCNN for 2-D black phosphorus, and we provide the reference bands data and coordinate file (xyz file) for training 

the Variation2 of TBHCNN for 13-AGNR.

And in ./data/output file folder we store their corresponding result files for the above mentioned systems.
