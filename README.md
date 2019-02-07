# Hyperparameters Optimization
The aim of this project is to compare two algorithms for hyperparameters global optimization, one based on Radial Basis Functions and the other based on Bayesian gaussian processes. 
## System requirements
+ download Bonmin, Ipopt and other binaries: from `https://ampl.com/dl/open/` download the binaries for your operating system
+ extract the binaries, and add extracted folder path to your `$PATH` environment variable
+ install pytorch
+ install these Python 3 packages too: rbfopt, bayesian-optimization, tensorboardX
## Usage
+ download the project:
`git clone https://github.com/emanuelevivoli/Hyperparameters_Optimization.git`
+ enter in the project directory:
`cd Hyperparameters_Optimization`
+ (modify and) run evaluation.py:
`python3 evaluate.py`
