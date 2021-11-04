# DHCN

Codes for AAAI 2021 paper 'Self-Supervised Hypergraph Convolutional Networks for Session-based Recommendation'.

### Please note that the default link of our paper in google scholar links to an obsolete version with incorrect experimental results. 
### The latest version of our paper is available at: 
https://ojs.aaai.org/index.php/AAAI/article/view/16578

Environments: Python3, Pytorch 1.6.0, Numpy 1.18.1, numba

Datasets are available at Dropbox: https://www.dropbox.com/sh/j12um64gsig5wqk/AAD4Vov6hUGwbLoVxh3wASg_a?dl=0 The datasets are already preprocessed and encoded by pickle.

For Diginetica, the best beta value is 0.01; for Tmall, the best beta value is 0.02.

Some people may encounter a cudaError in line 50 or line 74 when running our codes if your numpy and pytorch version are different with ours. Currently, we haven't found the solution to resolve the version problem. If you have this problem, please try to change numpy and pytorch version same with ours.
