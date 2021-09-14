## Markov Chain Monte Carlo Algorithm on geniric Euclidean Manifold

### Task
The goal of this project is to apply Monte Carlo methods on manifolds in the Euclidean space defined
by equality and inequality constraints. In particular, this project constructs an MCMC
sampler for probability distributions defined by unnormalized densities on such manifolds.
Once samples have been obtained, one can use them to compute integrals over the manifold. We base ourself on the paper [Monte Carlo on Manifolds](https://arxiv.org/abs/1702.08446)

### Run locally
* Clone the repository `git clone https://github.com/AlicKaufmann/MCMC_on_manifolds`
* Run torus.py for sampling uniformly at random from the tous
* Run SOn.py for sampling uniformly at random from the Special Orthogonal group SO(11) and display the distribution of the trace operator of the matrices that were sampled.
