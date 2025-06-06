# QFI dynamics in LMG Model
Script used to calculate dynamics of QFI in LMG model in paper https://arxiv.org/abs/2505.22731

```math
  \hat{H}(t) = -\frac{2J}{N} \hat{S}_z^2 - 2 B  \hat{S}_x
 -\pi \sum_{m=1}^{\infty} \delta(t - mT) \hat{S}_x + \hat V(t)
```

Dynamics of the QFI for different relevant initial preparations of the sensor: from low to high correlated initial states, and effective Floquet Hamiltonian eigenstates $\{|E_{i(\bar{i})}\rangle\}$. The QFI exhibits a step-like increasing/decreasing characteristic dynamics for each of these classes of preparations, on timescales proportional to their $\pi$-paired gaps. The results here are illustrated for a FTC sensor based on the LMG model ($N=40, T=1, J=1, B=0.4, T_{\rm ac} = 2T$, linear response \(h \to 0\) and sinusoidal signal.

![Plot for QFI](results/qfi_dynamics.png "QFI dynamics")


# Steps to run simulation for 4 standard states

1. Create environment using:
```
conda env create -f environment.yml
```
2. Run script 
```
./quantum_fisher_information_simulation.py
```
This will read params from file `qfi_simulation.ini`and generate results of simulation to `results/general_case.png.csv`
3. Run script to create graph based on data
```
./plot_qfi_results.py
```
