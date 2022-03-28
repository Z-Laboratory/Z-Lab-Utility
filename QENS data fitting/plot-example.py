import os
import sys
import numpy as np
import matplotlib.pyplot as plt

#use command "python plot-example.py <path to where fitting result file is located>"
#for example python plot-example.py 773k/0.5mev-test/fitting_results_FLiNaK_773K_3.32meV.txt

fitted_parameters_ = []
fitted_parameters_error_ = []
parameters_name_ = []
q_ = []
chi2_ = []
with open(sys.argv[1], "r") as fin:
    for aline in fin:
        if "Group#:" in aline:
            q_.append(float(aline.strip().split()[-2]))
            linelist = fin.readline().strip().split()[1:]
            fitted_parameters_.append([float(i) for i in linelist[:len(linelist)//2]])
            fitted_parameters_error_.append([float(i) for i in linelist[len(linelist)//2:]])
            linelist = fin.readline().strip().split()
            chi2_.append(float(linelist[-1]))
        elif "Fitting Results" in aline:
            linelist = fin.readline().strip().split()[1:]
            parameters_name_ = linelist[:len(linelist)//2]
            
q_ = np.array(q_)
fitted_parameters_ = np.array(fitted_parameters_)
fitted_parameters_error_ = np.array(fitted_parameters_error_)

target_parameter_index = parameters_name_.index("beta")
beta_ = fitted_parameters_[:,target_parameter_index]
beta_e = fitted_parameters_error_[:,target_parameter_index]
from scipy.special import gamma
target_parameter_index = parameters_name_.index("tau")
tau_ = fitted_parameters_[:,target_parameter_index]
tau_e = fitted_parameters_error_[:,target_parameter_index]
tau_avg = tau_/beta_ * gamma(1.0/beta_)

tickfontsize = 12
fig, ax = plt.subplots(2, 1)
plt.subplots_adjust(left=0.15, right=0.9, bottom=0.15, top=0.9, wspace=0.0, hspace=0.0)
ax[0].errorbar(q_, beta_, beta_e, fmt='-o', lw=2, elinewidth=2, capsize=3, label=r'$\beta$')
ax[0].set_ylabel(r'$\beta$', fontsize=tickfontsize)
ax[0].set_ylim((0.01,1.1))
ax[1].errorbar(q_, tau_, tau_e, fmt='-o', lw=2, elinewidth=2, capsize=3, label = r'$\tau_{KWW}$')
ax[1].plot(q_, tau_avg, 'o-', label =r'$\langle\tau\rangle=\frac{\tau_{KWW}}{\beta}\Gamma (\frac{1}{\beta})$')
ax[1].set_xlabel(r'$Q\ \mathrm{(\AA^{-1})}$', fontsize=tickfontsize)
ax[1].set_ylabel(r'$\tau$', fontsize=16)
ax[1].set_yscale('log')

ax[0].tick_params(axis="x",which ="major",length=9,width=2,labelsize=tickfontsize, pad=2)
ax[0].tick_params(axis="y",which ="major",length=9,width=2,labelsize=tickfontsize, pad=2)
ax[0].tick_params(axis="x",which ="minor",length=6,width=2,labelsize=tickfontsize, pad=2)
ax[0].tick_params(axis="y",which ="minor",length=6,width=2,labelsize=tickfontsize, pad=2)
ax[1].tick_params(axis="x",which ="major",length=9,width=2,labelsize=tickfontsize, pad=2)
ax[1].tick_params(axis="y",which ="major",length=9,width=2,labelsize=tickfontsize, pad=2)
ax[1].tick_params(axis="x",which ="minor",length=6,width=2,labelsize=tickfontsize, pad=2)
ax[1].tick_params(axis="y",which ="minor",length=6,width=2,labelsize=tickfontsize, pad=2)

ax[0].legend()
ax[1].legend()
plt.show()