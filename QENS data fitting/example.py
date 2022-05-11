import os
import sys
import QENSFit as QF

# example code for fitting resolution function
res_output_dir = "0.3mev-resolution"
grpfilename = "Resolution_file_265C_3.32meV_FLiNaK.grp"
res = QF.ResolutionDataModel(grpfilename = grpfilename, data_range = (-3,1), neutron_e0 = 3.32, max_n_gauss = 4, mirror = 'off')
res.fit(weighted_with_error = True)
res.output_results(output_dir = res_output_dir) #default output file: fitting_results_Resolution_file_265C_3.32meV_FLiNaK.txt
res.plot_results(output_dir = res_output_dir, show_errorbar = True)
res.plot_results(output_dir = res_output_dir, log_scale=True, show_errorbar = True)

# example code for fitting QENS Data
qens_output_dir = "773k/0.18mev-fixall"
grpfilename = "FLiNaK_773K_3.32meV.grp"
resolution_parameter_filename = res_output_dir + "/fitting_results_Resolution_file_265C_3.32meV_FLiNaK.txt"
qens = QF.QENSDataModel(grpfilename = grpfilename, resolution_parameter_filename = resolution_parameter_filename, \
data_range = [(-5,2),\
              (-4,2),\
              (-5,2),\
              (-4,2),\
              (-5,2),\
              (-4,2),\
              (-5,2),\
              (-4,2),\
              (-5,2),\
              (-4,2),\
              (-5,2)], neutron_e0 = 3.32)
#use_previous_q_as_initial_guess = an array of True or False [A,   f_elastic,  f_fast,  tau_fast,     tau , beta,  E_center,  background]
qens.fit(const_f_elastic = [0.0,0.1,0.2,0.0,0.1,0.2,0.0,0.1,0.2,0.0,0.1],\
         const_f_fast = 0.1, const_tau_fast = 2,\
         initial_A=[0.9,0.8,0.9,0.9,0.8,0.9,0.9,0.8,0.9,0.9,0.8],\
         use_previous_q_as_initial_guess=[False,      True,   True,     True,   True, True,    True,        True])
qens.output_results(output_dir = qens_output_dir) #default output file: fitting_results_FLiNaK_1073K_3.32meV.txt
qens.plot_results(output_dir = qens_output_dir, show_errorbar = True)
qens.plot_results(output_dir = qens_output_dir, log_scale = True, show_errorbar = True)
