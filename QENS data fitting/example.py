import os
import sys
import QENSFit as QF

# example code for fitting resolution function
res_output_dir = "0.3mev-resolution"
grpfilename = "resolution_qens.grp"
res = QF.ResolutionDataModel(grpfilename = grpfilename, data_range = 0.3, neutron_e0 = 3.32, max_n_gauss = 4, mirror = 'off')
res.fit(weighted_with_error = True)
res.output_results(output_dir = res_output_dir) #default output file: fitting_results_<grpfilename>.txt
res.plot_results(output_dir = res_output_dir, show_errorbar = True)
res.plot_results(output_dir = res_output_dir, log_scale=True, show_errorbar = True)

# example code for fitting QENS Data
qens_output_dir = "773k/0.3mev-test"
resolution_parameter_filename = res_output_dir + "/fitting_results_%s.txt"%(grpfilename[:-4])
grpfilename = "773k_qens.grp"
qens = QF.QENSDataModel(grpfilename = grpfilename, resolution_parameter_filename = resolution_parameter_filename, \
data_range = [(-5,2)]*7 + [(-4,2)]*3 + [(-3,2)]*10, neutron_e0 = 3.32)
qens.fit(const_f_elastic = [0.0,0.1,0.2,0.0,0.1,0.2,0.0,None,None,0.0,0.1] + [0.1]*9,\
         const_f1 = 0.1, const_tau1 = 1, const_f2 = 0.2, const_tau2 = 50,\
         const_tau = [None]*4+[100]+[None]*15,\
         initial_A = [0.9]*20,\
         use_previous_q_as_initial_guess = [False,      True, True, True, True, True, True, True,     True,       True])
#        use_previous_q_as_initial_guess = [    A, f_elastic,   f1, tau1,   f2, tau2,  tau, beta, E_center, background] an array of True or False 
qens.output_results(output_dir = qens_output_dir) #default output file: fitting_results_FLiNaK_1073K_3.32meV.txt
qens.plot_results(output_dir = qens_output_dir, show_errorbar = True)
qens.plot_results(output_dir = qens_output_dir, log_scale = True, show_errorbar = True)
