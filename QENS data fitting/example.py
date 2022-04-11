import os
import sys
import QENSFit as QF

# example code for fitting resolution function
res_output_dir = "3mev-resolution"
grpfilename = "Resolution_file_265C_3.32meV_FLiNaK.grp"
res = QF.ResolutionDataModel(grpfilename = grpfilename, data_range = 3, neutron_e0 = 3.32, max_n_gauss = 4, mirror='left')
res.fit(weighted_with_error = True)
res.output_results(output_dir = res_output_dir) #default output file: fitting_results_Resolution_file_265C_3.32meV_FLiNaK.txt
res.plot_results(output_dir = res_output_dir, show_errorbar = True)
res.plot_results(output_dir = res_output_dir, log_scale=True, show_errorbar = True)

# example code for fitting QENS Data
qens_output_dir = "773k/3mev-fixall"
grpfilename = "FLiNaK_773K_3.32meV.grp"
resolution_parameter_filename = res_output_dir + "/fitting_results_Resolution_file_265C_3.32meV_FLiNaK.txt"
qens = QF.QENSDataModel(grpfilename = grpfilename, resolution_parameter_filename = resolution_parameter_filename, data_range = 1, neutron_e0 = 3.32)
qens.fit(const_f_elastic = 0.0, const_f_fast = 0.1, const_tau_fast = 2, const_beta = 1)
qens.output_results(output_dir = qens_output_dir) #default output file: fitting_results_FLiNaK_1073K_3.32meV.txt
qens.plot_results(output_dir = qens_output_dir, show_errorbar = False)
qens.plot_results(output_dir = qens_output_dir, log_scale = True, show_errorbar = False)
