import QENSFit as QF

# remember to change file name!

# example code for fitting resolution function
grpfilename = "<name of your resolution data file>.grp"
grpfilename = "Resolution_file_265C_3.32meV_FLiNaK.grp"
res = QF.ResolutionDataModel(grpfilename = grpfilename, data_range = 0.4, neutron_e0 = 3.32)
res.fit()
res.output_results() #default output file: fitting_results_ < name of your resolution data file > .txt
res.plot_results()
res.plot_results(log_scale = True)

# example code for fitting QENS Data
grpfilename = "<name of your QENS data file>.grp"
grpfilename = "FLiNaK_1073K_3.32meV.grp"
resolution_parameter_filename = "fitting_results_<name of your resolution data file>.txt"
resolution_parameter_filename = "fitting_results_Resolution_file_265C_3.32meV_FLiNaK.txt"
qens = QF.QENSDataModel(grpfilename = grpfilename, resolution_parameter_filename = resolution_parameter_filename, neutron_e0 = 3.32, data_range = 0.4)
qens.fit(fix_elastic_contribution = 0.1)
qens.output_results()
qens.plot_results()
qens.plot_results(log_scale = True)
