#
#  QENSFit.py
#  
#  Copyright (c) 2022 Z-Group. All rights reserved.
#  -----------------------------------------------------
#  Current developers  : Shao-Chun Lee    (2022 - Present)
#  -----------------------------------------------------

import os
import numpy as np
import matplotlib.pyplot as plt
import ezfft
import CurveFitting as CF

def gauss(E_, f, mean, sigma):
    return f * np.exp(-0.5 * ((E_-mean)/sigma)**2)/(sigma * (2.0 * np.pi)**0.5)
def kww(t_, tau, beta):
    return np.exp(-(t_/tau)**beta)
def power(x_, b, e0):
    return b * (x_ + e0)**-1.5
def constant(x_, c):
    return np.ones_like(x_) * c

class DataPlot:
    def __init__(self):
        pass
    
    def plot_clear(self):
        plt.close('all')

    def plot_save(self, plotfilename):
        plt.savefig(plotfilename, transparent = False)

    def plot(self, x_, y_, plottype = None, plottitle = "", legendtitle = "", yerr_ = [], legend_ = [], linestyle_ = None, markerstyle_ = None, markerface = True, markersize = 5, lw_ = None, color_ = None, log_fg = (False, False), xlabel = None, ylabel = None, xminortick = 2, yminortick = 4, xlim = None, ylim = None, plottitle_fontsize = 24, legend_fontsize = 24, legendtitle_fontsize = 24, label_fontsize = 24, tick_fontsize = 24, figsize = (10, 8), axsize = .7, margin_ratio = 0.5, legend_column = 1, legend_location = 0):
        default_color_list = plt.rcParams['axes.prop_cycle'].by_key()['color']
        if color_:
            color_tmp = []
            for i in color_:
                if type(i) == tuple:
                    if len(i) == 3:
                        color_tmp.append("#%02x%02x%02x"%(i))
                    elif len(i) == 4:
                        color_tmp.append(i)
                elif type(i) == int:
                    color_tmp.append(default_color_list[i])
                else:
                    color_tmp.append(i)      
            color_ = color_tmp[:]
        LegendFontSize = int(legend_fontsize)
        LabelFontSize = int(label_fontsize)
        TickFontSize = int(tick_fontsize)
        PlottitleFontSize = int(plottitle_fontsize)
        FrameWidth = 1.5
        fig = plt.figure(figsize = figsize)
        ax = fig.add_axes([margin_ratio*(1-axsize), margin_ratio*(1-axsize), axsize, axsize])
        ax.spines["top"].set_linewidth(FrameWidth)
        ax.spines["left"].set_linewidth(FrameWidth)
        ax.spines["right"].set_linewidth(FrameWidth)
        ax.spines["bottom"].set_linewidth(FrameWidth)
        for index in range(len(x_)):  
            if len(legend_) > 0 and index < len(legend_): legend = "%s"%(legend_[index])
            else: legend = ""
            if len(yerr_) == 0 or len(yerr_[index]) == 0:
                thisline = ax.plot(x_[index], y_[index], label = legend)
            else:
                thisline = ax.errorbar(x_[index], y_[index], yerr_[index], lw = 2, fmt = '-o', elinewidth = 2, capsize = 3, markersize = 5, label = legend, fillstyle = 'none', zorder = 1)  
            if lw_ is not None and index < len(lw_):
                thisline[0].set_linewidth(lw_[index])
            if linestyle_ is not None and index < len(linestyle_):
                thisline[0].set_linestyle(linestyle_[index])
            if markerstyle_ is not None and index < len(markerstyle_):
                thisline[0].set_marker(markerstyle_[index])
                thisline[0].set_markersize(markersize)
                if markerface == False:
                    thisline[0].set_markerfacecolor('none')
            if color_ is not None and index < len(color_):
                thisline[0].set_color(color_[index])
        ax.set_title(plottitle, fontsize = PlottitleFontSize)        
        ax.legend(title = legendtitle, title_fontsize = legendtitle_fontsize, fontsize = LegendFontSize, ncol = legend_column, labelspacing = 0.5, frameon = False, loc = legend_location)
        if log_fg[0] == True:    ax.set_xscale('log')
        if log_fg[1] == True:    ax.set_yscale('log')
        if xlim is not None: ax.set_xlim(xlim)
        if ylim is not None: ax.set_ylim(ylim)
        ax.tick_params(axis = "x", which = "major", length = 9, width = 2, labelsize = TickFontSize, pad = 10)
        ax.tick_params(axis = "y", which = "major", length = 9, width = 2, labelsize = TickFontSize, pad = 10)
        ax.tick_params(axis = "x", which = "minor", length = 6, width = 2, labelsize = TickFontSize, pad = 10)
        ax.tick_params(axis = "y", which = "minor", length = 6, width = 2, labelsize = TickFontSize, pad = 10)
        ax.tick_params(axis = "both", direction = "in", which = "both", top = True, right = True)
        ax.set_xlabel(xlabel, fontsize = LabelFontSize)
        ax.set_ylabel(ylabel, fontsize = LabelFontSize)

class grpFileReader:
    def __init__(self, grpfilename):
        self.grpfilename_ = grpfilename
    def read(self):
        with open(self.grpfilename_, "r") as fin:
            energy_ = []
            q_ = []
            data_ = []
            error_ = []
            fin.readline()
            n_energy = int(fin.readline())
            fin.readline()
            n_q = int(fin.readline())
            fin.readline()
            for i in range(n_energy):
                energy_.append(float(fin.readline().strip()))
            fin.readline()
            for i in range(n_q):
                q_.append(float(fin.readline().strip()))
            for i in range(n_q):
                fin.readline()
                data_.append([])
                error_.append([])
                for i in range(n_energy):
                    aline = fin.readline()
                    linelist = aline.strip().split()
                    data_[-1].append(float(linelist[0]))
                    error_[-1].append(float(linelist[1]))
            energy_ = np.array(energy_)
            q_ = np.array(q_)
            data_ = np.array(data_)
            error_ = np.array(error_)
            print("Number of energies: %s"%(n_energy))
            print("Number of qs: %s"%(n_q))
            return energy_, q_, data_, error_

class ResolutionDataModel:
    def __init__(self, grpfilename, energy_range, q_index = None, max_n_gauss = 4, neutron_e0 = None, seed = 42, background_type = 'c', mirror = 'off'):
        
        self.seed_ = seed
        np.random.seed(self.seed_)
        self.max_n_gauss_ = max_n_gauss
        self.neutron_e0_ = neutron_e0 if neutron_e0 else 'N/A'
        self.mirror_ = mirror
        self.background_type_ = background_type #background_type: 'c' = constant, 'p' = power law
        self.grpfilename_ = grpfilename

        energy_tmp, self.q_, resolution_tmp, error_tmp = grpFileReader(self.grpfilename_).read()
        self.q_index_ = [i for i in range(len(self.q_))] if q_index is None else q_index
        self.q_ = self.q_[self.q_index_]
        resolution_tmp = resolution_tmp[self.q_index_]
        error_tmp = error_tmp[self.q_index_]

        if type(energy_range) == tuple or type(energy_range) == list:
            if type(energy_range[0]) == tuple or type(energy_range[0]) == list:
                self.energy_range_ = energy_range
            else:
                self.energy_range_ = [energy_range]*len(self.q_index_)
        else:
            self.energy_range_ = [(-energy_range, energy_range)]*len(self.q_index_)

        self.energy_ = []
        self.resolution_ = []
        self.error_ = []
        for iq, q_index in enumerate(self.q_index_):
            left  = np.where(energy_tmp >=  self.energy_range_[iq][0])[0]
            mid   = np.where(energy_tmp > 0.)[0][0]
            right = np.where(energy_tmp >   self.energy_range_[iq][1])[0]
            left  = left[0]  if len(left)  > 0 else 0
            right = right[0] if len(right) > 0 else len(energy_tmp)
            if self.mirror_ == 'off':
                self.resolution_.append(resolution_tmp[iq][left:right])
                self.error_.append(error_tmp[iq][left:right])
                self.energy_.append(energy_tmp[left:right])
            elif self.mirror_ == 'left':
                self.resolution_.append(np.concatenate((resolution_tmp[iq][left:mid], np.flip(resolution_tmp[iq][left:mid], axis=1)), axis=1))
                self.error_.append(np.concatenate((error_tmp[iq][left:mid], np.flip(error_tmp[iq][left:mid], axis=1)), axis=1))
                self.energy_.append(np.concatenate((energy_tmp[left:mid], -np.flip(energy_tmp[left:mid], axis=0)), axis=0))
            elif self.mirror_ == 'right':
                self.resolution_.append(np.concatenate((np.flip(resolution_tmp[iq][mid:right], axis=1), resolution_tmp[iq][mid:right]), axis=1))
                self.error_.append(np.concatenate((np.flip(error_tmp[iq][mid:right], axis=1), error_tmp[iq][mid:right]), axis=1))
                self.energy_.append(np.concatenate((-np.flip(energy_tmp[mid:right], axis=0), energy_tmp[mid:right]), axis=0))
        
            self.error_[-1]      /= self.resolution_[-1].max()
            self.resolution_[-1] /= self.resolution_[-1].max()
        

    def R_QE_component(self, E_, *args):
        component_ = []
        for i in range(0, 3*self.max_n_gauss_, 3):
            component_.append(gauss(E_, *args[i:i+3]))
        background = constant(E_, *args[-1:]) if self.background_type_ == 'c' else power(E_, *args[-2:])
        component_ = np.array(component_)
        component_ = np.vstack((component_,background))
        return component_
    
    def R_QE(self, E_, *args):
        return np.sum(self.R_QE_component(E_, *args), axis = 0)

    def fit(self, max_fail_count = 20, weighted_with_error = True):
        self.fitted_parameters_ = []
        self.fitted_parameters_error_ = []
        self.chi2_ = []
        for iq, q_index in enumerate(self.q_index_):
            print("Now fitting group %s..."%(iq))
            resolution_q_ = self.resolution_[iq]
            error_q_ = self.error_[iq]
            energy_ = self.energy_[iq]
            true_resolution_indices = np.where(resolution_q_ > 0)
            resolution_q_ = resolution_q_[true_resolution_indices]
            error_q_ = error_q_[true_resolution_indices]
            energy_ = energy_[true_resolution_indices]

            peak_value = resolution_q_.max()
            peak_position = energy_[np.where(resolution_q_ == peak_value)[0][0]]
            fwhm_range = energy_[np.where(resolution_q_ >= peak_value/2.)[0]]
            fwhm = fwhm_range[-1]-fwhm_range[0]
            init_sigma = fwhm/2.355
            init_f = peak_value*init_sigma*(2.0*np.pi)**0.5
            init_gauss_p0 = np.array([])
            peak_position_lowerbound = energy_.min()
            peak_position_upperbound = energy_.max()
            if self.max_n_gauss_ >= 1:
                init_gauss_p0   = np.append(init_gauss_p0, [init_f*0.4  , peak_position+init_sigma*0.0 , init_sigma*1.0     ])
                lowerbound      = [0.    , peak_position_lowerbound, 0.    ]
                upperbound      = [np.inf, peak_position_upperbound, np.inf]
            if self.max_n_gauss_ >= 2:
                init_gauss_p0   = np.append(init_gauss_p0, [init_f*0.3  , peak_position-init_sigma*0.2 , init_sigma*1.5     ])
                lowerbound     += [0.    , peak_position_lowerbound, 0.    ]
                upperbound     += [np.inf, peak_position_upperbound, np.inf]
            if self.max_n_gauss_ >= 3:
                init_gauss_p0   = np.append(init_gauss_p0, [init_f*0.2  , peak_position-init_sigma*1.0 , init_sigma*2.0 ])
                lowerbound     += [0.    , peak_position_lowerbound, 0.    ]
                upperbound     += [np.inf, peak_position_upperbound, np.inf]
            if self.max_n_gauss_ >= 4:
                init_gauss_p0   = np.append(init_gauss_p0, [init_f*0.01, peak_position-init_sigma*2.0 , init_sigma*10.0])
                lowerbound     += [0.    , peak_position_lowerbound, 0.    ]
                upperbound     += [np.inf, peak_position_upperbound, np.inf]
            if self.max_n_gauss_ >= 5: #now support only 4 gaussians
                print("N/A")
                exit()
            if self.background_type_ == 'c':
                init_c = 0.001
                p0 = np.append(init_gauss_p0, [init_c])
                lowerbound += [0.    ]
                upperbound += [np.inf]
                const_flag = None
            elif self.background_type_ == 'p':
                init_b = 0.0001
                p0 = np.append(init_gauss_p0, [init_b, self.neutron_e0_])
                lowerbound += [0.       , -np.inf]
                upperbound += [np.inf   ,  np.inf]
                const_flag = [False, False, False]*self.max_n_gauss_+[False, True]
                
            resolution_model = CF.Model(function = self.R_QE)
            fail_count = 0
            while fail_count < max_fail_count:
                try:
                    fity, popt, perr = resolution_model.fit_transform(xdata = energy_, ydata = resolution_q_, yerr = error_q_ if weighted_with_error == True else None,\
                    p0 = p0, bounds = [lowerbound, upperbound], \
                    const_flag = [False, False, False]*self.max_n_gauss_+[False, True] if self.background_type_ == 'p' else None)
                    break
                except:
                    print("fit fail %d"%(fail_count))
                    rnd_ratio = np.random.uniform(0.001, 2.000, size = len(p0))
                    if self.background_type_ == 'p': rnd_ratio[-1] = 1.0
                    p0 *= rnd_ratio
                    fail_count += 1
            if fail_count == max_fail_count: exit()
            
            self.fitted_parameters_.append(np.copy(popt))
            self.fitted_parameters_error_.append(np.copy(perr))
            if weighted_with_error:
                self.chi2_.append(np.mean(((resolution_q_-fity)/error_q_)**2/np.sum(1.0/error_q_**2))**0.5)
            else:
                self.chi2_.append(np.mean((resolution_q_-fity)**2)**0.5)

            print("The fitting has converged, chi^2 = %.3e"%(self.chi2_[-1]))

    def output_results(self, output_dir = "."):
        if not os.path.isdir(output_dir): os.makedirs(output_dir)
        fout = open(output_dir + "/fitting_results_%s.txt"%(self.grpfilename_[:-4]), "w")
        fout.write("Input file name: %s\n"%(self.grpfilename_))
        fout.write("Incident neutron energy: %s meV\n"%(self.neutron_e0_))
        fout.write("Seed: %s\n"%(self.seed_))
        fout.write("Number of qs: %d\n"%(len(self.q_index_)))
        fout.write("Maximum number of gaussians: %s\n"%(self.max_n_gauss_))
        fout.write("Mirror: %s\n"%(self.mirror_))
        fout.write("Group#      q |  Set data range  |  Actual data range (meV)\n")
        for iq, q_index in enumerate(self.q_index_):
            fout.write("%6s  %.3f | %6s ~ %6s  |  %6s ~ %6s\n"%(q_index, self.q_[iq], self.energy_range_[iq][0], self.energy_range_[iq][1], self.energy_[iq][0], self.energy_[iq][-1]))
        fout.write("\nFitting Results\n")
        fout.write("Gaussian\t%20s\t%17s\t%17s\t\t%17s\t%17s\t%17s\n"%("f", "mean", "sigma", "f_e", "mean_e", "sigma_e"))
        if self.background_type_ == 'c':
            fout.write("Const\t\t%20s\t\t\t\t\t\t\t\t\t\t\t\t\t\t%17s\n"%("c", "c_e"))
        elif self.background_type_ == 'p':
            fout.write("Power\t\t%20s\t%20s\t\t\t\t\t\t\t%20s\t%20s\n"%("b", "e0", "b_e", "e0_e"))

        for iq, q_index in enumerate(self.q_index_):
            group_str = "Group#: %s, q = %s 1/Angstrom"%(q_index, self.q_[iq])
            fout.write(group_str+"-"*(133-len(group_str))+"\n")


            for i in range(0, 3*self.max_n_gauss_, 3):
                fout.write("gauss%s\t\t"%(i//3+1))
                for j in self.fitted_parameters_[iq][i:i+3]: fout.write("%17.10e\t"%j)
                fout.write("\t")
                for j in self.fitted_parameters_error_[iq][i:i+3]: fout.write("%17.10e\t"%j)
                fout.write("\n")

            if self.background_type_ == 'c':
                fout.write("const\t\t\t")
                for j in self.fitted_parameters_[iq][-1:]: fout.write("%17.10e\t"%j)
                fout.write("\t\t\t\t\t\t\t\t\t\t\t\t\t")
                for j in self.fitted_parameters_error_[iq][-1:]: fout.write("%17.10e\t"%j)
                fout.write("\n")
            if self.background_type_ == 'p':
                fout.write("power\t\t")
                for j in self.fitted_parameters_[iq][-2:]: fout.write("%17.10e\t"%j)
                fout.write("\t\t\t\t\t\t")
                for j in self.fitted_parameters_error_[iq][-2:]: fout.write("%17.10e\t"%j)
                fout.write("\n")
            
            fout.write("chi^2 = %17.10e\n"%(self.chi2_[iq]))
            fout.write("-"*133+"\n")
        fout.close()
    
    def plot_results(self, output_dir = ".", log_scale = False, show_errorbar = True):
        if not os.path.isdir(output_dir): os.makedirs(output_dir)
        for iq, q_index in enumerate(self.q_index_):
            resolution_q_ = self.resolution_[iq]
            error_q_ = self.error_[iq]
            energy_q_ = self.energy_[iq]
            true_resolution_indices = np.where(resolution_q_ > 0)
            resolution_q_ = resolution_q_[true_resolution_indices]
            error_q_ = error_q_[true_resolution_indices]
            energy_q_ = energy_q_[true_resolution_indices]
            
            component_ = self.R_QE_component(energy_q_, *self.fitted_parameters_[iq])
            fity = self.R_QE(energy_q_, *self.fitted_parameters_[iq])
            component_legend_ = ["Gaussian component %s"%(i+1) for i in range(self.max_n_gauss_)]
            if self.background_type_ == 'c':
                component_legend_ += ["Constant Background"]
            else:
                component_legend_ += ["Power Law Background"]
            total_legend_ = "Fitted Curve"
            legend_ = ["Resolution Spectra"]+[total_legend_]+component_legend_

            y_ = np.vstack((resolution_q_, fity, component_))
            x_ = np.tile(energy_q_, (y_.shape[0], 1))
            if show_errorbar:
                yerr_ = [error_q_] + [[] for i in range(y_.shape[0]-1)]
            else:
                yerr_ = []

            peak_value = resolution_q_.max()
            log_sufix = ""
            if log_scale:
                ylim = (1e-4, 10)
                log_sufix = "-log"
            else: ylim = (0.0, 1.1)
            log_fg = (False, log_scale)
            plotfilename = "fitting_plot_%s_%d%s.png"%(self.grpfilename_[:-4], iq, log_sufix)
            datafilename = "fitting_data_%s_%d.csv"%(self.grpfilename_[:-4], iq)

            pt = DataPlot()
            pt.plot(x_, y_, yerr_ = yerr_, \
            plottitle = r"$Q\ =\ %s\ \mathrm{\AA^{-1}}$"%(self.q_[iq]), \
            lw_ = [0]+[2]*(1+component_.shape[0]), \
            markerstyle_ = ['o']+[None]*(1+component_.shape[0]), \
            linestyle_ = ['-']*2+[':']*component_.shape[0], \
            ylim = ylim, xlabel = r'$E$ (meV)', ylabel = r'$intensity$ (A.U.)', \
            legend_fontsize = 16, \
            log_fg = log_fg, \
            legend_ = legend_)
            pt.plot_save(output_dir + "/" + plotfilename)
            pt.plot_clear()

            with open(output_dir + "/" + datafilename, "w") as fout:
                fout.write("%s,"%("E (MeV)"))
                for j in range(x_.shape[0]):
                    fout.write("%s,"%(legend_[j]))
                    if j == 0: fout.write("%s,"%("error"))
                fout.write("\n")
                for i in range(x_.shape[1]):
                    fout.write("%s,"%(x_[j][i]))
                    for j in range(x_.shape[0]):
                        fout.write("%e,"%(y_[j][i]))
                        if j == 0: fout.write("%e,"%(yerr_[0][i]))
                    fout.write("\n")


class QENSDataModel:
    def __init__(self, grpfilename, resolution_parameter_filename, energy_range = None, neutron_e0 = None, seed = 42, background_type = 'c', mirror = None):
        
        self.resolution_fitted_parameters_ = []
        self.mirror_ = None
        with open(resolution_parameter_filename, "r") as fin:
            for aline in fin:
                if "Maximum number of gaussians" in aline:
                    self.max_n_gauss_ = int(aline.strip().split()[-1])
                elif "Mirror" in aline:
                    self.mirror_ = aline.strip().split()[-1]
                elif "Number of qs" in aline:
                    n_of_q = int(aline.strip().split()[-1])
                elif "Set data range" in aline:
                    self.resolution_energy_range = []
                    self.q_index_ = []
                    for iq in range(n_of_q):
                        linelist = fin.readline().strip().split()
                        self.q_index_.append(int(linelist[0]))
                        self.resolution_energy_range.append((float(linelist[3]), float(linelist[5])))
                elif "Group#" in aline:
                    self.resolution_fitted_parameters_.append([])
                    for i in range(self.max_n_gauss_):
                        linelist = fin.readline().strip().split()
                        if len(self.resolution_fitted_parameters_[-1]) == 0:
                            self.resolution_fitted_parameters_[-1] = np.array([float(i) for i in linelist[1:3+1]])
                        else:
                            self.resolution_fitted_parameters_[-1] = np.append(self.resolution_fitted_parameters_[-1], [float(i) for i in linelist[1:3+1]])

        if mirror: self.mirror_ = mirror
        if self.mirror_ is None: self.mirror_ = 'off'
        if self.mirror_ != 'off' and self.mirror_ != 'left' and self.mirror_ != 'right':
            print("mirror should be off/left/right, instead of %s."%(self.mirror_))
            exit()

        self.seed_ = seed
        np.random.seed(self.seed_)
        self.neutron_e0_ = neutron_e0 if neutron_e0 else 'N/A'
        self.background_type_ = background_type #background_type: 'c' = constant, 'p' = power law
        self.grpfilename_ = grpfilename

        energy_tmp, self.q_, QENSdata_tmp, error_tmp = grpFileReader(self.grpfilename_).read()
        self.q_ = self.q_[self.q_index_]
        QENSdata_tmp = QENSdata_tmp[self.q_index_]
        error_tmp = error_tmp[self.q_index_]

        if energy_range:
            if type(energy_range) == tuple or type(energy_range) == list:
                if type(energy_range[0]) == tuple or type(energy_range[0]) == list:
                    self.energy_range_ = energy_range
                else:
                    self.energy_range_ = [energy_range]*len(self.q_index_)
            else:
                self.energy_range_ = [(-energy_range, energy_range)]*len(self.q_index_)
        else:
            self.energy_range_ = self.resolution_energy_range

        self.energy_ = []
        self.QENSdata_ = []
        self.error_ = []
        for iq, q_index in enumerate(self.q_index_):
            left  = np.where(energy_tmp >=  self.energy_range_[iq][0])[0]
            mid   = np.where(energy_tmp > 0.)[0][0]
            right = np.where(energy_tmp >   self.energy_range_[iq][1])[0]
            left  = left[0]  if len(left)  > 0 else 0
            right = right[0] if len(right) > 0 else len(energy_tmp)
            if self.mirror_ == 'off':
                self.QENSdata_.append(QENSdata_tmp[iq][left:right])
                self.error_.append(error_tmp[iq][left:right])
                self.energy_.append(energy_tmp[left:right])
            elif self.mirror_ == 'left':
                self.QENSdata_.append(np.concatenate((QENSdata_tmp[iq][left:mid], np.flip(QENSdata_tmp[iq][left:mid], axis=1)), axis=1))
                self.error_.append(np.concatenate((error_tmp[iq][left:mid], np.flip(error_tmp[iq][left:mid], axis=1)), axis=1))
                self.energy_.append(np.concatenate((energy_tmp[left:mid], -np.flip(energy_tmp[left:mid], axis=0)), axis=0))
            elif self.mirror_ == 'right':
                self.QENSdata_.append(np.concatenate((np.flip(QENSdata_tmp[iq][mid:right], axis=1), QENSdata_tmp[iq][mid:right]), axis=1))
                self.error_.append(np.concatenate((np.flip(error_tmp[iq][mid:right], axis=1), error_tmp[iq][mid:right]), axis=1))
                self.energy_.append(np.concatenate((-np.flip(energy_tmp[mid:right], axis=0), energy_tmp[mid:right]), axis=0))
            
            self.error_[-1]    /= self.QENSdata_[-1].max()
            self.QENSdata_[-1] /= self.QENSdata_[-1].max()


        self.E_max = []
        self.t_ = []
        self.t_symmetric_ = []
        self.E_symmetric_ = []
        self.fitted_R_QE_symmetric_ = []
        self.delta_E = 0.002
        
        for iq, q_index in enumerate(self.q_index_):
            self.E_max.append(abs(self.energy_range_[iq][1]-self.energy_range_[iq][0])*10) # meV
            #compute time and energy axes
            NFFT = np.floor(self.E_max[iq] / self.delta_E) + 1
            delta_t = 2 * np.pi / ( 2 * self.E_max[iq] )
            self.t_.append(np.arange(NFFT - 1) * delta_t)
            self.t_symmetric_.append(np.concatenate((-np.flip(self.t_[iq][1:]), self.t_[iq])))
            e_symmetric, _ = ezfft.ezifft(self.t_symmetric_[iq], np.zeros_like(self.t_symmetric_[iq]))
            self.E_symmetric_.append(e_symmetric)

            #compute fitted R(Q, E)
            self.fitted_R_QE_symmetric_.append(self.R_QE(self.E_symmetric_[iq], *self.resolution_fitted_parameters_[iq]))

    def fit(self, const_f_elastic = None, const_f1 = None, const_tau1 = None, const_f2 = 0, const_tau2 = None, const_tau = None, const_beta = None, const_background = None,\
                  initial_A = 1, initial_f_elastic = 0.5, initial_f1 = 0.05, initial_tau1 = 1, initial_f2 = 0.1, initial_tau2 = 10, initial_tau = 100, initial_beta = 0.5, initial_E_center = 0, initial_background = 0.001, \
                  lowerbound_A =      0, lowerbound_f_elastic = 0, lowerbound_f1 = 0, lowerbound_tau1 =      0, lowerbound_f2 = 0, lowerbound_tau2 =      0, lowerbound_tau =      0, lowerbound_beta = 0, lowerbound_E_center = None, lowerbound_background =      0,\
                  upperbound_A = np.inf, upperbound_f_elastic = 1, upperbound_f1 = 1, upperbound_tau1 = np.inf, upperbound_f2 = 1, upperbound_tau2 = np.inf, upperbound_tau = np.inf, upperbound_beta = 1, upperbound_E_center = None, upperbound_background = np.inf,\
                  max_fail_count = 20, weighted_with_error = True, use_previous_q_as_initial_guess = False):
        self.fitted_parameters_ = []
        self.fitted_parameters_error_ = []
        self.chi2_ = []
        self.now_fitting_iq = -1
        if type(const_f_elastic) != tuple and type(const_f_elastic) != list:
                const_f_elastic = [const_f_elastic]*len(self.q_index_)
        if type(const_f1) != tuple and type(const_f1) != list:
                const_f1 = [const_f1]*len(self.q_index_)
        if type(const_tau1) != tuple and type(const_tau1) != list:
                const_tau1 = [const_tau1]*len(self.q_index_)
        if type(const_f2) != tuple and type(const_f2) != list:
                const_f2 = [const_f2]*len(self.q_index_)
        if type(const_tau2) != tuple and type(const_tau2) != list:
                const_tau2 = [const_tau2]*len(self.q_index_)
        if type(const_tau) != tuple and type(const_tau) != list:
                const_tau = [const_tau]*len(self.q_index_)
        if type(const_beta) != tuple and type(const_beta) != list:
                const_beta = [const_beta]*len(self.q_index_)
        if type(const_background) != tuple and type(const_background) != list:
                const_background = [const_background]*len(self.q_index_)

        if type(use_previous_q_as_initial_guess) != tuple and type(use_previous_q_as_initial_guess) != list:
            use_previous_q_as_initial_guess = [use_previous_q_as_initial_guess]*len(self.q_index_)
        if type(initial_A) != tuple and type(initial_A) != list:
            initial_A = [initial_A]*len(self.q_index_)
        if type(initial_f_elastic) != tuple and type(initial_f_elastic) != list:
            initial_f_elastic = [initial_f_elastic]*len(self.q_index_)
        if type(initial_f1) != tuple and type(initial_f1) != list:
            initial_f1 = [initial_f1]*len(self.q_index_)
        if type(initial_tau1) != tuple and type(initial_tau1) != list:
            initial_tau1 = [initial_tau1]*len(self.q_index_)
        if type(initial_f2) != tuple and type(initial_f2) != list:
            initial_f2 = [initial_f2]*len(self.q_index_)
        if type(initial_tau2) != tuple and type(initial_tau2) != list:
            initial_tau2 = [initial_tau2]*len(self.q_index_)
        if type(initial_tau) != tuple and type(initial_tau) != list:
            initial_tau = [initial_tau]*len(self.q_index_)
        if type(initial_beta) != tuple and type(initial_beta) != list:
            initial_beta = [initial_beta]*len(self.q_index_)
        if type(initial_E_center) != tuple and type(initial_E_center) != list:
            initial_E_center = [initial_E_center]*len(self.q_index_)
        if type(initial_background) != tuple and type(initial_background) != list:
            initial_background = [initial_background]*len(self.q_index_)
         
        for iq, q_index in enumerate(self.q_index_):
            print("Now fitting group %s"%(q_index))
            self.now_fitting_iq = iq
            QENSdata_q_ = self.QENSdata_[self.now_fitting_iq]
            error_q_ = self.error_[self.now_fitting_iq]
            energy_q_ = self.energy_[self.now_fitting_iq]
            true_QENSdata_indices = np.where(QENSdata_q_ > 0)
            QENSdata_q_ = QENSdata_q_[true_QENSdata_indices]
            error_q_ = error_q_[true_QENSdata_indices]
            energy_q_ = energy_q_[true_QENSdata_indices]

            peak_value = QENSdata_q_.max()
            QENS_model = CF.Model(function = self.QENSdata_function)
            if lowerbound_E_center is None: lowerbound_E_center = energy_q_.min()
            if upperbound_E_center is None: upperbound_E_center = energy_q_.max()

            # A  f_elastic f1 tau1 f2 tau2 tau beta E_center background
            lowerbound          = [lowerbound_A, lowerbound_f_elastic, lowerbound_f1, lowerbound_tau1, lowerbound_f2, lowerbound_tau2, lowerbound_tau, lowerbound_beta, lowerbound_E_center, lowerbound_background]
            upperbound          = [upperbound_A, upperbound_f_elastic, upperbound_f1, upperbound_tau1, upperbound_f2, upperbound_tau2, upperbound_tau, upperbound_beta, upperbound_E_center, upperbound_background]
            
            #A
            const_flag          = [False]
            p0                  = [self.fitted_parameters_[-1][0]] if iq > 0 and use_previous_q_as_initial_guess[0] == True else [initial_A[iq]]

            if const_f_elastic[iq] is not None: 
                const_flag     += [True]
                p0             += [const_f_elastic[iq]]
            else: 
                const_flag     += [False]
                p0             += [self.fitted_parameters_[-1][1]] if iq > 0 and use_previous_q_as_initial_guess[1] == True else [initial_f_elastic[iq]]
            
            #f1 and tau1
            if const_f1[iq] is not None:
                const_flag     += [True]
                p0             += [const_f1[iq]]
                if const_f1[iq] == 0:
                    const_flag += [True]
                    p0         += [self.fitted_parameters_[-1][3]] if iq > 0 and use_previous_q_as_initial_guess[3] == True else [initial_tau1[iq]]
                elif const_tau1[iq] is not None:
                    const_flag += [True]
                    p0         += [const_tau1[iq]]
                else:
                    const_flag += [False]
                    p0         += [self.fitted_parameters_[-1][3]] if iq > 0 and use_previous_q_as_initial_guess[3] == True else [initial_tau1[iq]]
            else:
                const_flag     += [False]
                p0             += [self.fitted_parameters_[-1][2]] if iq > 0 and use_previous_q_as_initial_guess[2] == True else [initial_f1[iq]]
                if const_tau1[iq] is not None:
                    const_flag += [True]
                    p0         += [const_tau1[iq]]
                else:
                    const_flag += [False]
                    p0         += [self.fitted_parameters_[-1][3]] if iq > 0 and use_previous_q_as_initial_guess[3] == True else [initial_tau1[iq]]

            #f2 and tau2
            if const_f2[iq] is not None:
                const_flag     += [True]
                p0             += [const_f2[iq]]
                if const_f2[iq] == 0:
                    const_flag += [True]
                    p0         += [self.fitted_parameters_[-1][5]] if iq > 0 and use_previous_q_as_initial_guess[5] == True else [initial_tau2[iq]]
                elif const_tau2[iq] is not None:
                    const_flag += [True]
                    p0         += [const_tau2[iq]]
                else:
                    const_flag += [False]
                    p0         += [self.fitted_parameters_[-1][5]] if iq > 0 and use_previous_q_as_initial_guess[5] == True else [initial_tau2[iq]]
            else:
                const_flag     += [False]
                p0             += [self.fitted_parameters_[-1][4]] if iq > 0 and use_previous_q_as_initial_guess[4] == True else [initial_f2[iq]]
                if const_tau2[iq] is not None:
                    const_flag += [True]
                    p0         += [const_tau2[iq]]
                else:
                    const_flag += [False]
                    p0         += [self.fitted_parameters_[-1][5]] if iq > 0 and use_previous_q_as_initial_guess[5] == True else [initial_tau2[iq]]

            #tau
            if const_tau[iq] is not None:
                const_flag         += [True]
                p0                 += [const_tau[iq]]
            else:
                const_flag         += [False]
                p0                 += [self.fitted_parameters_[-1][6]] if iq > 0 and use_previous_q_as_initial_guess[6] == True else [initial_tau[iq]] 
            
            #beta
            if const_beta[iq] is not None:
                const_flag     += [True]
                p0             += [const_beta[iq]]
            else:
                const_flag     += [False]
                p0             += [self.fitted_parameters_[-1][7]] if iq > 0 and use_previous_q_as_initial_guess[7] == True else [initial_beta[iq]] 
                
            #E_center
            const_flag         += [False]
            p0                 += [self.fitted_parameters_[-1][8]] if iq > 0 and use_previous_q_as_initial_guess[8] == True else [initial_E_center[iq]] 

            #background
            if const_background[iq] is not None:
                const_flag     += [True]
                p0             += [const_background[iq]]
            else:
                const_flag     += [False]
                p0             += [self.fitted_parameters_[-1][9]] if iq > 0 and use_previous_q_as_initial_guess[9] == True else [initial_background[iq]] 
            
            fail_count = 0

            # print(p0)
            # fity, popt, perr = QENS_model.fit_transform(xdata = energy_q_, ydata = QENSdata_q_, yerr = error_q_ if weighted_with_error == True else None,\
            # p0 = p0, bounds = [lowerbound, upperbound], const_flag = const_flag)
            while fail_count < max_fail_count:
                try:
                    fity, popt, perr = QENS_model.fit_transform(xdata = energy_q_, ydata = QENSdata_q_, yerr = error_q_ if weighted_with_error == True else None,\
                    p0 = p0, bounds = [lowerbound, upperbound], const_flag = const_flag)
                    break
                except:
                    print("fit fail %d"%(fail_count))
                    rnd_ratio = np.random.uniform(0.001, 2.000, size = len(p0))
                    rnd_ratio = np.array([a if const_flag[i] == False else 1.0 for i, a in enumerate(rnd_ratio)])
                    p0 *= rnd_ratio
                    fail_count += 1
            if fail_count == 20: exit()
            self.fitted_parameters_.append(np.copy(popt))
            self.fitted_parameters_error_.append(np.copy(perr))
            if weighted_with_error:
                self.chi2_.append(np.mean(((QENSdata_q_-fity)/error_q_)**2/np.sum(1.0/error_q_**2))**0.5)
            else:
                self.chi2_.append(np.mean((QENSdata_q_-fity)**2)**0.5)

            print("The fitting has converged, chi^2 = %.3e"%(self.chi2_[-1]))

    def R_QE_component(self, E_, *args):
        component_ = []
        for i in range(0, 3*self.max_n_gauss_, 3):
            component_.append(gauss(E_, *args[i:i+3]))
        component_ = np.array(component_)
        return component_
    
    def R_QE(self, E_, *args):
        return np.sum(self.R_QE_component(E_, *args), axis = 0)
    
    def F_Qt_component(self, t_, f1, tau1, f2, tau2, tau, beta):
        return np.vstack((f1 * np.exp(-t_/tau1), f2 * np.exp(-t_/tau2), (1.0 - f1 - f2) * kww(t_, tau, beta)))
    
    def QENS_function(self, E_data_, *args):
        #name the parameters
        A          = args[0]
        f_elastic  = args[1]
        f1         = args[2]
        tau1       = args[3] * 1.519
        f2         = args[4]
        tau2       = args[5] * 1.519
        tau        = args[6] * 1.519
        beta       = args[7]
        E_center   = args[8]
        background = args[9]

        #load pre-calculated time and energy axes
        t_ = self.t_[self.now_fitting_iq]
        t_symmetric_ = self.t_symmetric_[self.now_fitting_iq]
        E_symmetric_ = self.E_symmetric_[self.now_fitting_iq]
        
        #load pre-calculated fitted R(Q, E)
        R_QE_symmetric_ = self.fitted_R_QE_symmetric_[self.now_fitting_iq]
        
        #compute F(Q, t) (fast component, KWW function)
        #F_Qt_ = self.F_Qt(t_, tau, beta)
        F_Qt_component_ = self.F_Qt_component(t_, f1, tau1, f2, tau2, tau, beta)
        F_Qt_component_symmetric_ = np.concatenate((np.flip(F_Qt_component_[:, 1:], axis = 1), F_Qt_component_), axis = 1)
        
        #compute R(Q, t)
        _, R_Qt_symmetric_ = ezfft.ezfft(E_symmetric_, R_QE_symmetric_)
        
        #compute F(Q, t)R(Q, t) and it's FT
        F_Qt_component_times_R_Qt_symmetric_ = F_Qt_component_symmetric_ * R_Qt_symmetric_

        _, tmp0 = ezfft.ezifft(t_symmetric_, F_Qt_component_times_R_Qt_symmetric_[0])
        _, tmp1 = ezfft.ezifft(t_symmetric_, F_Qt_component_times_R_Qt_symmetric_[1])
        _, tmp2 = ezfft.ezifft(t_symmetric_, F_Qt_component_times_R_Qt_symmetric_[2])
        S_QE_component_conv_R_QE_symmetric_ = np.vstack((tmp0,tmp1,tmp2))

        #take only the real part
        R_QE_symmetric_ = np.real(R_QE_symmetric_)
        S_QE_component_conv_R_QE_symmetric_ = np.real(S_QE_component_conv_R_QE_symmetric_)

        #compute all the terms
        y_ENS_ = A * f_elastic * R_QE_symmetric_
        y_QENS_component_ = A * (1.0-f_elastic) * S_QE_component_conv_R_QE_symmetric_
        y_background_ = constant(E_symmetric_, background)
        
        #interpolate all the terms at data point
        E_symmetric_after_shift_ = E_symmetric_ + E_center
        y_ENS_data_ = np.interp(E_data_, E_symmetric_after_shift_, y_ENS_)
        y_QENS_component_data_ = np.interp(E_data_, E_symmetric_after_shift_, y_QENS_component_[0])
        for tmp in y_QENS_component_[1:]:
            s = np.interp(E_data_, E_symmetric_after_shift_, tmp)
            y_QENS_component_data_ = np.vstack((y_QENS_component_data_, s))
        y_background_data_ = np.interp(E_data_, E_symmetric_after_shift_, y_background_)
        y_model_data_ = y_ENS_data_ + np.sum(y_QENS_component_data_, axis = 0) + y_background_data_
        
        return y_model_data_, y_ENS_data_, y_QENS_component_data_, y_background_data_

    def QENSdata_function(self, E_data_, *args):
        y_model_data_, _, _, _ = self.QENS_function(E_data_, *args)
        return y_model_data_  

    def output_results(self, output_dir = "."):
        if not os.path.isdir(output_dir): os.makedirs(output_dir)
        fout = open(output_dir + "/fitting_results_%s.txt"%(self.grpfilename_[:-4]), "w")
        fout.write("Input file name: %s\n"%(self.grpfilename_))
        fout.write("Incident neutron energy: %s meV\n"%(self.neutron_e0_))
        fout.write("Seed: %s\n"%(self.seed_))
        fout.write("Number of qs: %d\n"%(len(self.q_index_)))
        fout.write("Mirror: %s\n"%(self.mirror_))
        fout.write("Group#      q |  Set data range  |  Actual data range (meV)\n")
        for iq, q_index in enumerate(self.q_index_):
            fout.write("%6s  %.3f | %6s ~ %6s  |  %6s ~ %6s\n"%(q_index, self.q_[iq], self.energy_range_[iq][0], self.energy_range_[iq][1], self.energy_[iq][0], self.energy_[iq][-1]))
        fout.write("Resolution data range (meV)\n")
        for iq, q_index in enumerate(self.q_index_):
            fout.write("%6s  %.3f | %6s ~ %6s\n"%(q_index, self.q_[iq], self.resolution_energy_range[iq][0], self.resolution_energy_range[iq][1])) 
        
        fout.write("\nFitting Results\n")
        fout.write("%6s\t%17s\t%17s"%("Group#","q (1/Angstrom)","chi^2"))
        fout.write("\t%17s\t%17s\t%17s\t%17s\t%17s\t%17s\t%17s\t%17s\t%17s"%("A", "f_elastic", "f1", "tau1 (ps)", "f2", "tau2 (ps)", "tau (ps)", "beta", "E_center (meV)"))
        if self.background_type_ == 'c': fout.write("\t%17s"%("c"))
        elif self.background_type_ == 'p': fout.write("\t%17s\t%17s"%("b", "e0"))
        fout.write("\t\t%17s\t%17s\t%17s\t%17s\t%17s\t%17s\t%17s\t%17s\t%17s"%("A_e", "f_elastic_e", "f1_e", "tau1_e", "f2_e", "tau2_e", "tau_e", "beta_e", "E_center_e"))
        if self.background_type_ == 'c': fout.write("\t%17s"%("c_e"))
        elif self.background_type_ == 'p': fout.write("\t%17s\t%17s"%("b_e", "e0_e"))
        fout.write("\n")
        for iq, q_index in enumerate(self.q_index_):
            fout.write("%6s\t%17s\t%17.10e\t"%(q_index, self.q_[iq],self.chi2_[iq]))
            for j in self.fitted_parameters_[iq]: fout.write("%17.10e\t"%j)
            fout.write("\t")
            for j in self.fitted_parameters_error_[iq]: fout.write("%17.10e\t"%j)
            fout.write("\n")
        fout.close()

        foutcsv = open(output_dir + "/fitting_results_%s.csv"%(self.grpfilename_[:-4]), "w")
        foutcsv.write("%s,%s,%s"%("Group#","q (1/Angstrom)","chi^2"))
        foutcsv.write(",%s,%s,%s,%s,%s,%s,%s,%s,%s"%("A", "f_elastic", "f1", "tau1 (ps)", "f2", "tau2 (ps)", "tau (ps)", "beta", "E_center (meV)"))
        if self.background_type_ == 'c': foutcsv.write(",%s"%("c"))
        elif self.background_type_ == 'p': foutcsv.write(",%s,%s"%("b", "e0"))
        foutcsv.write(",%s,%s,%s,%s,%s,%s,%s,%s,%s"%("A_e", "f_elastic_e", "f1_e", "tau1_e", "f2_e", "tau2_e", "tau_e", "beta_e", "E_center_e"))
        if self.background_type_ == 'c': foutcsv.write(",%s"%("c_e"))
        elif self.background_type_ == 'p': foutcsv.write(",%s,%s"%("b_e", "e0_e"))
        foutcsv.write("\n")
        for iq, q_index in enumerate(self.q_index_):
            foutcsv.write("%s,%s,%e,"%(q_index, self.q_[iq],self.chi2_[iq]))
            for j in self.fitted_parameters_[iq]: foutcsv.write("%e,"%j)
            for j in self.fitted_parameters_error_[iq]: foutcsv.write("%e,"%j)
            foutcsv.write("\n")
        foutcsv.close()

    def plot_results(self, output_dir = ".", log_scale = False, show_errorbar = True):
        if not os.path.isdir(output_dir): os.makedirs(output_dir)
        for iq, q_index in enumerate(self.q_index_):
            self.now_fitting_iq = iq
            QENSdata_q_ = self.QENSdata_[self.now_fitting_iq]
            error_q_ = self.error_[self.now_fitting_iq]
            energy_q_ = self.energy_[self.now_fitting_iq]
            true_QENSdata_indices = np.where(QENSdata_q_ > 0)
            QENSdata_q_ = QENSdata_q_[true_QENSdata_indices]
            error_q_ = error_q_[true_QENSdata_indices]
            energy_q_ = energy_q_[true_QENSdata_indices]

            fity, y_ENS_data_, y_QENS_component_data_, y_background_data_ = self.QENS_function(energy_q_, *self.fitted_parameters_[iq])
            
            legend_ = ["QENS Spectra", "Fitted Curve", "ENS component", "QENS(exp1) component", "QENS(exp2) component", "QENS(KWW) component"]
            if self.background_type_ == 'c':
                legend_.append("Constant Background")
            else:
                legend_.append("Power Law Background")
            legend_.append("Resolution (%s~%s meV)"%(self.resolution_energy_range[iq][0], self.resolution_energy_range[iq][1]))
            
            R_QE_symmetric_data_ = np.interp(energy_q_, self.E_symmetric_[self.now_fitting_iq], self.fitted_R_QE_symmetric_[self.now_fitting_iq])
            y_ = np.vstack((QENSdata_q_, fity, y_ENS_data_, y_QENS_component_data_, y_background_data_, R_QE_symmetric_data_))
            x_ = np.tile(energy_q_, (y_.shape[0], 1))

            if show_errorbar:
                yerr_ = [error_q_] + [[] for i in range(y_.shape[0]-1)]
            else:
                yerr_ = []

            peak_value = QENSdata_q_.max()
            log_sufix = ""
            if log_scale:
                ylim = (1e-4, 10)
                log_sufix = "-log"
            else: ylim = (0.0, 1.1)
            log_fg = (False, log_scale)
            plotfilename = "fitting_plot_%s_%d%s.png"%(self.grpfilename_[:-4], q_index, log_sufix)
            datafilename = "fitting_data_%s_%d.csv"%(self.grpfilename_[:-4], q_index)

            pt = DataPlot()
            pt.plot(x_, y_, yerr_ = yerr_, \
            plottitle = r"$Q\ =\ %s\ \mathrm{\AA^{-1}}$"%(self.q_[iq]), \
            lw_ = [0]+[2]*(len(y_)-1), \
            markerstyle_ = ['o']+[None]*(len(y_)-1), \
            linestyle_ = ['-']*2+[':']*(len(y_)-3)+['-.'], \
            ylim = ylim, xlabel = r'$E$ (meV)', ylabel = r'$intensity$ (A.U.)', \
            legend_fontsize = 16, \
            log_fg = log_fg, \
            legend_ = legend_)
            pt.plot_save(output_dir + "/" + plotfilename)
            pt.plot_clear()

            with open(output_dir + "/" + datafilename, "w") as fout:
                fout.write("%s,"%("E (MeV)"))
                for j in range(x_.shape[0]):
                    fout.write("%s,"%(legend_[j]))
                    if j == 0: fout.write("%s,"%("error"))
                fout.write("\n")
                for i in range(x_.shape[1]):
                    fout.write("%s,"%(x_[j][i]))
                    for j in range(x_.shape[0]):
                        fout.write("%e,"%(y_[j][i]))
                        if j == 0: fout.write("%e,"%(yerr_[0][i]))
                    fout.write("\n")