#
#  ZLabPlot.py
#  
#  Copyright (c) 2022 Z-Group. All rights reserved.
#  -----------------------------------------------------
#  Current developers  : Shao-Chun Lee    (2022 - Present)
#  -----------------------------------------------------

import sys
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

if 'win' in sys.platform.lower() and 'darwin' not in sys.platform.lower(): plt.rcParms['backend'] = 'TkAgg'
zlab_default_rcParams = plt.rcParams.copy()

class ZLabPlot:
    def __init__(self, rcmap = {}):
        self.clear()
        for key in zlab_default_rcParams.keys(): plt.rcParams[key] = zlab_default_rcParams[key]
        self.default_fontsize = 'large'
        plt.rcParams['axes.titlesize'] = 'large' # 'medium' in matplotlibrc
        plt.rcParams['xtick.major.size']  = plt.rcParams['ytick.major.size']  = 9  
        plt.rcParams['xtick.minor.size']  = plt.rcParams['ytick.minor.size']  = 6    
        plt.rcParams['xtick.major.width'] = plt.rcParams['ytick.major.width'] = 2  
        plt.rcParams['xtick.minor.width'] = plt.rcParams['ytick.minor.width'] = 2  
        plt.rcParams['xtick.major.pad']   = plt.rcParams['ytick.major.pad']   = 10  
        plt.rcParams['xtick.minor.pad']   = plt.rcParams['ytick.minor.pad']   = 10
        plt.rcParams['xtick.labelsize']   = plt.rcParams['ytick.labelsize']   = self.default_fontsize # 'medium' in matplotlibrc
        
        for a_key in rcmap:
            print("Manually set %s to %s"%(a_key, rcmap[a_key]))
            plt.rcParams[a_key] = rcmap[a_key]

        self.default_color_list = plt.rcParams['axes.prop_cycle'].by_key()['color']
        self.number_of_default_colors = len(self.default_color_list)

        self.xlabel_map = {\
        #Radial Distribution Function
        "PDF": r'$r\ \mathrm{(Å)}$',
        "RDF": r'$r\ \mathrm{(Å)}$',
        "gr": r'$r\ \mathrm{(Å)}$',
        #Angular Distribution Function
        "ADF": r'$\theta\ \mathrm{(degree)}$',
        #Structure Factor
        "sk": r'$Q\ \mathrm{(Å^{-1})}$',
        "sq": r'$Q\ \mathrm{(Å^{-1})}$',
        #Mean Squared Displacment
        "msd": r'$t\ \mathrm{(ps)}$',
        "r2t": r'$t\ \mathrm{(ps)}$',
        "mmsd":r'$〈r^2〉(t)\ \mathrm{(nm^2)}$',
        "mr2t":r'$〈r^2〉(t)\ \mathrm{(nm^2)}$',
        #Autocorrelation Functions
        "vacf": r'$t\ \mathrm{(ps)}$',                                      #Velocity
        "eacf": r'$t\ \mathrm{(ps)}$',                                      #Electrical Current
        "hacf": r'$t\ \mathrm{(ps)}$',                                      #Heat Flux
        "sacf": r'$t\ \mathrm{(ps)}$',                                      #Stress
        "fskt": r'$t\ \mathrm{(ps)}$',                                      #Self-intermediate scattering function
        "fkt": r'$t\ \mathrm{(ps)}$',                                       #Collective-intermediate scattering function
        "alpha_2": r'$t\ \mathrm{(ps)}$',                                   #Non-Gaussian parameter
        "chi_4": r'$t\ \mathrm{(ps)}$'}                                     #Four-Point
        self.ylabel_map = {\
        #Radial Distribution Function
        "PDF": r'$g(r)$',
        "RDF": r'$g(r)$',
        "gr": r'$g(r)$',
        #Angular Distribution Function
        "ADF": r'$ADF(\theta)$',
        #Structure Factor
        "sk": r'$S(Q)$',
        "sq": r'$S(Q)$',
        #Mean Squared Displacment
        "msd":r'$〈r^2〉(t)\ \mathrm{(nm^2/s)}$',
        "r2t":r'$〈r^2〉(t)\ \mathrm{(nm^2/s)}$',
        "mmsd":r'$〈r^2〉(t)\ \mathrm{(nm^2/s)}$',
        "mr2t":r'$〈r^2〉(t)\ \mathrm{(nm^2/s)}$',
        #Autocorrelation Functions
        "vacf": r'$\frac{〈v(t)v(0)〉/〈v(0)^2〉}$',                          #Velocity
        "eacf": r'$〈J(t)J(0)〉/〈J(0)^2〉$',                                 #Electrical Current
        "hacf": r'$\frac{〈J(t)J(0)〉/〈J(0)^2〉}$',                          #Heat Flux
        "sacf": r'$\frac{〈\tau_{ij}(t)\tau_{ij}(0)〉}{〈\tau_{ij}(0)^2〉}$', #Stress
        "fskt": r'$F_s(Q,t)$',                                               #Self-intermediate scattering function
        "fkt": r'$F(Q,t)$',                                                  #Collective-intermediate scattering function
        "alpha_2": r'$\alpha_2(t)$',                                         #Non-Gaussian parameter
        "chi_4": r'$\chi_4(t)$'}                                             #Four-Point

    def get_Color_from_RGB(self, RGB):
        #RGB = (int, int, int)
        return "#%02x%02x%02x"%RGB

    def get_color_series(self, color_index, cmap):
        #color_index = list of int
        #cmap = string
        color_rgb = []
        cNorm  = mpl.colors.Normalize(vmin=0, vmax=len(color_index))
        scalarMap = mpl.cm.ScalarMappable(norm=cNorm, cmap=plt.get_cmap(cmap)) #mpl.colormaps[cmap]
        color_rgb = [scalarMap.to_rgba(c) for c in color_index]
        return color_rgb

    # def gradient_image(ax, extent=(0, 1, 0, 1), direction=0, cmap_range=(0, 1), **kwargs):
    #     phi = direction * np.pi / 2
    #     v = np.array([np.cos(phi), np.sin(phi)])
    #     X = np.array([[v @ [1, 0], v @ [1, 1]],
    #                 [v @ [0 ,0], v @ [0, 1]]])
    #     a, b = cmap_range
    #     X = a + (b - a) / X.max() * X
    #     im = ax.imshow(X, extent=extent, interpolation='bicubic', vmin=0, vmax=1, **kwargs)
    #     return im

    def add_subplot(self, subplot_name = "0", subplot_spec = 111, \
                          plottitle = None,\
                          framewidth = None, twinx = False, \
                          projection = '2d'):
        subplot_name_ = str(subplot_name)
        self.projection = projection
        if self.subplot_map.get(subplot_name_) is not None:
            print("subplot name has already been used.")
        else:
            #, margin_ratio = 0.15, axsize = 0.7, wspace = 0.3, hspace = 0.3
            # self.subplot_map[subplot_name_] = plt.subplot(subplot_spec, position= [margin_ratio*(1-axsize), margin_ratio*(1-axsize), axsize, axsize])
            if self.projection == '3d':
                from mpl_toolkits.mplot3d import axes3d
                self.subplot_map[subplot_name_] = plt.subplot(subplot_spec, projection = self.projection)
            else:
                self.subplot_map[subplot_name_] = plt.subplot(subplot_spec, position = [0.1, 0.1, 0.9, 0.9])
            self.subplot_map[subplot_name_].set_title(label = plottitle, pad = 10)
            self.plot_data_map[subplot_name_] = []
            if twinx == True:
                self.subplot_map[subplot_name_+"-t"] = self.subplot_map[subplot_name_].twinx()
                self.plot_data_map[subplot_name_+"-t"] = []

    def add_data(self, data_x, data_y, data_z = None, subplot_name = "0", legend = None, \
                       xlim = None, ylim = None, xlog = False, ylog = False, xstart = None, xinc = None, ystart = None, yinc = None, \
                       xlabel = None, ylabel = None, zlabel = None, \
                       color = None, lw = None, ls = None, ms = None, msize = None, mf = None, cmap = None, max_cmap_index = None, \
                       legendtitle = None, legendtitle_fontsize = None, legend_fontsize = None, tick_fontsize = None, label_fontsize = None, hide_xtick = False, hide_ytick = False, \
                       ncol = 1, legend_location = None, \
                       twinx = False, plottype = None, is_scatter = False, zorder = None):
        #lw    = line width
        #ls    = line style https://matplotlib.org/stable/gallery/lines_bars_and_markers/linestyles.html
        #ms    = marker style https://matplotlib.org/stable/api/markers_api.html
        #msize = marker size
        #mf    = marker fill style {'full', 'left', 'right', 'bottom', 'top', 'none'}
        #color = https://matplotlib.org/stable/gallery/color/named_colors.html
        #cmap  = https://matplotlib.org/stable/tutorials/colors/colormaps.html
        
        #initialize
        subplot_name_ = str(subplot_name)
        if self.subplot_map.get(subplot_name_) is None:
            print("ZLabPlot error: subplot not found.")
            exit()
        if twinx == True:
            subplot_name_ += "-t"
            if self.subplot_map.get(subplot_name_) is None:
                print("ZLabPlot error: subplot specified has no twinx.")
                exit()
        ax = self.subplot_map[subplot_name_]
        if legendtitle_fontsize is None: legendtitle_fontsize = self.default_fontsize
        if legend_fontsize      is None: legend_fontsize      = self.default_fontsize
        if tick_fontsize        is None: tick_fontsize        = self.default_fontsize
        if label_fontsize       is None: label_fontsize       = self.default_fontsize

        #padding features
        data_len = len(data_x)
        legend_ = self.feature_padding(data_len, legend, None)
        if type(lw) == float or type(lw) == int:       lw_    = self.feature_padding(data_len, [], lw)
        else:                                          lw_    = self.feature_padding(data_len, lw, None)
        if type(ls) == str or type(ls) == tuple:       ls_    = self.feature_padding(data_len, [], ls)
        else:                                          ls_    = self.feature_padding(data_len, ls, None)
        if type(ms) == str or type(ms) == int:         ms_    = self.feature_padding(data_len, [], ms)
        else:                                          ms_    = self.feature_padding(data_len, ms, None)
        if type(msize) == float or type(msize) == int: msize_ = self.feature_padding(data_len, [], msize)
        else:                                          msize_ = self.feature_padding(data_len, msize, None)
        if type(mf) == str:                            mf_    = self.feature_padding(data_len, [], mf)
        else:                                          mf_    = self.feature_padding(data_len, mf, None)           
        if type(zorder) == int:                        zorder_= self.feature_padding(data_len, [], zorder)
        else:                                          zorder_= self.feature_padding(data_len, zorder, None)

        #setting colors
        if not is_scatter:
            if cmap is None:
                if type(color) == tuple or type(color) == str or type(color) == int: color_ = self.feature_padding(data_len, [], color)
                else:                                                                color_ = self.feature_padding(data_len, color, None)
                color_tmp = []
                for c in color_:
                    if type(c) == tuple:
                        if len(c) == 3:   color_tmp.append(self.get_Color_from_RGB(c))
                        elif len(c) == 4: color_tmp.append(c)
                    elif type(c) == int: color_tmp.append(self.default_color_list[c%self.number_of_default_colors])
                    else: color_tmp.append(c)      
                color_ = color_tmp
            else:
                color_tmp = []
                if max_cmap_index is None:
                    max_cmap_index = 0
                    if color is not None:
                        max_cmap_index = 0
                        for c in color:
                            if type(c) == int: max_cmap_index = max(max_cmap_index, c)
                    else:
                        max_cmap_index = data_len
                cNorm  = mpl.colors.Normalize(vmin = 0, vmax = max_cmap_index)
                scalarMap = mpl.cm.ScalarMappable(norm=cNorm, cmap=plt.get_cmap(cmap))
                if color is None: color_tmp = [scalarMap.to_rgba(i) for i in range(data_len)]
                else:
                    if len(color) != data_len:
                        print("ZLabPlot error: the length of color array specified is not consistent with number of curves provided.")
                        exit()
                    for c in color:
                        if type(c) == int: color_tmp.append(scalarMap.to_rgba(c))  
                        else: color_tmp.append(c)
                color_ = color_tmp[:]
        else:
            color_ = color

        #adding data
        if not is_scatter:
            for index in range(data_len):
                thisline = ax.plot(data_x[index], data_y[index], label=legend_[index])
                self.plot_data_map[subplot_name_].append(thisline[0])
                if lw_[index] is not None: self.plot_data_map[subplot_name_][-1].set_linewidth(lw_[index])
                if ls_[index] is not None: self.plot_data_map[subplot_name_][-1].set_linestyle(ls_[index])
                if ms_[index] is not None: self.plot_data_map[subplot_name_][-1].set_marker(ms_[index])
                if msize_[index] is not None: self.plot_data_map[subplot_name_][-1].set_markersize(msize_[index])
                if mf_[index] is not None: self.plot_data_map[subplot_name_][-1].set_fillstyle(mf_[index])
                if zorder_[index] is not None: self.plot_data_map[subplot_name_][-1].set_zorder(zorder_[index])
                #self.plot_data_map[subplot_name_][-1].set_markerfacecolor(markerface_[index])
                if color_[index] is not None: self.plot_data_map[subplot_name_][-1].set_color(color_[index])
        else:
            if self.projection == '2d':
                thisscatter = ax.scatter(data_x, data_y, s = msize_, c = color_, cmap = cmap)
            else:
                thisscatter = ax.scatter(data_x, data_y, data_z, s = msize_, c = color_, cmap = cmap)
            self.plot_data_map[subplot_name_].append(thisscatter)
            plt.colorbar(thisscatter)

        #set legend
        if legend is not None: ax.legend(title = legendtitle, ncol = ncol, labelspacing = 0.5, frameon = False, loc = legend_location)
        
        #set axis scale, limit, and ticks
        if xlog: ax.set_xscale('log')
        if ylog: ax.set_yscale('log')
        if xlim is not None and xlim[0] is not None and xlim[1] is not None and xlim[0] >= xlim[1]: print("ZLabPlot error: custom x-axis limit error.")
        if ylim is not None and ylim[0] is not None and ylim[1] is not None and ylim[0] >= ylim[1]: print("ZLabPlot error: custom y-axis limit error.")
        if xlim is not None: ax.set_xlim(xlim)
        else:                xlim = ax.get_xlim()
        if ylim is not None: ax.set_ylim(ylim)
        else:                ylim = ax.get_ylim()
        if xstart is not None and xinc is not None:
            xticks = self.custom_ticks(xstart, xinc, xlog, xlim)
            ax.set_xticks(xticks)
        if ystart is not None and yinc is not None:
            yticks = self.custom_ticks(ystart, yinc, ylog, ylim)
            ax.set_yticks(yticks)
        
        #self.gradient_image(ax, transform=ax.transAxes, extent=(*xlim,*ylim), cmap=bgcm, aspect='auto')
        if twinx == True:
            ax.tick_params(axis = "both", which ="both", bottom = not hide_xtick, top = not hide_xtick, left = False,          right = not hide_ytick)
        elif self.subplot_map.get(subplot_name_+"-t") is not None:     
            ax.tick_params(axis = "both", which ="both", bottom = not hide_xtick, top = not hide_xtick, left = not hide_ytick, right = False)
        else:
            ax.tick_params(axis = "both", which ="both", bottom = not hide_xtick, top = not hide_xtick, left = not hide_ytick, right = not hide_ytick)
        
        #set labels
        if plottype is not None:
            if xlabel is None: xlabel = self.xlabel_map[plottype]
            if ylabel is None: ylabel = self.ylabel_map[plottype]
            
        ax.set_xlabel(xlabel, fontsize = label_fontsize)
        ax.set_ylabel(ylabel, fontsize = label_fontsize)
        if self.projection == '3d':
            ax.set_zlabel(zlabel, fontisze = label_fontsize)

    def custom_ticks(self, tick_start, tick_inc, log, lim):
        i = 0
        ticks = []
        if log == False:
            while tick_start+tick_inc*i <= lim[1]:
                ticks.append(tick_start+tick_inc*i)
                i += 1
        else:
            while tick_start+tick_inc**i <= lim[1]:
                ticks.append(tick_start+tick_inc**i)
                i += 1
        return ticks

    def feature_padding(self, data_len, feature, pad_obj):
        if feature is None:           feature_  = [pad_obj]*data_len
        elif len(feature) < data_len: feature_  = feature + [pad_obj]*(data_len-len(feature))
        else:                         feature_  = feature[:]
        return feature_

    def show(self):
        plt.show()

    def save(self, filename = 'zlabplot.png', dpi = 'figure', transparent = True):
        plt.savefig(filename, dpi = dpi, transparent = transparent)

    def clear(self):
        self.subplot_map = {}
        self.plot_data_map = {}
        plt.close('all')