import os
import sys
import numpy as np
from ZLabPlot import ZLabPlot
import LiquidLibIO as LLIO

k_ = []
t_ = []
fkt_ = []
temp_ =[900, 850, 800, 750, 700, 650, 600]
legend = ["%d K"%(i) for i in temp_]
color = [3+i for i in range(len(temp_))]
rcmap = {'axes.titlesize':'small', 'legend.title_fontsize':'small'}
for temp in temp_:
    k, t, fkt, plottype = LLIO.read("fkt/fkt-%dk-n-Zn-Cl.txt"%(temp), target_k_index = 2)
    t_.append(t)
    k_.append(k)
    fkt_.append(fkt/fkt[0])
zp = ZLabPlot(rcmap = rcmap)
zp.add_subplot(plottitle = "Collective Intermediate Scattering Function")
zp.add_data(t_, fkt_, plottype = plottype, lw = 2, ylim = (0.0, 1.1), xlog = True, legendtitle = r'$Q=\ \sim %.1f\ \mathrm{Å^{-1}}$'%(k), cmap = 'OrRd', legend = legend, color = color, ncol = 2)
zp.save(filename = 'fkt', transparent = False)
zp.clear()

t_ = []
r2t_ = []
for temp in temp_:
    t, r2t, plottype = LLIO.read("r2t/r2t-%dk-Zn.txt"%(temp))
    t_.append(t)
    r2t_.append(r2t/100)
zp = ZLabPlot(rcmap = rcmap)
zp.add_subplot(plottitle = "Mean Squared Displacment")
zp.add_data(t_, r2t_, plottype = plottype, lw = 2, xlog = True, ylog = True, legendtitle = r'$MSD\ of\ Zn$', cmap = 'OrRd', legend = legend, color = color, ncol = 2)
zp.save(filename = 'r2t', transparent = False)
zp.clear()

test_x = np.linspace(0., 100., num = 30)
tau = [50*i for i in range(1,11)]
test_y1 = [np.exp(-test_x/i) for i in tau]
test_y2 = [np.exp(test_x/i) for i in tau]
for y in test_y2: y /= y.max()
test_x = [test_x]*len(tau)
zp = ZLabPlot()
zp.add_subplot(subplot_spec = 111, twinx = True, plottitle = "test plot")
zp.add_data(test_x, test_y1, xlabel = "XXX", ylabel = r"$\emph{\textbf{test1}}$", ms = 's',     msize= 9, mf = 'none', lw = [2,3], cmap = None,     max_cmap_index = None)
zp.add_data(test_x, test_y2,                 ylabel = r"$f_{eff}(k, t)$",         ms = ['s']*2, msize= 3, mf = 'left', lw = 0.5,   cmap = 'plasma', twinx = True)
zp.save(filename = 'test1', transparent = False)
zp.clear()
zp = ZLabPlot(rcmap = {'figure.figsize': (12, 12), 'ytick.labelsize': 'medium'})
zp.add_subplot(subplot_name = "1", subplot_spec = 211, twinx = True, plottitle = "test plot")
zp.add_data(test_x, test_y1, subplot_name = "1", xlabel = "XXX", ylabel = "test1", ms = 's', msize= 9, mf = 'none', lw = [2,3])
zp.add_data(test_x, test_y2, subplot_name = "1", ylabel = "test2", ms = ['s']*2, msize= 3, mf = 'left', cmap = 'plasma', lw = 0.5, twinx = True)
zp.add_subplot(subplot_name = "2", subplot_spec = 223)
zp.add_data(test_x, test_y1, subplot_name = "2", ylabel = "test1", ms = 's', msize= 9, mf = 'none', lw = [2,3])
zp.add_subplot(subplot_name = "3", subplot_spec = 224)
zp.add_data(test_x, test_y2, subplot_name = "3", ylabel = "test2", ms = ['s']*2, msize= 3, mf = 'left', cmap = 'plasma', lw = 0.5)
zp.save(filename = 'test2', transparent = False)
zp.clear()

exit()


