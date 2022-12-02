#
#  CurveFitting.py
#  
#  Copyright (c) 2022 Z-Group. All rights reserved.
#  -----------------------------------------------------
#  Current developers  : Shao-Chun Lee    (2022 - Present)
#  -----------------------------------------------------
#  
# CurveFitting is a modulus for general curve fitting based on scipy.optimize.curve_fit.
# The "const_flag" variable allows users to fix the paramters via redefining the fitted function,
# which will change the size of the covariance matrix when evaluating the uncertainties of the parameters.
# see https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.curve_fit.html 
# for more information about scipy.optimize.curve_fit

# Example code to use CurveFitting:
# import CurveFitting as CF
# def myfunction(x, a, b, c,...):
#     ...
#     return y
# mymodel = CF.Model(function = myfunction)
# fity, popt, perr = mymodel.fit_transform(xdata = xdata, ydata = ydata, yerr = yerr, \
#                                          p0 = p0, bounds = [lowerbound, upperbound], \
#                                          const_flag = const_flag)

import numpy as np
from scipy.optimize import curve_fit

class Model:
    def __init__(self, function):
        self.function = function
        self.fit_function = function

    def fix_const(self, const_value):
        def fit_function(x, *argv):
            fit_arvg = []
            index_const = 0
            index_arvg = 0
            for fg in self.const_flag:
                if fg:
                    fit_arvg.append(const_value[index_const])
                    index_const += 1
                else:
                    fit_arvg.append(argv[index_arvg])
                    index_arvg += 1
            return self.function(x, *fit_arvg)
        self.fit_function = fit_function

    def fit(self, xdata, ydata, yerr = None, p0 = None, bounds = (-np.inf, np.inf), const_flag = None, absolute_sigma = False):
        # argument
        #   xdata:      np.array
        #   ydata:      np.array
        #   yerr:       np.array or None
        #   p0:         list[float], initial values of fitting parameters
        #   lowerbound: list[float] or (float, float), 
        #   upperbound: list[float] or (float, float),
        #               upperbound/lowerbound of fitting parameters, if upperbound/lowerbound = (float, float), the boundaries are broadcasted to all parmeters.
        #   const_flag: list[bool],  set fitting parameters to constant according to p0
        #   absolute_sigma: bool
        # return
        #   popt:       np.array, optimized parameters
        #   perr:       np.array, uncertainty of the optimized parameters
        new_p0 = []
        new_bounds = [[], []]
        self.const_flag = const_flag
        const_value = []
        if self.const_flag:
            for index, fg in enumerate(self.const_flag):
                if fg:
                    const_value.append(p0[index])
                else:
                    new_p0.append(p0[index])
                    new_bounds[0].append(bounds[0][index])
                    new_bounds[1].append(bounds[1][index])
            self.fix_const(const_value)
        else:
            new_p0 = p0
            new_bounds = bounds
            self.fit_function = self.function
            
        popt, pcov = curve_fit(self.fit_function, xdata, ydata, p0 = new_p0, bounds = new_bounds, sigma = yerr, absolute_sigma = absolute_sigma)
        perr = np.sqrt(np.diag(pcov))
        self.popt = []
        self.perr = []
        index_const = 0
        index_arvg = 0
        if self.const_flag:
            for fg in self.const_flag:
                if fg:
                    self.popt.append(const_value[index_const])
                    self.perr.append(0)
                    index_const += 1
                else:
                    self.popt.append(popt[index_arvg])
                    self.perr.append(perr[index_arvg])
                    index_arvg += 1
            self.popt = np.array(self.popt)
            self.perr = np.array(self.perr)
        else:
            self.popt = popt
            self.perr = perr

        return self.popt, self.perr
    
    def transform(self, xdata):
        # evaluate functional values at xdata with optimized parameters
        return self.function(xdata, *self.popt)
    
    def fit_transform(self, xdata, ydata, yerr = None, p0 = None, bounds = (-np.inf, np.inf), const_flag = None, absolute_sigma = False):
        # equivalent to fit + transform
        self.fit(xdata, ydata, yerr = yerr, p0 = p0, bounds = bounds, const_flag = const_flag, absolute_sigma = absolute_sigma)
        return self.transform(xdata), self.popt, self.perr

        
