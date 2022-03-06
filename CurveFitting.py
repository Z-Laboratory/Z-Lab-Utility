import numpy as np
from scipy.optimize import curve_fit

class FitModel:
    def __init__(self, function):
        self.function = function
        self.fit_function = function

    def fix_const(self,const_value):
        def fit_function(x,*argv):
            fit_arvg = []
            index_const = 0
            index_arvg = 0
            for fg in self.const_flag:
                if fg:
                    fit_arvg.append(const_value[index_const])
                    index_const += 1
                else:
                    fit_arvg.append(argv[index_arvg])
                    index_arvg +=1
            return self.function(x,*fit_arvg)
        self.fit_function = fit_function

    def fit(self, xdata, ydata, yerr=None, p0=None, bounds=(-np.inf,np.inf), const_flag=None, absolute_sigma=False):
        new_p0 = []
        new_bounds = [[],[]]
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
            
        popt, pcov = curve_fit(self.fit_function,xdata,ydata,p0=new_p0,bounds=new_bounds,sigma=yerr,absolute_sigma=absolute_sigma)
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
        return self.function(xdata, *self.popt)
    
    def fit_transform(self, xdata, ydata, yerr=None, p0=None, bounds=(-np.inf,np.inf), const_flag=None, absolute_sigma=False):
        self.fit(xdata, ydata, yerr=yerr, p0=p0, bounds=bounds, const_flag=const_flag, absolute_sigma=absolute_sigma)
        return self.transform(xdata), self.popt, self.perr

        
