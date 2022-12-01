#
#  LiquidLibIO.py
#  
#  Copyright (c) 2022 Z-Group. All rights reserved.
#  -----------------------------------------------------
#  Current developers  : Shao-Chun Lee    (2022 - Present)
#  -----------------------------------------------------

import numpy as np

def smooth(x, y, n_points = 1000, sg_filter = True, sg_window = 101):
    #use CubicSpline and savgol_filter to smooth a function when the data points are too scarce
    #argument
    #   x:         np.array(float) (1D) 
    #   y:         np.array(float) (1D)
    #   sg_window: int
    #return
    #   x_smth:         np.array(float) (1D)
    #   y_smth:         np.array(float) (1D)
    from scipy.interpolate import CubicSpline
    from scipy.signal import savgol_filter
    x_smth = np.linspace(x.min(), x.max(), n_points)
    cs = CubicSpline(x, y)
    y_smth = cs(x_smth)
    if sg_filter: y_smth = savgol_filter(y_smth, sg_window, 3)
    return x_smth, y_smth
        
def sg_smooth(x, y, sg_window=11):
    #use avgol_filter to smooth a function when the data points are too dense and noisey
    #argument
    #   x:         np.array(float) (1D) 
    #   y:         np.array(float) (1D)
    #   sg_window: int
    #return
    #   x:         np.array(float) (1D)
    #  sg:         np.array(float) (1D)
    from scipy.signal import savgol_filter
    sg = savgol_filter(y, sg_window, 3)
    return x, sg

def compute_sk_abs_prefactor(atom_name, atom_num, volume, weighting_type = "neutron", k_values=[]):
    #argument
    #   atom_name:      list[str], nemas of the atoms 
    #                   see ScatteringLengthTable.scattering_length_table and AtomicFormFactorTable.atomic_form_factor_table 
    #                   for available names.
    #   atom_num:       list[int], number of the atoms
    #   volume:         float, volume of the simulation box
    #   weighting_type: 'neutron' or 'xray'
    #   k_values:       np.array(float), only specify k_values when weighting_type = 'xray'
    #return
    #   prefactor:      float           if weighting_type = 'neutron' 
    #                   np.array(float) if weighting_type = 'xray'

    #mutiply normalized sk with prefactor to get absolute sk
    #unit of scattering lengths: fm
    #unit of volume:             angstrom^3
    #unit of prefactor:          cm^-1
    from AtomicFormFactorTable import scattering_length_table, atomic_form_factor_table
    unit_prefactor = 1e-2 # cm^-1 = 1.0 (fm^2) * 1e-30 (m^2/fm^2) * 1e+30 (A^3/m^3) * 1e-2 m/cm
    N = sum(atom_num)
    if weighting_type == 'neutron':
        sum_scattering_length_square = 0.0
        for i_particle in range(len(atom_name)):
            sum_scattering_length_square += scattering_length_table[atom_name[i_particle]]*atom_num[i_particle]
        sum_scattering_length_square *= sum_scattering_length_square
        prefactor = sum_scattering_length_square/N/volume * unit_prefactor
    elif weighting_type == 'xray':
        k_values = np.array(k_values)
        atomic_form_factor_ = np.zeros((len(atom_num),len(k_values)))
        sum_atomic_form_factor_square = 0.0
        for i_particle in range(len(atom_name)):
            for i_coefficient in range(0,len(atomic_form_factor_table[atom_name[i_particle]])-1,2):
                atomic_form_factor_[i_particle] += atomic_form_factor_table[atom_name[i_particle]][i_coefficient] * np.exp(-atomic_form_factor_table[atom_name[i_particle]][i_coefficient+1]*((k_values/4.0/np.pi)**2))
            atomic_form_factor_[i_particle] += atomic_form_factor_table[atom_name[i_particle]][len(atomic_form_factor_table[atom_name[i_particle]])-1]
            sum_atomic_form_factor_square += atomic_form_factor_[i_particle]*atom_num[i_particle]
        sum_atomic_form_factor_square *= sum_atomic_form_factor_square
        sum_atomic_form_factor_square /= k_values.shape[0]**2
        prefactor = sum_atomic_form_factor_square/N/volume * unit_prefactor
    else:
        prefactor = N/volume * unit_prefactor
    return prefactor

def compute_absolute_cross_term_structure_factor(sk_n, sk_n1, sk_n2, pref_n, pref_n1, pref_n2):
    #argument
    #   sk_n:    np.array(float) (1D), normalized sk of n1+n2
    #   sk_n1:   np.array(float) (1D), normalized sk of n1
    #   sk_n2:   np.array(float) (1D), normalized sk of n2
    #   pref_n:  float or np.array(float) (1D), absolute prefactor of n1+n2
    #   pref_n1: float or np.array(float) (1D), absolute prefactor of n1
    #   pref_n2: float or np.array(float) (1D), absolute prefactor of n2
    #return
    #   sk_abs_cross: np.array(float), absolute sk of n1xn2
    sk_abs_cross = sk_n*pref_n - sk_n1*pref_n1 - sk_n2*pref_n2 
    return sk_abs_cross

def compute_normalized_cross_term_structure_factor(sk_n, sk_n1, sk_n2, pref_n, pref_n1, pref_n2):
    #argument
    #   same as compute_absolute_cross_term_structure_factor
    #return
    #   sk_norm_cross: np.array(float), absolute sk of n1xn2
    sk_norm_cross = 0.5*(compute_absolute_cross_term_structure_factor(sk_n, sk_n1, sk_n2, pref_n, pref_n1, pref_n2))/(pref_n1*pref_n2)**0.5
    return sk_norm_cross

def make_molecule_file(output_filename, atom_type, molecule_name, n_atom_in_molecule, atom_type_of_first_atom_in_molecule):
    #create molecule_file for "molecule_file_path" in LiquidLib
    #argument
    #   output_filename: str
    #   atom_type: list[int] or list[str], atom type
    #   molecule_name: list[str], list of molecule names
    #   n_atom_in_molecule: list[int], list of number of atoms in each molecule
    #   atom_type_of_first_atom_in_molecule: list[int], list of atom type (in atom_type) of the first atom in each molecule
    #                                        we always assume the all the atoms in each molecule are listed consecutively in atom_type
    #                                        and the atoms for each molecule are sorted the same way, the same element in different molecules are named differtly.
    #return
    #   None
    #example for 3 water molecules and 2 oxygen molecules: O of water = 1, H of water = 2, O of oxygen = 3
    #atom_type                           = [1,2,2,1,2,2,3,3,1,2,2,3,3]
    #molecule_name                       = ['water','oxygen']
    #n_atom_in_molecule                  = [3, 2]
    #atom_type_of_first_atom_in_molecule = [1, 3]
    #output file:
    '''
    #
    13
    1 water
    1 water
    1 water
    2 water
    2 water
    2 water
    3 oxygen
    3 oxygen
    4 water
    4 water
    4 water
    5 oxygen
    5 oxygen
    '''
    with open(output_filename,"w") as fout:
        fout.write("#\n%d\n"%(len(atom_type)))
        molecule_id = 0
        i = 0
        while i < len(atom_type):
            j = atom_type_of_first_atom_in_molecule.index(atom_type[i])
            molecule_id += 1
            for k in range(n_atom_in_molecule[j]):
                fout.write("%d %s\n"%(molecule_id, molecule_name[j]))
            i += n_atom_in_molecule[j]

def read(filename, target_k_index = -1, target_k = 0):
    #note: BondOrientationalOrderParameter and Van-Hove functions have not been included yet
    #read output files from LiquidLib
    #argument
    #   filename: str
    #   target_k_index: int
    #   target_k: float
    #return
    #   if quantity in ["gr", "sk", "r2t", "mr2t", "alpha_2", "chi_4", "eacf"]
    #      x: np.array(float) (1D)
    #      y: np.array(float) (1D)
    #   if quantity in ["fskt", "fkt"]
    #      if target_k > 0 (fitting is involved)
    #           t:    np.array(float) (1D)
    #           f_kt: np.array(float) (1D)
    #      if target_k_index > -1
    #           k:    float
    #           t:    np.array(float) (1D)
    #           f_kt: np.array(float) (1D)
    #      otherwise
    #           k:    np.array(float) (1D)
    #           t:    np.array(float) (1D)
    #           f_kt: np.array(float) (2D)
    #   quantity: str
    quantity = ""
    with open(filename,"r") as fin:
        aline1 = fin.readline()
        if "Self intermediate scattering function" in aline1:         quantity="fskt"
        elif "Collective intermediate scattering function" in aline1: quantity="fkt"
        aline2 = fin.readline()
        if quantity == "":
            if "S(k)" in aline2:        quantity = "sk"
            elif "g(r)" in aline2:      quantity = "gr"
            elif "r2(t)" in aline2:
                if "Mutual" in aline1:  quantity += "mr2t"
                else:                   quantity = "r2t"
            elif "alpha_2" in aline2:   quantity = "alpha_2"
            elif "chi_4" in aline2:     quantity = "chi_4"
            elif "C_jj" in aline2:      quantity = "eacf"
        if quantity in ["gr", "sk", "r2t", "mr2t", "alpha_2", "chi_4", "eacf"]:
            x = []
            y = []
            for aline in fin:
                if "#" not in aline:
                    linelist = aline.strip().split()
                    x.append(float(linelist[0]))
                    y.append(float(linelist[1]))
            x = np.array(x)
            y = np.array(y)
            return x, y, quantity
        elif quantity in ["fskt", "fkt"]:
            k = []
            t = []
            f_tk = []
            aline = fin.readline()
            while "#" not in aline:
                k.append(float(aline.strip()))
                aline = fin.readline()
            for aline in fin:
                linelist = aline.strip().split()
                t.append(float(linelist[0]))
                f_tk.append(np.array([float(i) for i in linelist[1:]]))
            k = np.array(k)
            t = np.array(t)
            f_tk = np.array(f_tk)
            f_kt = f_tk.transpose()
            if target_k > 0 and target_k_index > -1:
                print("Error: Both target_k and target_k_index specified.")
                exit()
            elif target_k > 0:
                #use leastsq to fit f_tk along k assuming gaussian shape
                from scipy.optimize import leastsq
                f_kt_fitk = []
                gaussian = lambda p, x: p[0]*np.exp(-p[1]*(x**2))
                error_gaussian  = lambda p, x, y : y - gaussian(p, x)
                fit_coeff = []
                for a_f_tk in f_tk:
                    init  = [1, 0.001]
                    coeff = leastsq(error_gaussian, init, args=(k, a_f_tk))[0]
                    f_kt_fitk.append(gaussian(coeff[:],target_k))
                return t, np.array(f_kt_fitk), quantity                
            elif target_k_index > -1:
                return k[target_k_index], t, f_kt[target_k_index], quantity
            else:
                return k, t, f_kt, quantity
        else:
            print("Error: quantity not recognize.")
            exit()


def write(quantity,input_filename,trajectory_file_path,output_file_path,\
          start_frame,end_frame,frame_interval,number_of_frames_to_average,\
          time_scale_type="log",trajectory_delta_time=1,time_interval=1.2,number_of_time_points=20,\
          calculation_type="atom",gro_file_path=None,molecule_file_path=None,dimension=3,\
          atom_name_1=None,atom_name_2=None,mass_1=None,mass_2=None,charge_1=None,charge_2=None,molecule_name_1=None,molecule_name_2=None,\
          weighting_type=None,atomic_form_factor_1=None,atomic_form_factor_2=None,scattering_length_1=None,scattering_length_2=None,\
          input_box_length=None,k_start_value=0,k_end_value=5,k_interval=0.01,\
          include_intramolecular=None,number_of_bins=400,max_cutoff_length=10,\
          overlap_length=1):
    #argument
    #   the same as LiquidLib input
    #   note: atom_name, mass, charge, atomic_form_factor, and scattering_length are all str, not list
    quantity_function_map = {"sk" :"StructureFactor",\
                             "gr" :"PairDistributionFunction",\
                             "fskt":"SelfIntermediateScatteringFunction",\
                             "fkt":"CollectiveIntermediateScatteringFunction",\
                             "r2t":"MeanSquaredDisplacement",\
                             "mr2t":"MutualMeanSquaredDisplacement",\
                             "msd":"MeanSquaredDsiplacement",\
                             "chi4":"FourPointCorrelationFunction",\
                             "eacf":"ElectricCurrentAutocorrelationFunction"}
    quantity_function = quantity_function_map[quantity]
    with open(input_filename,"w",newline='\n') as fout:
        #general information
        fout.write('''-function=%s
-calculation_type=%s
-trajectory_file_path=%s'''%(quantity_function,calculation_type,trajectory_file_path))
        if gro_file_path:
            fout.write('''
-gro_file_path=%s'''%(gro_file_path))
        if molecule_file_path:
            fout.write('''
-molecule_file_path=%s'''%(molecule_file_path))
        fout.write('''
-output_file_path=%s'''%(output_file_path))
        fout.write('''
-start_frame=%s
-end_frame=%s
-frame_interval=%s
-dimension=%s
-number_of_frames_to_average=%s'''%(start_frame,end_frame,frame_interval,dimension,number_of_frames_to_average))
        if quantity in ['sk','fskt','fkt'] and weighting_type:
            fout.write('''
-weighting_type=%s'''%(weighting_type))
        #time correlation
        if quantity in ['msd','r2t','mr2t','fskt','fkt','chi4','eacf']:
            fout.write('''
-time_scale_type=%s
-trajectory_delta_time=%s
-time_interval=%s
-number_of_time_points=%s
#-time_array_indices='''%(time_scale_type,trajectory_delta_time,time_interval,number_of_time_points))
        #first atom group
        if atom_name_1:
            fout.write('''
-atom_name_1=%s'''%(atom_name_1))
        if molecule_name_1:
            fout.write('''
-molecule_name_1=%s'''%(molecule_name_1))
        if quantity in ['sk','fskt','fkt']:
            if atomic_form_factor_1:
                fout.write('''
-atomic_form_factor_1=%s'''%(atomic_form_factor_1))
            if scattering_length_1:
                fout.write('''
-scattering_length_1=%s'''%(scattering_length_1))
        if mass_1:
            fout.write('''
-mass_1=%s'''%(mass_1))
        if charge_1:
            fout.write('''
-charge_1=%s'''%(charge_1))
        #second atom group
        if atom_name_2:
            fout.write('''
-atom_name_2=%s'''%(atom_name_2))
        if molecule_name_2:
            fout.write('''
-molecule_name_2=%s'''%(molecule_name_2))
        if quantity in ['sk','fskt','fkt']:
            if atomic_form_factor_2:
                fout.write('''
-atomic_form_factor_2=%s'''%(atomic_form_factor_2))
            if scattering_length_2:
                fout.write('''
-scattering_length_2=%s'''%(scattering_length_2))
        if mass_2:
            fout.write('''
-mass_2=%s'''%(mass_2))
        if charge_2:
            fout.write('''
-charge_2=%s'''%(charge_2))
        #additional information
        if quantity in ['sk','fskt','fkt']:
            fout.write('''
-k_start_value=%s
-k_end_value=%s
-k_interval=%s
#-max_k_number='''%(k_start_value,k_end_value,k_interval))
        if input_box_length:
            fout.write('''
-input_box_length=%s'''%(input_box_length))
        if quantity in ['gr']:
            if include_intramolecular == None:
                if molecule_file_path: include_intramolecular = True
                else: include_intramolecular = False
            fout.write('''
-include_intramolecular=%s'''%(include_intramolecular))
            fout.write('''
-number_of_bins=%s'''%(number_of_bins))
            fout.write('''
-max_cutoff_length=%s'''%(max_cutoff_length))
        if quantity in ['chi4']:
            fout.write('''
-overlap_length=%s'''%(overlap_length))




    