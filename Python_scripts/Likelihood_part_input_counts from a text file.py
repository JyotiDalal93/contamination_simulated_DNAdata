# program to calculate contamination fraction (maximum likelihood estimate) from the data ('anything.txt' containing the counts of bases and frequencies)

import numpy                                    
from numpy import log, random

import scipy.special
from scipy.optimize import minimize                    

import matplotlib.pyplot as plt
 
#_______________________________________________________________________________   
                     
def calc_likeli(cont, freq_cont, number_base_A, number_base_C, seq_error, total_bases, endog_freq): 
 # this function calculates the likelihood of data given parameters of the model 
 # cont = contamination rate 
 
  prob_1 = cont*((4/3)*seq_error - 1)*(1 - freq_cont) + (1 - seq_error)
  prob_2 = cont*(1 - (4/3)*seq_error)*(freq_cont) + ((1/3)*seq_error)

  bin_coeff_1 = (endog_freq)*scipy.special.binom(total_bases, number_base_A)
  bin_coeff_2 = (endog_freq)*scipy.special.binom(total_bases, number_base_A)
  
  binom_distr_1 = ((prob_1)**(number_base_A))*((1-prob_1)**(total_bases - number_base_A))
  binom_distr_2 = ((prob_2)**(number_base_A))*((1-prob_2)**(total_bases - number_base_A))

  likeli = (binom_distr_1)*(bin_coeff_1) + (binom_distr_2)*(bin_coeff_2)
  log_likeli = numpy.sum(numpy.log(likeli)) 
  
  t6 = (log_likeli)*(-1.0)

  return(t6)
  
#______________________________________________________________________________________________-  

def calculating_contam(endog_freq, seq_error):
    
    data = numpy.loadtxt('/Users/jyotidalal/Desktop/Counts.txt', usecols = (0, 1, 2, 3)) # 'Counts.txt' contains the counts of bases and frequencies
    
    number_base_A = data[:, [0]]
    number_base_C = data[:, [1]]
    total_bases = data[:, [2]]
    freq = data[:, [3]]
    
    x0 = 0.00001 # the initial guess
    cmle = minimize(calc_likeli, x0, args = (freq, number_base_A, number_base_C, seq_error, total_bases, endog_freq), bounds = [(0.00001, 1.0)]) 
    return(cmle.x)
    
contam_rate = calculating_contam(endog_freq = 0.5, seq_error = 0.0)
print(contam_rate) # OUTPUT = contamination rate
