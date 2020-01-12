# A Program that gives the value of present-day human contamination (with error bars) (c_mle) from simulated-data, by implementing the model given in OUR recent WORK (https://doi.org/10.1093/bioinformatics/btz660)

# c_mle stands for the maximum likelihood estimate of contamination

# doc stands for the depth of coverage

# steps to simulate data: check 'simulating_data_steps.pdf' of this repository

# generates plots like as given in the *error_bars* folder (see 'Figures/error_bars/' of this repository) 


import numpy                                    
from numpy import log, random

import scipy.special
from scipy.optimize import minimize                    

import matplotlib.pyplot as plt                         
   
#___________________________________________________________________________________________

# to implement the different choices for the frequency distribution of alleles present in the contaminating population

def power_distr(freq_min, freq_max, y, exp_pow):
    return((freq_max**(exp_pow + 1) - freq_min**(exp_pow + 1))*y  + freq_min**(exp_pow + 1))**(1/(exp_pow + 1.0))


def different_frequency_distr_contaminating_pop(choice, number_site):
    freq_cont = numpy.full(number_site, 0.0)
    
    if(choice==1): # Uniform distribution for the frequencies (0, 1)
        freq_min = 0.00
        freq_max = 1.00
        
        freq_cont = numpy.random.uniform(low = freq_min, high = freq_max, size = number_site)
    
    if(choice==2):  # Power distribution for the frequencies (0, 1)
        freq_min = 0.00001
        freq_max = 1.00
        exp_pow = -0.99999
        
        for i in range(number_site):
            freq_cont[i] = power_distr(freq_min, freq_max, numpy.random.uniform(0, 1), exp_pow)
   
# exp_pow is the exponent of power law distribution! P(x) = C x^{exp_pow}

    return freq_cont
#_____________________________________________________________________________________
    
def simulating_data(contam_rate, freq_cont, number_site, doc, endog_freq):
    # this function is for simulating data  
 
    dummy_1 = numpy.full(number_site, 0)
    dummy_2 = numpy.full(number_site, 0)
    
    number_base_A = numpy.full(number_site, 0)
    number_base_C = numpy.full(number_site, 0)
    total_bases =  numpy.full(number_site, 0)
    
    num_bases_contam = numpy.full(number_site, 0)
    jm = []

#_________________________________________________________________________
   # simulating the number of total bases or alleles at each site
   # controlling the depth of coverage
    total_bases = numpy.round(numpy.asarray(numpy.random.poisson(doc, size = number_site))) 
#_____________________________________________________________________________________
# simulating the number of total bases or alleles at each site (num_bases_contam) coming from the contaminating individuals
# controllling the contamination rate to be simulated
    
    for i in total_bases:
        if(i!= 0):
            jm.append(numpy.random.binomial(i, contam_rate))
        else:
            jm.append(0)
    
    num_bases_contam = jm 
    
#__________________________________________________________________________
    # simulating the number of base 'A' and 'C' present among num_bases_contam
    # freq_cont = frequency of allele 'A' in the contaminating population
    
    dummy_1 = numpy.random.binomial(num_bases_contam, (freq_cont), size = number_site) 
    dummy_2 = num_bases_contam - dummy_1

    prob = numpy.random.uniform(0.0, 1.0, size = number_site)
    
    for j, rand_no in enumerate(prob):
        
        if(rand_no < endog_freq):   
            number_base_A[j] = (dummy_1[j] + total_bases[j] - num_bases_contam[j]) 
            number_base_C[j] = dummy_2[j]
            
        else:
            number_base_A[j] = dummy_1[j]
            number_base_C[j] = (dummy_2[j] + total_bases[j] - num_bases_contam[j]) 
              
    
    return number_base_A, number_base_C, total_bases; 
#____________________________________________________________________________________-

    
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
  

def calc_cmle(c, seq_site, doc, endog_freq, seq_error, choice): # optimizing the likelihood
    
    freq = different_frequency_distr_contaminating_pop(choice, seq_site)
    n_a, n_c, n_T = simulating_data(c, freq, seq_site, doc, endog_freq)
    
    x0 = 0.00001 # the initial guess
    cmle = minimize(calc_likeli, x0, args = (freq, n_a, n_c, seq_error, n_T, endog_freq), bounds = [(0.00001, 1.0)]) 
    # cmle = maximum likelihood estimate of 'c' or contamination rate
    return(cmle.x[0])


def avg_simulations(c, doc, seq_site, n_simul, endog_freq, seq_error, choice): # to get the average from many simulations
    contam_rate_all = []
    for _ in range(n_simul):    
        contam_rate_all.append(calc_cmle(c, seq_site, doc, endog_freq, seq_error, choice))
    return(contam_rate_all)

  
def moments(seq_site, doc, endog_freq, seq_error, x_p, choice, n_simu): # to calculate the moments (mean, variance etc.)
      all_avg = []
      all_sd = []
      
      for c in x_p:
          all_esti = avg_simulations(c, doc, seq_site, n_simu, endog_freq, seq_error, choice)
          all_avg.append(numpy.mean(all_esti))
          all_sd.append(numpy.std(all_esti))
      return(all_avg, all_sd)

# exploring the parameter space for seq_site (#sites), doc (depth of coverage), endogenous frequency (endog_freq)     
c1 = []
c2 = []  # dummy lists
c3 = []
c4 = []

n_simu = 10 # number of simulations
seq_site = 1000 # number of polymorphic sites
endog_freq = [0.20, 0.40, 0.50, 0.90] # endogenous frequency
doc = 0.25  # depth of coverage
seq_error = 0.0 # sequencing error

x_p = numpy.arange(0.0, 1.01, 0.1) 
choice_var_freq_distr = 2   # choice of frequency distr., 1: uniform, 2: power law distr.

choice_var_no_sites = 0  #sites = a variable,'doc' and 'endog_freq' = fixed
choice_var_doc = 0  # if equal to 1, then 'doc' = a variable, '#sites' and 'endog_freq' = fixed
choice_var_endog_freq = 1  # if equal to 1, then 'endogenous frequency' = a variable, '#sites' and 'doc' = fixed

color = ['r', 'b', 'k', 'g']  
width = [0.6, 0.6, 0.6, 0.6] 
  
fig, axs = plt.subplots(2, 2)
fig.subplots_adjust(hspace = 0.70, wspace = 0.50)

axs = axs.ravel()

if(choice_var_endog_freq==1):
    for i in endog_freq:
        c1, c2 = moments(seq_site, doc, i, seq_error, x_p, choice_var_freq_distr, n_simu)
        c3.append(c1)
        c4.append(c2)
        
    for i in range(len(endog_freq)):
        axs[i].errorbar(x_p, c3[i], c4[i], alpha = 0.5, ecolor = color[i], capsize = 1, c = color[i], lw = width[i])
        axs[i].set_title("Endogenous Frequency = " + str(endog_freq[i])) 
        axs[i].set_xlabel('c (expected)') 
        axs[i].set_ylabel('c (estimated)')
    fig.suptitle('#sites = 1000, doc = 0.25X') 
    plt.savefig('/Users/jyotidalal/Desktop/variation_with_endog_frequency.pdf')
    
    plt.show()      
    
        
if(choice_var_doc==1):
    for i in doc:
        c1, c2 = moments(seq_site, i, endog_freq, seq_error, x_p, choice_var_freq_distr, n_simu)
        c3.append(c1)
        c4.append(c2)
        
    for i in range(len(doc)):
        axs[i].errorbar(x_p, c3[i], c4[i], alpha = 0.5, ecolor = color[i], capsize = 1, c = color[i], lw = width[i])
        axs[i].set_title(" depth of coverage = " + str(doc[i])) 
        axs[i].set_xlabel('c (expected)') 
        axs[i].set_ylabel('c (estimated)') 
    fig.suptitle('#sites = 1000, endog freq = 0.5, #simulation = 10') 
    plt.savefig('/Users/jyotidalal/Desktop/variation_with_depth_of_coverage.pdf')
    
    plt.show()
    #plt.savefig('/Users/jyotidalal/Desktop/.....eps')
        
if(choice_var_no_sites==1):
    
    for i in seq_site:
        c1, c2 = moments(i, doc, endog_freq, seq_error, x_p, choice_var_freq_distr, n_simu)
        c3.append(c1)
        c4.append(c2) 
        
    for i in range(len(seq_site)):
        axs[i].errorbar(x_p, c3[i], c4[i], alpha = 0.5, ecolor = color[i], capsize = 1, c = color[i], lw = width[i])
        axs[i].set_title("# sites = " + str(seq_site[i])) 
        axs[i].set_xlabel('c (expected)') 
        axs[i].set_ylabel('c (estimated)') 
    fig.suptitle('doc = 1.00 X, endog freq = 0.5') #
   
    plt.savefig('/Users/jyotidalal/Desktop/variation_with_sites.pdf')
    plt.show()
