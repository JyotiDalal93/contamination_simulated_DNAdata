# program to simulate the data implementing the model given in OUR recent WORK: https://doi.org/10.1093/bioinformatics/btz660) for a fixed contamination rate followed by writing the output to a 'anything.txt' file

# doc stands for the depth of coverage

# steps to simulate data: check 'simulating_data_steps.pdf' of this repository
# see 'demo.txt' containing simulated data, present in the folder 'Figure' of this repository

import numpy                                    
from numpy import log, random

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
    # freq_cont = frequency of allele 'A' in the contaminating population
    
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

    #simulating the number of base 'A' and 'C' present among num_bases_contam
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

#_________________________________________________________________________
    
def writing_data_file(no_site, doc, contam_rate, endog_freq, choice): # to write simulated-data to a '.txt' file
     freq = different_frequency_distr_contaminating_pop(choice, no_site)
     no_base_A, no_base_C, total_bases=  simulating_data(contam_rate, freq, no_site, doc, endog_freq)
     numpy.savetxt('/Users/jyotidalal/Desktop/demo.txt', numpy.c_[no_base_A, no_base_C, total_bases, freq], fmt = '%.10g', delimiter='\t\t', header = 'no_A\t\tno_C\t\tnT\t\tfreq', footer ='at the nd of file aayega') 
     return();
     
writing_data_file(no_site = 1000, doc = 1, contam_rate = 0.5, endog_freq = 0.5, choice = 1)    
# here, contamination rate = 30% (that's quite a lot!)
