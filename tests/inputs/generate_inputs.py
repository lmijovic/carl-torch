# generate toy events
# generate 1D gaussians for input features
# we also test event number functionality needed for ATLAS reweighting

import csv
import numpy as np

def make_sample(mu=0.,sigma=1.,nevents=1000,start_eventnumber=1,weight=1):
    s = np.random.normal(mu, sigma, nevents)
    # toy event index:
    enum = np.arange(start_eventnumber, start_eventnumber+nevents, 1, dtype=int)
    # toy weight:
    eventweight = weight*np.ones(nevents)
    # concatenate 
    s = np.append(enum.reshape(nevents,1), s.reshape(nevents,1), axis=1)
    s = np.append(s,eventweight.reshape(nevents,1), axis=1)
    return(s)

def write_to_csv(sample,filename):    
    with open(filename,'w') as tofile:
        writer = csv.writer(tofile,delimiter=",")
        #writer.writerow(('eventnumber','x'))
    with open (filename,'a') as tofile:
        np.savetxt(tofile, s, delimiter=',')
    return

#main

# reference sample:
start_en=1
nevents=100000
s = make_sample(mu=0,sigma=1,nevents=nevents,start_eventnumber=start_en,weight=1)
# reference file, which will get the weights to reweight it to newfile
write_to_csv(s,"old.csv")

# asssign the 2nd sample distinct event numbers 
start_en+=(start_en+nevents)*10
nevents=100000
# new sample, to which we reweight:
s1 = make_sample(mu=-0.5,sigma=0.3,nevents=nevents,start_eventnumber=start_en,weight=0.5)
s2 = make_sample(mu=0.5,sigma=0.3,nevents=nevents,start_eventnumber=start_en+nevents+1,weight=2)
s=np.concatenate((s1,s2), axis=0)
# write file with distribution to which we want to reweight
write_to_csv(s,"new.csv")

