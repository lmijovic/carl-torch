# generate toy events 
import csv
import numpy as np

def make_sample(mu=0,sigma=1,nevents=1000,startevent=1):
    s = np.random.normal(mu, sigma, nevents)
    # toy event index:
    enum = np.arange(startevent, startevent+nevents, 1, dtype=int)
    # concatenate 
    s = np.append(enum.reshape(nevents,1), s.reshape(nevents,1), axis=1)
    return(s)

def write_to_csv(sample,filename):    
    with open(filename,'w') as tofile:
        writer = csv.writer(tofile,delimiter=",")
        writer.writerow(('eventnumber','x'))
    with open (filename,'a') as tofile:
        np.savetxt(tofile, s, delimiter=',')
    return

#main

# reference sample:
s = make_sample(mu=0,sigma=1,nevents=10000,startevent=1)
# reference file, which will get the weights to reweight it to newfile
write_to_csv(s,"old.csv")

# new sample, to which we reweight:
s = make_sample(mu=0.0,sigma=2.0,nevents=20000,startevent=100000)
# reference file, which will get the weights to reweight it to newfile
write_to_csv(s,"new.csv")

