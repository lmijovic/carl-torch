# generate toy events
# generate 2D gaussians with correlated features 
# we also test event number functionality needed for ATLAS reweighting

import csv
import numpy as np

def make_sample(mu=0.,sigma=1., corr=0.5, nevents=1000,start_eventnumber=1):
    s1 = np.random.normal(mu, sigma, nevents)
    s2 = corr*s1 + (1-corr)*np.random.normal(mu, sigma, nevents)
    # toy event index:
    enum = np.arange(start_eventnumber, start_eventnumber+nevents, 1, dtype=int)
    s = np.append(enum.reshape(nevents,1), s1.reshape(nevents,1), axis=1)
    s = np.append(s, s2.reshape(nevents,1), axis=1)
    #print(type(s))
    #print(s)
    #s = np.append(s1.reshape(nevents,1), s2.reshape(nevents,1), axis=1)
    #s = np.append(enum.reshape(nevents,1), axis=1)
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
nevents=10000
s = make_sample(mu=0,sigma=1,corr=0.5, nevents=nevents,start_eventnumber=start_en)
# reference file, which will get the weights to reweight it to newfile
write_to_csv(s,"old_2d.csv")



# asssign the 2nd sample distinct event numbers 
start_en+=(start_en+nevents)*10
nevents=5000
# new sample, to which we reweight:
s1 = make_sample(mu=0.5,sigma=0.35,corr=-0.5, nevents=nevents,start_eventnumber=start_en)
s2 = make_sample(mu=0.5,sigma=0.35,corr=-0.5, nevents=nevents,start_eventnumber=start_en)
s=np.concatenate((s1,s2), axis=0)
# write file with distribution to which we want to reweight
write_to_csv(s,"new_2d.csv")

