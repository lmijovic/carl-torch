import json
import argparse

# params
parser = argparse.ArgumentParser(description='Predict carl weights')
parser.add_argument('--patch', required=False, help='hyperparameters patch in .json format', default='')

args = parser.parse_args()
if (args.patch!=""):
    nodevals=[]
    with open(args.patch) as f:
        patch_hyperparas=json.load(f) 
        print(patch_hyperparas)
        print(type(patch_hyperparas))
        for k,v in patch_hyperparas.items():
            if ('epochs'==k):
                epochs=v
            elif ('nodes'==k):
                for nv in v.split(","):
                    nodevals.append(int(nv))
            else:
                print("Warning, the patch contains unknown key",k)
    print(nodevals)
    print(tuple(nodevals))
