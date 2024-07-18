import sys

import yaml

from agent import Agent

def usage():
    print("Invalid usage:")
    print("Arguments: ")
    print("--training: Runs in training mode")
    print("--parameters [file] [heading]: Sets hyperparamters")
    

if __name__ == "__main__":
    
    training = False
    hyperparameters = None
    
    i = 1
    while i < len(sys.argv):
        
        arg = sys.argv[i] 

        match arg:
            case '--training':
                training = True
            case '--parameters':
                with open(sys.argv[i+1], 'r') as file:
                    hyperparameters = yaml.safe_load(file)
                    hyperparameters = hyperparameters[sys.argv[i+2]]
                i += 2
            case other:
                usage()
                
        
        
        i += 1
    
    if hyperparameters == None:
        with open("hyperparameters.yaml", 'r') as file:
            hyperparameters = yaml.safe_load(file)
            hyperparameters = hyperparameters["test1"]
    
    a = Agent(hyperparameters)
    a.run(training)