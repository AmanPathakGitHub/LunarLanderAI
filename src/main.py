import sys

import yaml

from agent import Agent

def usage():
    print("Arguments: ")
    print("--training: Runs in training mode")
    print("--parameters [file] [heading]: Sets hyperparamters")
    print("--model [path]: loads a model")
    

if __name__ == "__main__":
    
    training = False
    hyperparameters = None
    model_path = None
    
    sys.argv.reverse()
    sys.argv.pop()
    
    while len(sys.argv) > 0:
        
        arg = sys.argv.pop()
        
        match arg:
            case '--training':
                training = True
            case '--parameters':
                with open(sys.argv.pop(), 'r') as file:
                    hyperparameters = yaml.safe_load(file)
                    hyperparameters = hyperparameters[sys.argv.pop()]
            case '--model':
                model_path = sys.argv.pop()
            case '--help':
                usage()
                exit(0)
                
            case other:
                print("Invalid usage:")
                usage()
                exit(0)
                
    
    if hyperparameters == None:
        with open("hyperparameters.yaml", 'r') as file:
            hyperparameters = yaml.safe_load(file)
            hyperparameters = hyperparameters["test1"]
    
    a = Agent(hyperparameters, model_path)
    a.run(training)