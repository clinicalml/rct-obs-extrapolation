import yaml 
from yaml.loader import SafeLoader

def read_yaml(path='configs/demo.yaml'): 
    ''' 
        Helper function to read in config yaml.
    '''
    with open(path) as f:
        data = yaml.load(f, Loader=SafeLoader)
    return data 