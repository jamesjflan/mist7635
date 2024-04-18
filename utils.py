#%%  
import pandas as pd

#Put classes here for work done across models 

# clean data CLASS 

class CleanData:
    #def __init__(self, data):
    #    self.data = data
    
    def convert_draft_pick(data):
        data['NFL Draft Pick'] = data['NFL Draft Pick'].map({'Yes': 1, 'No': 0})
        return data
    
    def fill_na(data, value):
        data.fillna(value, inplace=True)
        return data
    

