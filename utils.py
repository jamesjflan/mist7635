#%%  
import pandas as pd

#Put classes here for work done across models 

# clean data CLASS 

class CleanData:
    def __init__(self, data):
        self.data = data
    
    def convert_draft_pick(self):
        self.data['NFL Draft Pick'] = self.data['NFL Draft Pick'].map({'Yes': 1, 'No': 0})
        return self.data
    
    def fill_na(self, value):
        self.data.fillna(value, inplace=True)
        return self.data
    

