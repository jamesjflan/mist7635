#%%  
import pandas as pd

#Put classes here for work done across models 

# clean data 
class CleanData:
    def convert_draft_pick(self, data):
        data['NFL Draft Pick'] = data['NFL Draft Pick'].map({'Yes': 1, 'No': 0})
        return data
    
    def fill_na(self, data, value):
        data.fillna(value, inplace=True)
        return data
