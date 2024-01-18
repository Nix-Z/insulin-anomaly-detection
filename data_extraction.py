import pandas as pd

def load_data():
  data = pd.read_csv('data.csv')
  return data

load_data()
