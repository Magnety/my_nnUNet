import pickle
path = 'plans.pkl'
f = open(path,'rb')
data = pickle.load(f)
print(data)