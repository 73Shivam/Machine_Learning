import pickle
import numpy as np

with open('poly_reg_features.p', 'rb') as f:
    pf = pickle.load(f)
    print('model loaded')

with open('poly_reg_salary_prediction.p', 'rb') as f:
    preg = pickle.load(f)
    print('model loaded')

position = input('enter a position')

xinp = np.array([[float(position)]])
xinp = pf.transform(xinp)
result = preg.predict(xinp)
print(result[0])