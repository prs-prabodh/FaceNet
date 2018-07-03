#code to erase off database without using GUI
import numpy as np
w=dict({})
np.save('./database.npy',w)
