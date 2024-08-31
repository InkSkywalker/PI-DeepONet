import learner as ln
import pickle
import torch
import learner.nn.deeponet
from learner.utils import mse,grad
import numpy as np
from scipy.interpolate import griddata

class DeepON(ln.nn.Algorithm):
    def __init__(self, net, lam=0):
        super(DeepON, self).__init__()
        self.net = net
        self.lam = lam
        
    def criterion(self, X, y):
        z_2 = X[1].requires_grad_(True)
        u = self.net(X)
        MSEd = mse(u.view(-1),y.view(-1))
        # MSEb = mse((0.0273*(u_x2 + u_y2)).view(-1),torch.zeros_like(u_x2 + u_y2).view(-1))
        return MSEd
    
    def predict(self, x, returnnp=False):
        return self.net.predict(x, returnnp)
def load():
    with open("outputs\Testing1\model_best.pkl","rb") as f:
        model = torch.load(f,map_location='cpu')
    return model

def inter_sample(data):
  #Surround the data with a boundary non-nan value, making it 103*103
  data = np.pad(data, 1, mode='constant', constant_values=1)
  # Replace 0 values with NaN to mark them as missing
  data_nan = data.copy()  # Create a copy to avoid modifying the original array
  data_nan[data_nan == 0] = np.nan

  # Create a grid of coordinates
  x, y = np.mgrid[0:103, 0:103]

  # Get the valid data points (non-NaN)
  valid_mask = ~np.isnan(data_nan)
  valid_points = np.array((x[valid_mask], y[valid_mask])).T
  valid_values = data_nan[valid_mask]

  # Interpolate the missing values using griddata
  interpolated_data = griddata(valid_points, valid_values, (x, y), method='cubic', fill_value=np.nan)
  interpolated_data[np.isnan(interpolated_data)] = griddata(valid_points, valid_values, (x[np.isnan(interpolated_data)], y[np.isnan(interpolated_data)]), method='linear')
  interpolated_data = interpolated_data[1:-1,1:-1]
  return interpolated_data

def seq2mat(input):
  '''
  this is a function with input of an array of data points of shape (101*101,8). The eight columns are: "X,Y" coordinates,and five sensor values.
  in fact, all rows of data coexist on a 101x101 grid, with X ranging from -10 to 10 (increment by 0.2) and Y ranging from -10 to 10 (increment by 0.2), and Z does not change.
  This means that we should output a matrix with shape (101,101,5), according to the first two columns of the sequence. Let's do this.
  '''
  output = np.zeros((101,101,5))
  for i in range(101):
    for j in range(101):
      output[i,j,0] = input[i*101+j,0+3]
      output[i,j,1] = input[i*101+j,1+3]
      output[i,j,2] = input[i*101+j,2+3]
      output[i,j,3] = input[i*101+j,3+3]
      output[i,j,4] = input[i*101+j,4+3]
  return output

def seq2mat2d(input):
  '''
  this is a function with input of an array of data points of shape (101*101,8). The eight columns are: "X,Y" coordinates,and five sensor values.
  in fact, all rows of data coexist on a 101x101 grid, with X ranging from -10 to 10 (increment by 0.2) and Y ranging from -10 to 10 (increment by 0.2), and Z does not change.
  This means that we should output a matrix with shape (101,101,5), according to the first two columns of the sequence. Let's do this.
  '''
  output = np.zeros((101,101))
  for i in range(101):
    for j in range(101):
      output[i,j] = input[i*101+j]
  return output

def mat2seq(input,z):
  '''
  This is the function that does the opposite thing. For each point in the (101,101,5) plane, we should translate them into a sequence of (101*101,8).
  The eight columns are: "X,Y" coordinates,and five sensor values(directly from the input).
  X range from -10 to 10 (increment by 0.2) and Y range from -10 to 10 (increment by 0.2), and Z does not change.
  Let's do this.
  '''
  output = np.zeros((101*101,8))
  for i in range(101):
    for j in range(101):
      output[i*101+j,0] = -10+i*0.2
      output[i*101+j,1] = -10+j*0.2
      output[i*101+j,2] = z
      output[i*101+j,3] = input[i,j,0]
      output[i*101+j,4] = input[i,j,1]
      output[i*101+j,5] = input[i,j,2]
      output[i*101+j,6] = input[i,j,3]
      output[i*101+j,7] = input[i,j,4]
  return output

def sensor(data):
        Indexes = []
        for i in range(-10, 11, 1):
            for j in range(-10, 11, 1):
                Indexes.append((i,j))
        extracted_points = []
        for x, y in Indexes:
            target_xy = np.array([x,y])
            points = data[np.all(data[:,:2] == target_xy, axis=1)]
            if not points.empty:
                extracted_points.append(points)
        result = np.vstack(extracted_points)
        return result

def pred(model,inputs,zi):
    X = [[],[]]
    preds = []
    x,y,z,t = np.zeros(101*101),np.zeros(101*101),np.zeros(101*101),np.zeros(101*101)
    for i in range(101):
        for j in range(101):
            x[i*101+j] = -10+i*0.2
            y[i*101+j] = -10+j*0.2
            z[i*101+j] = zi
    sensors = sensor(mat2seq(inputs,zi))
    X[1] = np.concatenate((x,y,z,t),axis=1)
    X[0] = np.tile(np.tile(sensors,(10201,1)))
    g = X[1].requires_grad_(True)
    preds = model(X)
    return preds,g
# def encode(model,inputs):
#     '''
#     This function takes in a data from MATLAB, interpolates and pushes into the model.
#     It then uses seq2mat to restore the matrix format to output the current predicted field.
#     '''
#     data = inter_sample(inputs)

#     return
# def decode(model,outputs):
#     '''
#     This function takes in a sequence of predicted points and outputs a matrix of the temperature gradients.
#     '''
#     return

def run(z,mat):
    m = load()
    complete_mat = inter_sample(mat)
    preds,g = pred(m,complete_mat,z)
    u = grad(preds,g)
    u_Tx,u_Ty,u_Tz = u[:,0,0],u[:,0,1],u[:,0,2]
    u_T = np.sqrt(u_Tx**2+u_Ty**2+u_Tz**2)
    predsmat = seq2mat(preds)
    predsmatg = seq2mat2d(u_T)
    return predsmat, predsmatg
            
            
