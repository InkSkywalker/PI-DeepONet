"""
@author: jpzxshi
"""
import torch
import numpy as np
import learner as ln
import pandas as pd
from learner.utils import mse, grad
from sklearn.model_selection import train_test_split

class TempData(ln.Data):
    def __init__(self, dfs):
        super(TempData, self).__init__()
        self.train_num = len(dfs)*8160
        self.test_num = len(dfs)*2041
        self.__init_data(dfs)
        
    def sensor(self,df):
        Indexes = []
        for i in range(-10, 11, 1):  
            for j in range(-10, 11, 1):  
                Indexes.append((i,j))
        extracted_points = []
        for x, y in Indexes:
            mask = (df['X'] == x) & (df['Y'] == y) 
            points = df[mask]
            if not points.empty:  
                extracted_points.append(points)
        result_df = pd.concat(extracted_points)
        return result_df[["T"]]
    def vrepeat(self, array, times):
        narray = array
        for i in range(times-1):
            narray = np.vstack((narray,array))
        return narray
    def generate(self, dfs):
        X_train,X_val = [[],[]] , [[],[]]
        y_train,y_val = [],[]
        for df in dfs:
            X_1,X_2,y_f = [],[],[]
            sdf = np.array(self.sensor(df))
            print(f"Sensor shape: {sdf.shape}")
            X_1 = self.vrepeat(sdf,10201).reshape((-1,441))   
            X_2 = np.array(df[["X","Y","Z"]]).reshape((-1,3))
            y_1 = np.array(df["T"])
            X_1_t,X_1_v = train_test_split(X_1,test_size=0.2)
            X_2_t,X_2_v = train_test_split(X_2,test_size=0.2)
            y_1_t,y_1_v = train_test_split(y_1,test_size=0.2)
            X_train[0].append(X_1_t)
            X_train[1].append(X_2_t)
            X_val[0].append(X_1_v)
            X_val[1].append(X_2_v)
            y_train.append(y_1_t)
            y_val.append(y_1_v)
        X_train[0] = np.array(X_train[0]).reshape((-1,441))
        X_train[1] = np.array(X_train[1]).reshape((-1,3))
        X_val[0] = np.array(X_val[0]).reshape((-1,441))
        X_val[1] = np.array(X_val[1]).reshape((-1,3))
        y_train = np.array(y_train).reshape((-1))
        y_val = np.array(y_val).reshape((-1))
        return X_train, y_train, X_val,y_val
    
    def __init_data(self,dfs):
        self.X_train, self.y_train,self.X_test,self.y_test = self.generate(dfs)
        
class DeepON(ln.nn.Algorithm):
    def __init__(self, net, lam=0):
        super(DeepON, self).__init__()
        self.net = net
        self.lam = lam
        
    def criterion(self, X, y):
        z_2 = X[1].requires_grad_(True)
        # print(f"z_2 shape:{z_2.shape}")
        u = self.net(X)
        u_g = grad(u, z_2)
        # print(f"u_g shape:{u_g.shape}")
        u_x, u_y = u_g[:, 0], u_g[:, 1]
        u_x2, u_y2 = grad(u_x, z_2)[:, 0], grad(u_y, z_2)[:, 1]
        MSEd = mse(u.view(-1),y.view(-1))
        MSEb = mse((0.0273*(u_x2 + u_y2)).view(-1),torch.zeros_like(u_x2 + u_y2).view(-1))
        return MSEd + self.lam * MSEb
    
    def predict(self, x, returnnp=False):
        return self.net.predict(x, returnnp)

# def plot(data, net):
#     import matplotlib.pyplot as plt
#     import itertools
#     X = np.array(list(itertools.product(np.linspace(0, 1, num=100), np.linspace(0, 1, num=100))))
#     solution_true = np.rot90((np.sin(X[:, 0]) * np.sin(X[:, 1])).reshape(100, 100))
#     solution_pred = np.rot90(net.predict(X, returnnp=True).reshape(100, 100))
#     L2_error = np.sqrt(np.mean((solution_pred - solution_true) ** 2))
#     print('L2_error:', L2_error)
    
#     plt.figure(figsize=[6.4 * 2, 4.8])
#     plt.subplot(121)
#     plt.imshow(solution_true, cmap='rainbow')
#     plt.title(r'Exact solution $\sin(x)\sin(y)$', fontsize=18)
#     plt.xticks([0, 49.5, 99], [0, 0.5, 1])
#     plt.yticks([0, 49.5, 99], [1, 0.5, 0])
#     plt.subplot(122)
#     plt.imshow(solution_pred, cmap='rainbow')
#     plt.title('Prediction', fontsize=18)
#     plt.xticks([0, 49.5, 99], [0, 0.5, 1])
#     plt.yticks([0, 49.5, 99], [1, 0.5, 0])
#     plt.savefig('pinn.pdf')

def main():
    device = 'gpu' # 'cpu' or 'gpu'
    branch_size = [441,200,100,50,25,1]
    trunk_size = [3,6,9,6,3,1]
    activation = 'relu'
    #data
    df_10_16 = pd.read_csv("10_16.csv")
    dfs = [df_10_16]
    # training
    lr = 0.0001
    iterations = 500000
    print_every = 1000
    batch_size = 80
    
    data = TempData(dfs)
    deeponet = ln.nn.DeepONet(branch_size = branch_size, trunk_size = trunk_size, activation = activation)
    net = DeepON(deeponet)
    args = {
        'data': data,
        'net': net,
        'criterion': None,
        'optimizer': 'adam',
        'lr': lr,
        'iterations': iterations,
        'batch_size': batch_size,
        'print_every': print_every,
        'save': True,
        'callback': None,
        'dtype': 'float',
        'device': device,
    }
    
    ln.Brain.Init(**args)
    ln.Brain.Run()
    ln.Brain.Restore()
    ln.Brain.Output()
    
    # plot(data, ln.Brain.Best_model())
    
if __name__ == '__main__':
    main()