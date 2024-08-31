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
        return result_df[["T","P","Xspd","Yspd","Zspd"]]
    def vrepeat(self, array, times):
        return np.tile(array, (times, 1))
    def generate(self,dfs):
        X_train,X_val = [[],[]] , [[],[]]
        y_train,y_val = [],[]
        for df in dfs:
            X_1,X_2,y_f = [],[],[]
            sdf = np.array(self.sensor(df))
            print(f"Sensor shape: {sdf.shape}")
            X_1 = self.vrepeat(sdf,10201).reshape((-1,441*5))   
            X_2 = np.array(df[["X","Y","Z"]]).reshape((-1,3))
            X_2 = np.hstack((X_2, np.zeros((X_2.shape[0], 1))))
            y_1 = np.array(df[["T","P","Xspd","Yspd","Zspd"]])
            X_1_t,X_1_v = train_test_split(X_1,test_size=0.2)
            X_2_t,X_2_v = train_test_split(X_2,test_size=0.2)
            y_1_t,y_1_v = train_test_split(y_1,test_size=0.2)
            X_train[0].append(X_1_t)
            X_train[1].append(X_2_t)
            X_val[0].append(X_1_v)
            X_val[1].append(X_2_v)
            y_train.append(y_1_t)
            y_val.append(y_1_v)
        X_train[0] = np.array(X_train[0]).reshape((-1,441*5))
        X_train[1] = np.array(X_train[1]).reshape((-1,4))
        X_val[0] = np.array(X_val[0]).reshape((-1,441*5))
        X_val[1] = np.array(X_val[1]).reshape((-1,4))
        y_train = np.array(y_train).reshape((-1,5))
        y_val = np.array(y_val).reshape((-1,5))
        return X_train, y_train, X_val,y_val
    
    def __init_data(self,dfs):
        self.X_train, self.y_train,self.X_test,self.y_test = self.generate(dfs)
        
class DeepON(ln.nn.Algorithm):
    def __init__(self, net, lam=0):
        super(DeepON, self).__init__()
        self.net = net
        self.lam = lam
        
    # def criterion(self, X, y):
    #     z_2 = X[1].requires_grad_(True)
    #     # print(f"0: {X[0].shape},1:{X[1].shape}")
    #     # print(f"z_2 shape:{z_2.shape}")
    #     u = self.net(X)
    #     # u_g = grad(u, z_2)
    #     # print(f"u_g shape:{u_g.shape}")
    #     # u_x, u_y = u_g[:, 0], u_g[:, 1]
    #     # u_x2, u_y2 = grad(u_x, z_2)[:, 0], grad(u_y, z_2)[:, 1]
    #     # print(f"u_shape:{u.shape},y_shape:{y.shape}")
    #     MSEd = mse(u.view(-1),y.view(-1))
    #     # MSEb = mse((0.0273*(u_x2 + u_y2)).view(-1),torch.zeros_like(u_x2 + u_y2).view(-1))
    #     return MSEd
    def criterion(self, X, yy):
        zz = X[1].requires_grad_(True)
        x,y,z,t = zz[0].requires_grad_(True),zz[1].requires_grad_(True),zz[2].requires_grad_(True),zz[3].requires_grad_(True)
        u = self.net(X)
        T,P,Xspd,Yspd,Zspd = u[:,0],u[:,1],u[:,2],u[:,3],u[:,4]
        g_1 = grad(u,zz)
        u_Tx,u_Ty,u_Tz = g_1[:,0,0].view((-1,1)),g_1[:,0,1].view((-1,1)),g_1[:,0,2].view((-1,1))
        u_Px,u_Py,u_Pz = g_1[:,1,0],g_1[:,1,1],g_1[:,1,2]
        u_Xspdx, u_Xspdy, u_Xspdz  = g_1[:,2,0].view((-1,1)),g_1[:,2,1].view((-1,1)),g_1[:,2,2].view((-1,1))
        u_Yspdx, u_Yspdy, u_Yspdz  = g_1[:,3,0].view((-1,1)),g_1[:,3,1].view((-1,1)),g_1[:,3,2].view((-1,1))
        u_Zspdx, u_Zspdy, u_Zspdz  = g_1[:,4,0].view((-1,1)),g_1[:,4,1].view((-1,1)),g_1[:,4,2].view((-1,1))
        # u_Tx2 ,u_Ty2, u_Tz2 = grad(u_Tx,x), grad(u_Ty,y), grad(u_Tz,z)
        # u_Xspdx2, u_Xspdy2, u_Xspdz2 = grad(u_Xspdx,x), grad(u_Xspdy,y), grad(u_Xspdz,z)
        # u_Yspdx2, u_Yspdy2, u_Yspdz2 = grad(u_Yspdx,x), grad(u_Yspdy,y), grad(u_Yspdz,z)
        # u_Zspdx2, u_Zspdy2, u_Zspdz2 = grad(u_Zspdx,x), grad(u_Zspdy,y), grad(u_Zspdz,z)
        u_Tx2, u_Ty2, u_Tz2 = grad(u_Tx,zz)[:,0],grad(u_Ty,zz)[:,1],grad(u_Tz,zz)[:,2]
        u_Xspdx2, u_Xspdy2, u_Xspdz2 = grad(u_Xspdx,zz)[:,0], grad(u_Xspdy,zz)[:,1], grad(u_Xspdz,zz)[:,2]
        u_Yspdx2, u_Yspdy2, u_Yspdz2 = grad(u_Yspdx,zz)[:,0], grad(u_Yspdy,zz)[:,1], grad(u_Yspdz,zz)[:,2]
        u_Zspdx2, u_Zspdy2, u_Zspdz2 = grad(u_Zspdx,zz)[:,0], grad(u_Zspdy,zz)[:,1], grad(u_Zspdz,zz)[:,2]
        rho = P/(T+273.15)*(0.029/8.31446261815324)
        MSE_ground = mse(u.view(-1),yy.view(-1))
        return MSE_ground
    
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
    device = 'cpu' # 'cpu' or 'gpu'
    branch_size = [2205,1000,441,200,100,50,25,15,5]
    trunk_size = [4,6,12,6,5]
    activation = 'relu'
    #data
    df_10_16 = pd.read_csv("10_16.csv")
    dfs = [df_10_16]
    # training
    lr = 0.0001
    iterations = 1000
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