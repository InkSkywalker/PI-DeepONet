"""
@author: jpzxshi
"""
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
        self.__init_data()
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
        return result_df["T"]
    def vrepeat(self, array, times):
        narray = array
        for i in range(times-1):
            narray = np.vstack((narray,array))
        return narray
    def generate(self, dfs):
       X_b,X_t,y = [],[],[]
       for df in dfs:
            X_1,X_2,y_f = [],[],[]
            sdf = np.array(self.sensor(df))
            X_1 = self.vrepeat(sdf,10201).reshape((-1,441))   
            X_2 = np.array(df[["X","Y","Z"]]).reshape((-1,3))
            y_f = np.array(df["T"])
            X_b.append(X_1)
            X_t.append(X_2)
            y.append(y_f)
       X_b = np.array(X_b)
       X_t = np.array(X_t)
       y = np.array(y)
       return X_b,X_t,y
    
    def __init_data(self,dfs,rate):
        self.X_train, self.y_train = self.generate(dfs,rate)
        
class DeepON(ln.nn.Algorithm):
    def __init__(self, net, lam=1):
        super(DeepON, self).__init__()
        self.net = net
        self.lam = lam
        
    def criterion(self, X, y):
        z = X['diff'].requires_grad_(True)
        u = self.net(z)
        
        MSEd = mse()
        MSEb = (mse(self.net(X['x_0']), y['x_0']) + mse(self.net(X['x_1']), y['x_1']) + 
                mse(self.net(X['0_y']), y['0_y']) + mse(self.net(X['1_y']), y['1_y']))
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
    device = 'cpu' # 'cpu' or 'gpu'
    
    branch_size = [441 , 200, 100, 50, 25, 1]
    trunk_size = [4,2,1]
    activation = 'relu'
    # training
    lr = 0.001
    iterations = 10000
    print_every = 1000
    batch_size = 10201
    
    data = TempData()
    deeponet = ln.nn.deeponet(branch_size = branch_size, trunk_size = trunk_size, activation = activation)
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