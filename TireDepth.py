import numpy as np
import prettytable as pt
import matplotlib.pyplot as plt
import matplotlib.patches as ps
from matplotlib.patches import CirclePolygon

Tire_Depth=[[["Front Driver Side Tire Depth", "Front Passenger Side Tire Depth"],["Classification"]],
            [[7.5, 7.4], [+1]],
            [[6.5, 6.7], [+1]],
            [[1.3, 1.1], [-1]],
            [[3.2, 2.8], [-1]],
            [[4.6, 4.8], [+1]],
            [[2.4, 2.6], [-1]],
            [[5.4, 5.3], [+1]],
            [[8.0, 7.9], [+1]],
            [[3.6, 3.8], [-1]],
            [[4.0, 3.9], [-1]],
            [[7.0, 6.8], [+1]],
            [[5.0, 4.8], [+1]],
            [[4.3, 4.2], [-1]],
            [[1.7, 1.8], [-1]],
            [[5.8, 5.9], [+1]],
            [[2.2, 2.3], [-1]],
            [[7.2, 7.4], [+1]],
            [[3.3, 3.4], [-1]]]

Numb_of_Iterations = 50
Epsilon = .001
c = 1
Min_Opt_Alpha = .00001


class SVM:
    def __init__(self, x, y):
        self._x = x
        self._y = y
        self._b = np.mat([[0]])
        self._alpha = np.mat(np.zeros((np.shape(x)[0], 1)))

        i=0
        while (i < Numb_of_Iterations):
            if (self.smo() == 0):
                i += 1          
            else:
                i = 0                
        
        self._w = self.calc_w(self._alpha, self._x, self._y)
            
        
    def smo(self):
        alphaPairsOptimized = 0
        
        for i in range(np.shape(self._x)[0]):
            Ei = np.multiply(self._y, self._alpha).T * self._x * self._x[i].T +self._b - self._y[i]   
            
            if (self.check_alpha_kkt(self._alpha[i], Ei)):
                j = self.second_alpha(i, np.shape(self._x)[0])
                Ej = np.multiply(self._y, self._alpha).T * self._x * self._x[j].T + self._b - self._y[j]
                alphaIO = self._alpha[i].copy()
                alphaJO = self._alpha[j].copy()
                boundary = self.alpha_boundaries(self._alpha[i], self._alpha[j], self._y[i], self._y[j])
                ETA = 2 * self._x[i] * self._x[j].T - self._x[i] * self._x[i].T - self._x[j] * self._x[j].T
                
                if boundary[0] != boundary[1] and ETA < 0:
                    if self.opt_alphas(i, j, Ei, Ej, ETA, boundary, alphaIO, alphaJO):
                        alphaPairsOptimized += 1                                           
              
        return alphaPairsOptimized
    
    
    def opt_alphas(self, i, j, Ei, Ej, ETA, boundary, alphaIO, alphaJO):
        flag = False
        self._alpha[j] -= self._y[j] * (Ei - Ej) / ETA
        self.clip_alphaj(j, boundary)
        
        if (abs(self._alpha[j] - alphaJO) >= Min_Opt_Alpha):
            self.optimize_alphai_with_alphaj(i, j, alphaJO)
            self.opt_b(Ei, Ej, alphaIO, alphaJO, i, j)
            flag = True
            
        return flag
    
    
    def opt_b(self, Ei, Ej, alphaIO, alphaJO, i, j):
        b1 = self._b - Ei - self._y[i] * (self._alpha[i] - alphaIO) * self._x[i] * self._x[i].T - self._y[j] * (self._alpha[j] - alphaJO) * self._x[i] * self._x[j].T
        b2 = self._b - Ej - self._y[i] * (self._alpha[i] - alphaIO) * self._x[i] * self._x[j].T - self._y[j] * (self._alpha[j] - alphaJO) * self._x[j] * self._x[j].T
        
        if (0 < self._alpha[i]) and (c > self._alpha[i]):
            self._b = b1
        elif (0 < self._alpha[j]) and (c > self._alpha[j]):
            self._b = b2
        else:
            self._b = (b1+b2) * .5
    
    
    def second_alpha(self, iFirstAlpha, rows):
        iSecondAlpha = iFirstAlpha
        while (iFirstAlpha == iSecondAlpha):
            iSecondAlpha = int(np.random.uniform(0, rows))
            
        return iSecondAlpha
        
    
    def optimize_alphai_with_alphaj(self, i, j, alphaJO):
        self._alpha[i] += self._y[j] * self._y[i] * (alphaJO - self._alpha[j])
              
    
    def clip_alphaj(self, j, boundary):
        if self._alpha[j] < boundary[0]: 
            self._alpha[j] = boundary[0]
        if self._alpha[j] > boundary[1]:
            self._alpha[j] = boundary[1]
        
        
    def check_alpha_kkt(self, alpha, E):
        return (alpha > 0 and np.abs(E) < Epsilon) or (alpha < c and np.abs(E) > Epsilon)

    
    def alpha_boundaries(self, alphai, alphaj, yi, yj):
        boundary = [2]
        if (yi == yj):
            boundary.insert(0, max(0, alphaj + alphai - c))
            boundary.insert(1, min(c, alphaj + alphai))
        else:
            boundary.insert(0, max(0, alphaj - alphai))
            boundary.insert(1, min(c, alphaj - alphai + c))
            
        return boundary
    
    
    def calc_w(self, alpha, x, y):
        w = np.zeros((np.shape(x)[1], 1))
        
        for i in range(np.shape(x)[0]):
            m=y[i]*alpha[i]
            w += np.multiply(m, x[i].T)  
            
        return w
           
            
    def classify(self, x):
        classification = "Driver should consider tire rotation or change"
        
        if (np.sign((x @ self._w +self._b).item(0,0)) == 1):
            classification = "Tires are safe for continued use"
            
        return classification    
    
    
    def get_w(self):
        return self._w
    
    
    def get_alpha(self):
        return self._alpha
    
    
    def get_b(self):
        return self._b


def cmd_line():
    flag = True
    
    while (flag):
        entry = input("Enter Front Tire Depth (mm) or Type (complete) to exit:\n")        
        if (entry != "complete"):
            depth = entry.split()
            print(svm.classify(np.mat([float(depth[0]), float(depth[1])])))
        else:
            print("\n>Thank you for trying out my tire depth gauge!<\n")
            flag = False
            
            
def plot(w, alpha, b):
    x0=[]; y0=[]; x1=[]; y1=[]
    
    for i in range(1, len(Tire_Depth)):
        if (Tire_Depth[i][1][0] == 1):
            x0.append(Tire_Depth[i][0][0])
            y0.append(Tire_Depth[i][0][1])
        else:
            x1.append(Tire_Depth[i][0][0])
            y1.append(Tire_Depth[i][0][1])
            
    plot = plt.figure()
    
    plt.xlabel(Tire_Depth[0][0][0])
    plt.ylabel(Tire_Depth[0][0][1])
    ax = plot.add_subplot(1,1,1)
    
    ax.scatter(x0, y0, marker = 'o', s = 25, c = 'red')
    ax.scatter(x1, y1, marker = 'o', s = 25, c = 'blue')
    
    for i in range(len(xR)):
        if alpha[i] != c and alpha[i] > 0:
            ax.add_patch(ps.CirclePolygon((xR[i][0], xR[i][1]), .2, facecolor='none', edgecolor=(0,0,0), linewidth=1, alpha=.9))
            
    x = np.arange(0, 10, .1)
    y = (-w[0] * x - b) / w[1]
    ax.axis([0, 10, 0, 10])
    ax.plot(x,y)   
    plt.show()
    

def table():
    svmt = pt.PrettyTable(['Support Vector', 'Label', 'Alpha'])
    
    for i in range(len(xR)):
        if svm.get_alpha()[i] != c and svm.get_alpha()[i] > 0:
            svmt.add_row([xR[i], yR[i], svm.get_alpha()[i].item(0,0)])
            
    print(svmt)
    wbt = pt.PrettyTable(['wT', 'b'])
    wbt.add_row([svm.get_w().T, svm.get_b()])
    print(wbt)
        
xR = []
yR = []

for i in range(1, len(Tire_Depth)):
    xR.append(Tire_Depth[i][0])
    yR.append(Tire_Depth[i][1][0])

svm = SVM(np.mat(xR), np.mat(yR).transpose())

table()
print("This Window Shows Examples Of Tire Depths That Are Good And Bad\n")
print("Close Chart To Check Tire Depths")
plot(svm.get_w(), svm.get_alpha(), svm.get_b().item(0,0))
cmd_line()
            
