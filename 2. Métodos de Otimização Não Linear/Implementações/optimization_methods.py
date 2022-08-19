import numpy as np
from random import random, uniform
from sympy import *
from sympy.tensor.array import derive_by_array
from scipy.linalg import cholesky
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
import pylab
import scipy.optimize as sopt
from scipy import random, linalg
from level_curves import *


class Newton():
    
    def __init__(self, f, gf, Hf, epsilon1, epsilon2, max_iterations, initial = []):
        self.f = f
        self.g = gf
        self.H = Hf
        self.epsilon1 = epsilon1
        self.epsilon2 = epsilon2
        self.max_iterations = max_iterations
        self.x = initial if len(initial) else self.initialize()
        self.parameters = [str(epsilon1), str(epsilon2), str(max_iterations), str(self.x)]
        
        
    def initialize(self):
        x1 = uniform(-3, 3)
        x2 = uniform(-2, 2)
        return np.array([x1,x2])
    
    def run(self):
        
        stopping_criterion1 = True
        stopping_criterion2 = True
        stopping_criterion3 = True
        
        alpha, beta = 0.2, 0.95
        
        X_ = [self.x]
        p_opt = np.min(Z) #Optimum point
        points = []
        diffs = []
        
        i = 0
        while(stopping_criterion1 and stopping_criterion2 and stopping_criterion3):
            x = self.x
            
            points.append(self.x)
            v = np.array(points)
            
            plot_graphic(v, i, "Newton")
            
            H = self.H(x)
            print("Hessian ", H)
            
            H = H.astype('float')
            inv_H = np.linalg.inv(H)
            
            print("Hessian Inverse ", inv_H)
            
            self.gr = self.g(x)
            print("Gradiente de x ")
            print(self.gr)
            
            #self.delta = -np.linalg.solve(H, self.gr)
            self.delta = -np.dot(inv_H, self.gr.T) 
            print("delta: ", self.delta)
            
            s_lambda = np.dot(self.gr, -self.delta)
            print("s_lambda: ", s_lambda)
            
            self.backtracking(alpha, beta)
            self.x = self.x + self.t*self.delta
            
            X_.append(self.x)
            
            print("x->f(x)", x, "->", self.f(x))
        
            difference = self.f(X_[i]) - p_opt
            diffs.append(difference)
    
            
            stopping_criterion1 = s_lambda/2.0 > self.epsilon1
            stopping_criterion2 = abs(self.f(X_[i+1]) - self.f(X_[i])) > self.epsilon2
            stopping_criterion3 = i < self.max_iterations
            
            i = i + 1
        
        points.append(self.x)
        v = np.array(points)   
        plot_graphic(v, i, "Newton")
        
        print("Number of iterations: ", i)
        print("Point of convergence: x -> f(x) ", self.x, "->", self.f(x))
        
        save("Newton", self.x, self.f(x), i, self.parameters)
        
        plt.clf()
        plt.plot([y for y in range(len(diffs))],np.array(diffs))
        plt.xlabel('Iterations')
        plt.ylabel('f(x) - p* (error)')
        plt.savefig('./ONLS_Trb/Newton/error_curve.png')
        plt.show()
             
        return self.x
        
    def backtracking(self, alpha, beta):
        self.t = 1
        while((self.f(self.x + self.t*self.delta)) >= (self.f(self.x) + alpha*self.t*np.dot(self.gr, self.delta))):
            self.t *= beta

class GradientDescent():
    
    def __init__(self, f, gf, epsilon1, epsilon2, max_iterations, initial = []):
        self.f = f
        self.g = gf
        self.epsilon1 = epsilon1
        self.epsilon2 = epsilon2
        self.max_iterations = max_iterations
        self.x = initial if len(initial) else self.initialize()
        self.parameters = [str(epsilon1), str(epsilon2), str(max_iterations), str(self.x)]
        
        
    def initialize(self):
        x1 = uniform(-3, 3)
        x2 = uniform(-2, 2)
        return np.array([x1, x2])

    def run(self):
        
        stopping_criterion1 = True
        stopping_criterion2 = True
        stopping_criterion3 = True
        
        alpha, beta = 0.2, 0.95
    
        
        X_ = [self.x]
        p_opt = np.min(Z) #Optimum point
        points = []
        diffs = []
        
        i = 0
        
        while(stopping_criterion1 and stopping_criterion2 and stopping_criterion3):
            
            x = self.x
            
            points.append(self.x)
            v = np.array(points)
            
            plot_graphic(v, i, "Gradient_Descent")
            
            self.gr = self.g(x)
            
            print("Gradiente de x")
            print(self.gr)
            
            self.delta = -self.gr
            print("delta")
            print(self.delta)
            
            self.backtracking(alpha, beta)
            self.x = self.x + self.t*self.delta
            X_.append(self.x)
            
            norm_l2_gr = np.linalg.norm(self.gr, ord=2)
            
            print("x - f(x)")
            print(self.x, self.f(x))
            
            difference = self.f(X_[i+1]) - self.f(X_[i])
            diffs.append(difference)
            
            
            
            stopping_criterion1 = abs(norm_l2_gr) >= self.epsilon1
            #stopping_criterion2 = (self.f(X_[i+1]) - self.f(X_[i])) >= self.epsilon2
            stopping_criterion2 = difference >= self.epsilon2
            stopping_criterion3 = i >= self.max_iterations
            
            i = i + 1
        
        points.append(self.x)
        v = np.array(points)   
        plot_graphic(v, i, "Gradient_Descent")
        
        print("Number of iterations: ", i)
        print("Point of convergence: x -> f(x) ", self.x, "->", self.f(x))
        
        save("Gradient_Descent", self.x, self.f(x), i, self.parameters)
        
        plt.clf()
        plt.plot([y for y in range(len(diffs))],np.array(diffs))
        plt.xlabel('Iterations')
        plt.ylabel('f(x) - p* (error)')
        plt.savefig('./ONLS_Trb/Gradient_Descent/error_curve2.png')
        plt.show()
        return self.x, self.f(x)
    
    def backtracking(self, alpha, beta):
        self.t = 1
        while((self.f(self.x + self.t*self.delta)) >= (self.f(self.x) + alpha*self.t*np.dot(self.gr, self.delta))):
            self.t *= beta


    
class SteepestDescent():
    
    def __init__(self, f, gf, eps1, eps2, max_iter, initial = []):
        self.f = f
        self.g = gf
        self.eps1 = eps1
        self.eps2 = eps2
        self.max_iter = max_iter
        self.x = initial if len(initial) else self.initialize()
        self.parameters = [str(eps1), str(eps2), str(max_iter), str(self.x)]
        
        
    def initialize(self):
        x1 = uniform(-3, 3)
        x2 = uniform(-2, 2)
        return np.array([x1, x2])

    def run(self):
        
        stopping_criterion1 = True
        stopping_criterion2 = True
        stopping_criterion3 = True
        
        alpha, beta = 0.2, 0.95
        
        X_ = [self.x]
        points = []
        p_opt = np.min(Z) #Optimum point
        diffs = []
        i = 0
        
        matrixSize = 2 
        A = random.rand(matrixSize,matrixSize)
        P = np.dot(A,A.transpose())
        
        print("Matrix P:")
        print(P)
                
        while(stopping_criterion1 and stopping_criterion2 and stopping_criterion3):
            
            x = self.x
            
            #Add point x in points
            points.append(self.x)
            v = np.array(points)
            
            #Show the points' set
            print(v)
            
            #Show the graphic
            plot_graphic(v, i, "Steepest_Descent")
            
            self.gr = self.g(x)
            
            #Print Gradient of x
            print("Gradiente de x")
            print(self.gr)
            
            #Calculate the delta
            self.delta = - np.dot(np.linalg.inv(P), self.gr)
            self.s = self.delta
            
            #Show the delta
            print("delta")
            print(self.delta)
            
            #Update delta end update the x value
            self.backtracking(alpha, beta)
            self.x = self.x + self.t*self.delta
            X_.append(self.x)
        
            #Calculate the L2 norm of gradient
            norm_l2_gr = np.linalg.norm(self.gr, ord=2)
            
            #Show x and f(x) values
            print("x - f(x)")
            print(self.x, self.f(x))
            
            #Calculate the errors
            difference = self.f(X_[i]) - p_opt
            diffs.append(difference)
            
            #Criterion of stopped
            stopping_criterion1 = abs(norm_l2_gr) > self.eps1
            stopping_criterion2 = abs(self.f(X_[i+1]) - self.f(X_[i])) >= self.eps2 
            stopping_criterion3 = i < self.max_iter
            print(i)
            i += 1
        
        points.append(self.x)
        v = np.array(points)   
        plot_graphic(v, i, "Steepest_Descent")
        
        print("Number of iterations: ", i)
        print("Point of convergence: x -> f(x) ", self.x, "->", self.f(x))
        save("Steepest_Descent", self.x, self.f(x), i, self.parameters)
        
        plt.clf()
        plt.plot([y for y in range(len(diffs))],np.array(diffs))
        plt.xlabel('Iterations')
        plt.ylabel('f(x) - p* (error)')
        plt.savefig('./ONLS_Trb/Steepest_Descent/error_curve2.png')
        plt.show()
        return self.x, self.f(x)
        
        
    def backtracking(self, alpha, beta):
        self.t = 1
        while((self.f(self.x + self.t*self.delta)) >= (self.f(self.x) + alpha*self.t*np.dot(self.gr, self.delta))):
            self.t *= beta
            
    '''def f1d(self, alpha):
        return self.f(self.x + alpha*self.s)

    def updateAlpha(self):
        alpha_opt = sopt.golden(self.f1d)
        return alpha_opt'''


class LevenbergMarquardt:
    
    def __init__(self, f, gf, Hf, alpha, eps, max_iterations, initial = [], c = []):
        const = 10**(-5)
        self.f = f
        self.g = gf
        self.H = Hf
        self.x = initial if len(initial) else self.initialize()
        self.alpha = alpha
        self.c1, self.c2 = c if len(c) else [random() + const, 1 + random() + const]
        self.eps = eps 
        self.max_iterations = max_iterations
        self.parameters = [str(alpha), str(eps), str(max_iterations), str(self.x), str(self.c1) +"-"+ str(self.c2)]
        
        
                
    def initialize(self):
        x1 = uniform(-3,3)
        x2 = uniform(-2,2)

        return np.array([x1,x2])
        
    def run(self):
        '''x_opt1, x_opt2 = sopt.golden(f1)
        print("x_opt " + str(x_opt1) + " "+ str(x_opt2))
        p_opt = f1(x_opt1, x_opt2)
        print(p_opt)'''
        i = 0
        print("x_0: ", self.x) 
        X_ = [self.x]
        points = []
        
        error = 1
        p_opt = np.min(Z)
        diffs = []
        
        alpha, beta = 0.2, 0.95 
        #0.95
        
        stopping_criterion1 = True
        stopping_criterion2 = True
        
        while(stopping_criterion1 and stopping_criterion2):
            
            x = self.x
            
            #Add point x in points
            points.append(self.x)
            v = np.array(points)
            
            #Show the points' set
            print(v)
            
            #Show the graphic
            plot_graphic(v, i, "Levenberg_Marquardt")
            
            self.gr = self.g(x)
            
            #Print Gradient of x
            print("Gradiente de x")
            print(self.gr)
            
            #Calculate Hessian
            H = self.H(x)
            #print H
            H = H.astype('float')
            print("shape of H ", H.shape)
        
            identity = np.eye(H.shape[0], H.shape[1])
            print("alpha: ", self.alpha)
            print("identity: ", identity)
            m = H + self.alpha*identity
            inv_matrix = np.linalg.inv(m)
            
            self.delta = -np.dot((inv_matrix), self.gr)
            self.backtracking(alpha, beta)
            self.x = self.x + self.t*self.delta
        
            X_.append(self.x)
            
            print("X ", (X_[i+1]))
            print("Function ", self.f((X_[i+1]))) 
            error = np.abs(self.f(X_[i+1]) - self.f(X_[i]))  
            print("Error ", error)
            print("Eps ", self.eps)
            stopping_criterion1 = error > self.eps
            stopping_criterion2 = i < self.max_iterations
            
                        
            difference = self.f(X_[i]) - p_opt
            diffs.append(difference)
            if (self.f(X_[i+1]) < self.f(X_[i])):
                self.alpha = self.c1*self.alpha
            
            else:
                self.alpha = self.c2*self.alpha

            
            i = i + 1
            print("iteration ", i)
            
        points.append(self.x)
        v = np.array(points)   
        plot_graphic(v, i, "Levenberg_Marquardt")
        
        print("Number of iterations: ", i)
        print("Point of convergence: x -> f(x) ", self.x, "->", self.f(x))
        save("Levenberg_Marquardt", self.x, self.f(x), i, self.parameters)
        
        plt.clf()
        plt.plot([x for x in range(len(diffs))],np.array(diffs))
        plt.xlabel('Iterations')
        plt.ylabel('f(x) - p* (error)')
        plt.savefig('./ONLS_Trb/Levenberg_Marquardt/error_curve.png')
        plt.show()
        plt.clf()
        plt.plot([x for x in range(len(diffs))],np.array(diffs))
        plt.show()    


    def backtracking(self, alpha, beta):
        self.t = 1
        while((self.f(self.x + self.t*self.delta)) >= (self.f(self.x) + alpha*self.t*np.dot(self.gr, self.delta))):
            self.t *= beta


'''  
def function(x):
    global x1, x2
    x_str = ['x'+str(i) for i in range(len(x))]
    x_str = ' '.join(x_str)
    x1, x2 = symbols(x_str)
    return exp(x1 + 3*x2 - 0.1) + exp(x1 - 3*x2 - 0.1) + exp(-x1 - 0.1)
    #x1, x2 = x[0], x[1]
    #return x1**2 - x2
'''
##### Analiticaly function ##### 
def f(x):
    return np.exp(x[0] + 3*x[1] - 0.1) + np.exp(x[0] - 3*x[1] - 0.1) + np.exp(-x[0] - 0.1)

def gf(x):
    return np.array([np.exp(x[0] + 3*x[1] - 0.1) + np.exp(x[0] - 3*x[1] - 0.1) - np.exp(-x[0] - 0.1), 3 * np.exp(x[0] + 3*x[1] - 0.1) - 3 * np.exp(x[0] - 3*x[1] - 0.1) ]) 

def Hf(x):
    return np.array([[np.exp(x[0] + 3*x[1] - 0.1) + np.exp(x[0] - 3*x[1] - 0.1) + np.exp(-x[0] - 0.1), 3 * np.exp(x[0] + 3*x[1] - 0.1) - 3 * np.exp(x[0] - 3*x[1] - 0.1)],[3 * np.exp(x[0] + 3*x[1] - 0.1) - 3 * np.exp(x[0] - 3*x[1] - 0.1), 6 * np.exp(x[0] + 3*x[1] - 0.1) + 6 * np.exp(x[0] - 3*x[1] - 0.1)]])
################################
'''
def f(x):
    return float(subs_(function(x), x))
    
def gf(x):
    x_str = ['x'+str(i) for i in range(len(x))]
    x_str = ' '.join(x_str)
    x1, x2 = symbols(x_str)
    derivatives = list(derive_by_array(function(x), (x1, x2)))
    derivatives = [subs_(derivatives[i], x) for i in range(len(derivatives))]
    derivatives = np.array(derivatives, dtype = np.float64)
    return derivatives

def Hf(x):
    x_str = ['x'+str(i) for i in range(len(x))]
    x_str = ' '.join(x_str)
    x1, x2 = symbols(x_str)
    h = list(derive_by_array(derive_by_array(function(x), (x1, x2)), (x1, x2)))
    h = [subs_(h[i], x) for i in range(len(h))]
    h = np.array(h,dtype = np.float64)
    h = h.reshape((len(h)/2, len(h)/2))
    
    return h


def subs_(f, x):
    global x1, x2
    return N(((f.subs(x1, x[0])).subs(x2, x[1])))

'''          
    
if __name__ == "__main__":
    '''x = [1,1]
    print('f(x)')
    print((f(x)))
    print('gf(x)')
    print(gf(x))
    #h = hf(x)
    print('Hf(x)')
    H = Hf(x)
    H = H.astype('float')
    print(H)
    print('H^(-1)')
    print(np.linalg.inv(H))
    print("Cholesky")
    c = cholesky(H, lower=True)
    print(c)'''
    '''
    
    '''
    #newton = Newton(f, gf, Hf, 10**(-4), 10**(-4), 20)
    #newton.run()
    
    #gradient = GradientDescent(f, gf, 10**(-4), 10**(-4), 20)
    #gradient.run()
    
    steepest = SteepestDescent(f, gf, 10**(-4), 10**(-4), 15)
    steepest.run()
    
    #LM = LevenbergMarquardt(f, gf, Hf, 10**(4), 10**(-5))
    #LM.run()