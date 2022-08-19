import matplotlib.pyplot as plt
import numpy as np

def save(dir_name, x_opt, f_opt, iter, parameters):
    print("save")
    with open('./ONLS_Trb/'+dir_name+'/results.txt', 'a') as arquivo:
        arquivo.write("Parameters\n")
        arquivo.write(" ".join(parameters))
        arquivo.write("\nNumber of iterations\n")
        arquivo.write(str(iter))
        arquivo.write("\nPonto otimo:\n")
        arquivo.write("x: "+str(x_opt) + "- f(x): "+str(f_opt))
    
    arquivo.close()
        

def plot_graphic(v, iteration, dir_name):
    
    fig, ax = plt.subplots()
    ax.contour(X, Y, Z,  1000, colors = 'k')
            
    ax.plot(v[:,0], v[:,1], color='green')
    ax.scatter(v[:,0], v[:,1], facecolors='none', edgecolors='g')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    plt.savefig("./ONLS_Trb/"+dir_name+"/image"+str(iteration)+".png")
    plt.show()   
    

def f1(x1, x2):
    return np.exp(x1 + 3*x2 - 0.1) + np.exp(x1 - 3*x2 - 0.1) + np.exp(-x1 - 0.1)


x = np.linspace(-3, 3, 1000)
y = np.linspace(-2, 2, 1000)

X, Y = np.meshgrid(x, y)
Z = f1(X, Y)