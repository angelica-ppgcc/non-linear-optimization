from optimization_methods import *

if __name__ == "__main__":
    #gradient = GradientDescent(f, gf, 10**(-3), 10**(-3), 40, initial = [-0.5, 0.5])
    #gradient.run()
    
    steepest = SteepestDescent(f, gf, 10**(-3), 10**(-3), 500, initial = [-0.5, 0.5])
    steepest.run()
    
    #newton = Newton(f, gf, Hf, 10**(-3), 10**(-3), 20, initial = [2.5, 0.5])
    #newton.run()
    
    '''
    alpha = 10**(4)
    eps = 10**(-5)
    
    LM = LevenbergMarquardt(f, gf, Hf, 10**(1), 10**(-3), 20, [-0.5, 0.5], c = [0.5, 1.5])
    LM.run()'''