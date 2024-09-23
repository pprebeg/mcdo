"""
Example 15.4 from Nocedal, J, and S J Wright. 2006. Numerical Optimization. Springer New York.
Used as example in SciPy documentation
x_opt={1.4,1.7}
f_opt = 0.8
"""

try:
    from moobench.optbase import *
except ImportError:
    pass

class EX_16_4_AnMod(SimpleInputOutputArrayAnalysisExecutor):
    def __init__(self):
        super().__init__(2,4)

    def analyze_old(self):
        self.outarray[0] = (self.inarray[0] - 1)**2.0 + (self.inarray[1] - 2.5)**2.0
        self.outarray[1] =  self.inarray[0] - 2 * self.inarray[1]
        self.outarray[2] = -self.inarray[0] - 2 * self.inarray[1]
        self.outarray[3] = -self.inarray[0] + 2 * self.inarray[1]
        return AnalysisResultType.OK

    def analyze(self):
        x1=self.inarray[0]
        x2 = self.inarray[1]
        o = (x1-1)**2.0+(x2-2.5)**2.0
        g1 =  x1 - 2 * x2
        g2 = -x1 - 2 * x2
        g3 = -x1 + 2 * x2
        self.outarray[0] = o
        self.outarray[1] =  g1
        self.outarray[2] =  g2
        self.outarray[3] =  g3
        return AnalysisResultType.OK

class EX_16_4_OptProb(OptimizationProblem):
    def __init__(self,name=''):
        if name == '':
            name = 'EX_16_4'
        super().__init__(name)
        am = EX_16_4_AnMod()
        self.add_design_variable(DesignVariable('x1', NdArrayGetSetConnector(am.inarray, 0), 0.0,5.0))
        self.add_design_variable(DesignVariable('x2', NdArrayGetSetConnector(am.inarray, 1), 0.0,5.0))
        self.add_objective(DesignObjective('obj', NdArrayGetConnector(am.outarray,0)))
        self.add_constraint(DesignConstraint('g1', NdArrayGetConnector(am.outarray, 1), -2.0,ConstrType.GT))
        self.add_constraint(DesignConstraint('g2', NdArrayGetConnector(am.outarray, 2), -6.0,ConstrType.GT))
        self.add_constraint(DesignConstraint('g3', NdArrayGetConnector(am.outarray, 3), -2.0,ConstrType.GT))
        self.add_analysis_executor(am)

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    x1_vals = np.linspace(0, 4, 200)
    x2_vals = np.linspace(0, 3, 200)
    X1, X2 = np.meshgrid(x1_vals, x2_vals)


    o = (X1 - 1) ** 2 + (X2 - 2.5) ** 2
    g1 = X1 - 2 * X2 +2 # g1
    g2 = -X1 - 2 * X2 + 6 # g2
    g3 = -X1 + 2 * X2 + 2  # g3


    fig, ax = plt.subplots()


    CS = ax.contour(X1, X2, o, levels=20, cmap="viridis")
    ax.clabel(CS, inline=True, fontsize=10)


    contour_g1 = ax.contour(X1, X2, g1, levels=[0], colors='r', linewidths=2)
    contour_g2 = ax.contour(X1, X2, g2, levels=[0], colors='b', linewidths=2)
    contour_g3 = ax.contour(X1, X2, g3, levels=[0], colors='g', linewidths=2)


    ax.clabel(contour_g1, inline=True, fontsize=10, fmt='g1')
    ax.clabel(contour_g2, inline=True, fontsize=10, fmt='g2')
    ax.clabel(contour_g3, inline=True, fontsize=10, fmt='g3')


    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_title('Objective function and constraints')

    # Prikaz grafa
    plt.grid(True)
    plt.show()