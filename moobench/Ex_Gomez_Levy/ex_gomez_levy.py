"""
Example 15.4 from Nocedal, J, and S J Wright. 2006. Numerical Optimization. Springer New York.
Used as example in SciPy documentation
x_opt={1.4,1.7}
f_opt = 0.8
"""
import math
try:
    from moobench.optbase import *
except ImportError:
    pass

class EX_GomezLevy_AnMod(SimpleInputOutputArrayAnalysisExecutor):
    def __init__(self):
        super().__init__(2,2)

    def analyze(self):
        x = self.inarray[0]
        y = self.inarray[1]

        f = 4*x**2-2.1*x**4+1/3*x**6+x*y-4*y**2+4*y**4
        g = -math.sin(4*math.pi*x)+2*(math.sin(2*math.pi*y))**2

        self.outarray[0] = f
        self.outarray[1] = g

        return AnalysisResultType.OK

class EX_GomezLevy_OptProb(OptimizationProblem):
    def __init__(self,name=''):
        if name == '':
            name = 'GomezLevy'
        super().__init__(name)
        am = EX_GomezLevy_AnMod()
        self.add_design_variable(DesignVariable('x', NdArrayGetSetConnector(am.inarray, 0), -1.0,0.75))
        self.add_design_variable(DesignVariable('y', NdArrayGetSetConnector(am.inarray, 1), -1.0,1.0))
        self.add_objective(DesignObjective('f', NdArrayGetConnector(am.outarray,0)))
        self.add_constraint(DesignConstraint('g', NdArrayGetConnector(am.outarray, 1), 1.5,ConstrType.LT))
        self.add_analysis_executor(am)

