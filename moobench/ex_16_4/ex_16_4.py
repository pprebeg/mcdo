try:
    from moobench.optbase import *
except ImportError:
    pass
from moobench.optlib_scipy import ScipyOptimizationAlgorithm

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