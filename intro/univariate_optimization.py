from scipy.optimize import minimize_scalar
import matplotlib.pyplot as plt
import numpy as np
from typing import List,Dict
from datetime import datetime

def f_ex1(x):
    y = x**2 + 2*np.sin(x*np.pi)
    return y
def problem_ex1():
    return [f_ex1,(-2,2),0]

def f_ex2(x):
    y = (x - 2) * x * (x + 2)**2
    return y
def problem_ex2():
    return [f_ex2,(-8,10),0]


def f_ex3(x):
    y = x**6-4*x**5-2*x**3+2*x+40
    return y
def problem_ex3():
    return [f_ex3,(-3,5),0.0]

def f_ex4(x):
    y = x/(1+x**2)
    return y
def problem_ex4():
    return [f_ex4,(-0.6,0.75),-0.4]

def f_ex5(x):
    y = (5*x**2-4*x+2)/(x+10)
    return y
def problem_ex5():
    return [f_ex5,(-8,9.99),-3.0]

def f_ex6(x):
    y = x**2-2*x +8/(x-1)+6
    return y
def problem_ex6():
    return [f_ex6,(1.001,10),3.0]

def f_ex7(x):
    y = x**3+12*(3-x)**2 +3
    return y
def problem_ex7():
    return [f_ex7,(-10,10),0.1]

def f_ex8(x):
    y = (3-x)**2 +3
    return y
def problem_ex8():
    return [f_ex8,(-10,10),3]

def write_log(lines):
    if glob_do_logg:
        f = open("log.txt", "a")
        now = datetime.now()
        dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
        f.write(dt_string + '\t Univariate optimization log entry'+ '\r')
        for line in lines:
            f.write('\t'+line+'\r')
        f.close()
def sol_simple_plot(problem):
    fun = problem[0]
    lb,ub = problem[1]

    x = np.linspace(lb, ub, glob_npp)
    y = fun(x)
    title = 'Problem: ' + fun.__name__ + "; Solution: simple plot"
    plt.title(title)
    plt.plot(x, y)
    plt.grid()
    plt.show()


def sol_scipy(problem,method):
    fun = problem[0]
    lb, ub = problem[1]
    x0 = problem[2]
    x = np.linspace(lb, ub, glob_npp)
    y = fun(x)
    title= 'Problem: '+fun.__name__+"; Solution: Scipy "+method
    plt.title(title)
    plt.plot(x, y,'-b', label=fun.__name__)
    plt.plot(x0, fun(x0), 'og', label="x0")

    if method == 'bounded':
        options = {'xatol': glob_tolvar_req, 'maxiter': glob_maxiter}
    else:
        options = {'xtol': glob_tolvar_req, 'maxiter': glob_maxiter}
    if method=='golden':
        result = minimize_scalar(fun, method=method,bracket=(lb,ub),bounds=(lb,ub),options=options)
    else:
        result = minimize_scalar(fun, method=method, bracket=(lb, x0, ub), bounds=(lb, ub), options=options)
    print(result)
    msg = 'Result: xopt = {0:8.4f}; fopt = {1:8.4f}; xtol = {2:12.10f}; nfev = {3}'.format(result['x'], result['fun'],glob_tolvar_req,result['nfev'])
    print(title)
    print(msg)
    write_log([title, msg])

    plt.plot(result['x'], result['fun'], 'xr', label="optimum")
    plt.legend(loc='best', fancybox=True, shadow=True)
    plt.figtext(0.5, 0.02,
                'xopt = {0:8.4f}; fopt = {1:8.4f}; nfev = {2}'.format(result['x'], result['fun'],result['nfev']),
                fontsize=10, color='darkred',ha='center', bbox=dict(facecolor='white', alpha=0.1))
    plt.grid()
    plt.show()

def sol_scipy_brent(problem):
    sol_scipy(problem,'brent')

def sol_scipy_golden(problem):
    sol_scipy(problem,'golden')

def sol_scipy_bounded_brent(problem):
    sol_scipy(problem,'bounded')


def sol_interval_halving(problem):
    fun = problem[0]
    lb, ub = problem[1]
    x0 = problem[2] # overriden by method

    x = np.linspace(lb, ub, glob_npp)
    y = fun(x)
    title= 'Problem: '+fun.__name__+'; Solution: Interval halving'
    plt.title(title)
    plt.plot(x, y,'-b', label=fun.__name__)
    plt.plot(x0, fun(x0), 'og', label="x0")
    # Interval halving algorithm
    xa=lb
    xb = ub
    x0 = (xa + xb) / 2.0
    x1 = (xa + x0) / 2.0
    x2 = (xb + x0) / 2.0
    f0=fun(x0)
    f1 = fun(x1)
    f2 = fun(x2)
    xsearch = []
    fsearch = []
    xsearch.append(x0)
    fsearch.append(f0)
    tolfun = glob_tolfun_req+100
    tolvar = glob_tolvar_req+100
    it=0

    while(tolfun > glob_tolfun_req and tolvar > glob_tolvar_req ):
        if f1 > f0 and f0 > f2:
            xa=x0
            x0=x2
            f0=f2
        elif f2 > f0 and f0 > f1:
            xb = x0
            x0 = x1
            f0 = f1
        else:
            xa = x1
            xb = x2
        x1 = (xa + x0) / 2.0
        x2 = (xb + x0) / 2.0
        f1 = fun(x1)
        f2 = fun(x2)
        tolfun = np.abs(f2-f1)
        tolvar = np.abs(xb-xa)
        it+=1
        xsearch.append(x0)
        fsearch.append(f0)

    nfev = it*2+3
    msg = 'Result: xopt = {0:8.4f}; fopt = {1:8.4f}; xtol = {2:12.10f}; nfev = {3}'.format(x0, f0, tolvar, nfev)
    print(title)
    print(msg)
    write_log([title, msg])
    plt.plot(xsearch, fsearch, '*-c', label= 'search hystory')
    plt.plot(x0, f0, 'xr', label="optimum")
    plt.legend(loc='best', fancybox=True, shadow=True)

    plt.figtext(0.5, 0.02,
                'xopt = {0:8.4f}; fopt = {1:8.4f}; xtol = {2:12.10f}; nfev = {3}'.format(x0,f0,tolvar,nfev),
                fontsize = 10,color = 'darkred',ha='center',bbox = dict(facecolor = 'white', alpha = 0.1))
    plt.grid()
    plt.show()

def sol_golden_section(problem):
    fun = problem[0]
    lb, ub = problem[1]
    x0 = problem[2] # not used by method

    x = np.linspace(lb, ub, glob_npp)
    y = fun(x)
    title= 'Problem: '+fun.__name__+'; Solution: Golden section'
    plt.title(title)
    plt.plot(x, y,'-b', label=fun.__name__)
    #plt.plot(x0, fun(x0), 'og', label="x0")
    # Interval halving algorithm
    xa=lb
    xb = ub
    l=xb -xa
    lg = 0.382*l
    x1 = xa + lg
    x2 = xb - lg
    f1 = fun(x1)
    f2 = fun(x2)
    xsearch = [x1,x2]
    fsearch = [f1,f2]
    tolfun = glob_tolfun_req+100
    tolvar = glob_tolvar_req+100
    it=0
    nfev=2
    while(tolfun > glob_tolfun_req and tolvar > glob_tolvar_req ):
        if f1 > f2:
            xa=x1
            lg = 0.382 * (xb -xa)
            x1=x2
            f1=f2
            x2 = xb - lg
            f2 = fun(x2)
            nfev += 1
            xsearch.append(x2)
            fsearch.append(f2)


        elif f2 > f1:
            xb = x2
            lg = 0.382 * (xb - xa)
            x2=x1
            f2=f1
            x1=xa+lg
            f1=fun(x1)
            nfev+=1
            xsearch.append(x1)
            fsearch.append(f1)
        else:
            xa = x1
            xb = x2
            lg = 0.382 * (xb - xa)
            x1 = xa + lg
            x2 = xb - lg
            f1 = fun(x1)
            f2 = fun(x2)
            nfev += 2
            xsearch.append(x1)
            fsearch.append(f1)
            xsearch.append(x2)
            fsearch.append(f2)
        tolfun = np.abs(f2-f1)
        tolvar = np.abs(xb-xa)
        it+=1
    print(xsearch)
    x0 = (xa+xb)/2
    f0=fun(x0)
    nfev += 1
    msg = 'Result: xopt = {0:8.4f}; fopt = {1:8.4f}; xtol = {2:12.10f}; nfev = {3}'.format(x0,f0,tolvar,nfev)
    print(title)
    print(msg)
    write_log([title,msg])
    plt.plot(xsearch, fsearch, '*-c', label= 'search hystory')
    plt.plot(x0, f0, 'xr', label="optimum")
    plt.legend(loc='best', fancybox=True, shadow=True)

    plt.figtext(0.5, 0.02,
                'xopt = {0:8.4f}; fopt = {1:8.4f}; xtol = {2:12.10f}; nfev = {3}'.format(x0,f0,tolvar,nfev),
                fontsize = 10,color = 'darkred',ha='center',bbox = dict(facecolor = 'white', alpha = 0.1))
    plt.grid()
    plt.show()

def sol_quadratic_interpolation(problem):
    fun = problem[0]
    lb, ub = problem[1]
    x0 = problem[2]

    x = np.linspace(lb, ub, glob_npp)
    y = fun(x)
    title= 'Problem: '+fun.__name__+'; Solution: quadratic interpolation'
    plt.title(title)
    plt.plot(x, y,'-b', label=fun.__name__)
    plt.plot(x0, fun(x0), 'og', label="x0")
    # Succesive quadratic approximation algorithm
    x1=lb
    x2 = x0
    x3 = ub
    f1 = fun(x1)
    f2 = fun(x2)
    f3 = fun(x3)
    nfev = 3
    xsearch = []
    fsearch = []
    tolfun = glob_tolfun_req+100
    tolvar = glob_tolvar_req+100
    it=0
    f0i_1 = f1
    x0_i_1 = x1
    while(tolfun > glob_tolfun_req and tolvar > glob_tolvar_req ):
        yiter= []
        c0 = f1
        c1 = (f2-f1)/(x2-x1)
        c2 = 1/(x3-x2)*((f3-f1)/(x3-x1)-c1)
        x0 = (x1+x2)/2 - c1/2/c2
        f0 = fun(x0)
        for xi in x:
            yiter.append(c0+c1*(xi-x1)+c2*(xi-x1)*(xi-x2))
        nfev+=1
        plt.plot(x, yiter, '-', label='it_'+str(it+1))
        xsearch.append(x0)
        fsearch.append(f0)
        # select two points that is bracketing x0
        if x0 < x2 and f0 < f2:
            x3 = x2
            f3 = f2
            x2 = x0
            f2=f0
        if x0 < x2 and f0 >= f2:
            x1 = x0
            f1=f0
        elif x0 > x2 and f0 < f2:
            x1 = x2
            f1=f2
            x2 = x0
            f2 = f0
        elif x0 > x2 and f0 >= f2:
            x3 = x0
            f3=f0
        elif x0 == x2:
            # added correction / not part of original algorithm
            if f1 < f3:
                x2 = x1 + (x2-x1)*0.5
            else:
                x2 = x2 + (x3 - x2) * 0.5
            f2 = fun(x2)
            nfev+=1
            print('Warning: correction step applied')
        else:
            print('Error: solution algorithm cannot handle situation')
            break

        tolfun = np.abs(f0i_1-f0)
        tolvar = np.abs(x0_i_1-x0)
        f0i_1=f0
        x0_i_1 = x0

        fvals = [f1,f2,f3]
        xvals = [x1, x2, x3]
        imin = np.argmin(fvals)
        fmin= fvals[imin]
        xmin = xvals[imin]
        it+=1

    msg = 'Result: xopt = {0:8.4f}; fopt = {1:8.4f}; xtol = {2:12.10f}; nfev = {3}'.format(x0, f0, tolvar, nfev)
    print(title)
    print(msg)
    write_log([title, msg])
    plt.plot(xsearch, fsearch, '*-c', label= 'search hystory')
    plt.plot(x0, f0, 'xr', label="optimum")
    plt.legend(loc='best', fancybox=True, shadow=True)

    plt.figtext(0.5, 0.02,
                'xopt = {0:8.4f}; fopt = {1:8.4f}; xtol = {2:12.10f}; nfev = {3}'.format(x0,f0,tolvar,nfev),
                fontsize = 10,color = 'darkred',ha='center',bbox = dict(facecolor = 'white', alpha = 0.1))
    plt.grid()
    plt.show()

def get_msg_available_solution_methods(dict:Dict[int,tuple]):
    msg= 'Select solution method:\n'
    for key, value in dict.items():
        msg+= str(key)+ ': '+value.__name__+'\n'
    return msg

def get_msg_available_problems(dict:Dict):
    msg= 'Select problem:\n'
    for key, value in dict.items():
        msg+= str(key)+ ': '+(value[0]).__name__+'\n'
    return msg

if __name__ == '__main__':
    glob_do_logg = True
    glob_npp = 1000 # number of plot points
    glob_tolfun_req = 1e-16
    glob_tolvar_req = 1e-4
    glob_maxiter = 500
    glob_random_x0 = False
    glob_seed = 101

    dproblems= {
        1:problem_ex1(),
        2:problem_ex2(),
        3:problem_ex3(),
        4:problem_ex4(),
        5:problem_ex5(),
        6:problem_ex6(),
        7:problem_ex7(),
        8: problem_ex8()
    }

    dsols = {
        1:sol_simple_plot,
        2:sol_scipy_brent,
        3:sol_scipy_golden,
        4:sol_scipy_bounded_brent,
        5:sol_interval_halving,
        6:sol_golden_section,
        7:sol_quadratic_interpolation}

    # select problem
    msg=get_msg_available_problems(dproblems)
    problem = int(input(msg))
    problem_data=dproblems.get(problem)

    if problem_data is not None:
        print('Selected problem:',(problem_data[0]).__name__)
    else:
        print('Error: Unknown problem selected:',problem)
        exit(-1)
    # select solution method
    msg=get_msg_available_solution_methods(dsols)
    sol = int(input(msg))
    solfun=dsols.get(sol)

    if solfun is not None:
        print('Selected solution method:',solfun.__name__)
    else:
        print('Error: Unknown solution method selected:',sol)
        exit(-1)
    if glob_random_x0:
        if glob_seed != 0:
            np.random.seed(glob_seed+1)
        problem_data[2] = np.random.rand() * (problem_data[1][1] - problem_data[1][0]) + problem_data[1][0]
    solfun(problem_data)
    print('Script executed sucessfully')
