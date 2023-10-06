import numpy as np
#Optimality criteria example
A=[[2.,0,0,-5,-2],
   [0,4,0,1,-1],
   [0,0,6,3,-2],
   [5,-1,-3,0,0],
   [2,1,2,0,0]]
b= [0,0,0,3,6]
A=np.array(A,dtype=float)
b=np.array(b,dtype=float)
x_sol=np.linalg.solve(A,b)
print("x_sol=",x_sol)

A1=np.array(
   [[2.,0,0,-5,-2],
   [0,4,0,1,-1],
   [0,0,6,3,-2],
   [5,-1,-3,999999999999999999999999999999,0],
   [2,1,2,0,0]],dtype=float)
x_sol1=np.linalg.solve(A1,b)
print("x_sol1=",x_sol1)

A2 = np.delete(A,3,0)
A2 = np.delete(A2,3,1)
print(A2)
b2 = np.delete(b,3,0)
x_sol2=np.linalg.solve(A2,b2)
print("x_sol2=",x_sol2)