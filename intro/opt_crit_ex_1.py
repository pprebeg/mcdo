import numpy as np
import time
#Optimality criteria example
A=[[1,2,-1,0,0],
   [2,-2,3,0,0],
   [4,0,0,-1,-2],
   [0,2,0,-2,2],
   [0,0,8,1,-3]]
b= [6,12,0,0,0]
A=np.array(A)
b=np.array(b)
start_time = time.perf_counter()
Ainv = np.linalg.inv(A)
x =np.matmul(Ainv,b)
end_time = time.perf_counter()
elapsed_time= end_time-start_time
print("elapsed_time:",elapsed_time)
print("Ainv=",Ainv)
print("x=",x)
xdot =np.dot(Ainv,b)
print("xdot=",xdot)
start_time = time.perf_counter()
xsolve=np.linalg.solve(A,b)
end_time = time.perf_counter()
elapsed_time= end_time-start_time
print("elapsed_time:",elapsed_time)
print("xsolve=",xsolve)

xdot2=Ainv@b
print(xdot2)