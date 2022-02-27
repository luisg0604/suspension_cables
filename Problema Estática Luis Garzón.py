import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as scipy

print(
    "Se tiene un anillo de masa despreciable suspendido en por dos cables de longitud s cada uno, con una densidad lineal determinada")
print(
    "En este anillo se carga un cuerpo de cierta masa finita, pero no despreciable. Se necesita saber la tensión que ejerce el cable sobre sus extremos")
print("Además, se quiere conocer la geometría que toma el cable después de colgarse dicha masa.")
print("")
print("Ingrese los datos iniciales del problema:")
mass = float(input("Ingrese la masa del objeto suspendido [kg]: "))
sarc = float(input("Ingrese la longitud por cable [m]: "))
density = float(input("Ingrese la densidad lineal del cable [kg/m]: "))
light = float(input("Ingrese la distancia entre los puntos de de suspensión [m]: "))

# Primero es conveniente mostrar el cable suspendido antes de colgar la carga.
print(" ")


def f(t):  # Esta función calcula la tensión que sufre el anillo por lados
    f1 = (t / (density * 9.81)) * np.sinh((density * 9.81 * (light / 2)) / t) - sarc
    return f1


T1 = scipy.fsolve(f, 80 * ((light * light * density) / (sarc * sarc)))
print(T1)
x = np.arange(-light / 2, light / 2, 0.05)

y1 = (T1 / (density * 9.81)) * (np.cosh((density * 9.81 * x) / T1) - 1)
y1o = (T1 / (density * 9.81)) * (np.cosh((density * 9.81 * (light / 2)) / T1) - 1)
print(y1o)


# Se obtiene un sistema de ecuaciones al momento de colocar la carga
# non linear system


def equation_system(z):
    xa, ya, Ta, To = z
    u = (To / (density * 9.81))*(np.cosh((density * 9.81 * xa) / To) - 1) - ya
    v = (To / (density * 9.81))*(np.sinh((density * 9.81 / To)*(xa + (light / 2)))) - 21
    w = (Ta*Ta - To*To)**(1/2) - (density * 9.81) * ya
    t = 2 * float(Ta - To) - 9.81 * mass
    return [t, u, v, w]
t2 = scipy.fsolve(equation_system, [1, 4, 500, 800], maxfev=100000000)
print(t2)

y2 = (t2[4] / (density * 9.81)) * (np.cosh((density * 9.81 * x) / t2[4]) - 1)

plt.plot(x, y1, 'o', color=(0.2, 0.1, 0.4))
plt.hold(True)
plt.plot(x, y2, '-', color='g')
plt.grid()
plt.axis('equal')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Suspensión del cable antes y después de colocar la carga')
plt.show()

'''
#linear system
a1=7.91
a2=-5.16
a3=3.86
a4=-13.44
b1=2.771
b2=-1.6
a=numpy.array([[a1,a2],[a3,a4]])
b=numpy.array([b1,b2])
X=numpy.linalg.solve(a,b)
print(X)
'''

'''
#non linear system
def f(z):
    x,y=z
    f1=numpy.cosh(15*x/(11180+147*y))-1-(60-y)*(15)/(11180+147*y)
    f2=numpy.cosh(15*(500-x)/(11180+147*y))-1-(40-y)*(15)/(11180+147*y)
    return [f1,f2]
X=scipy.fsolve(f,[250,1])
print(X)

'''
'''
def f(x):
    f1=2.*numpy.cos(2.*x)+numpy.sin(x)
    return f1
X=scipy.fsolve(f,1.14)
print(X*180./numpy.pi)
'''
'''
def f(x):
    f1=x*numpy.sinh(1.2*9.81*20/x)-21*(1.2*9.81)
    return f1
X=scipy.fsolve(f,433)
print(X)
'''
'''
def equation_system(z):
    o1,o2,o3=z
    myu=2.0*9.81
    u=o1*(numpy.sinh(myu*(o2+32)/o1)-numpy.sinh(myu*(o2)/o1))/myu-87
    v=o1*(numpy.cosh(myu*(o2)/o1)-1)/myu-o3
    w=o1*(numpy.cosh(myu*(o2+32)/o1)-1)/myu-o3-80
    return [u,v,w]
X=scipy.fsolve(equation_system,[200,10,10])
print(X)
'''

'''
def f(x):
    f1=(50.435/2)*(numpy.cosh(2.*(24.)/50.435)-1.)-x
    return f1
X=scipy.fsolve(f,12.)
print(X)

'''

'''
def equation_system(z):
    xa, ya, to = z
    u = to * (numpy.cosh(6 * (xa + 40.) / to) - 1) / 6 - ya - 8.
    v = to * (numpy.cosh(6 * (xa) / to) - 1.) / 6 - ya
    w = 5 * ya - xa - 10.

    return [u, v, w]
'''
'''
X = scipy.fsolve(equation_system, [-2., 1., 300.])
print(X)
'''
