import sympy as sym         #When working with symbols
from scipy.linalg import expm
import numpy as num         #When working with numerals
import matplotlib.pyplot as plt
from scipy.linalg import expm
from math import factorial
import pandas as pd


# Define Initial Values (See anexo_A.txt)
a = 11
b = 3
c = 4
d = 12
e = 8

#Define symbolic variables
x1, x2, x3, x4 = sym.symbols('x1 x2 x3 x4')
u1, u2 = sym.symbols('u1 u2')
u, alpha, theta, q, deltaT, deltaC = sym.symbols('u alpha theta q deltaT deltaC')

#Define functions
x1_dot = x1**2 * sym.cos(x2**3) + x3 + a*x1*x4**2 + b*x2*u1
x2_dot = x1 + x2**2 + x3 + c*x4
x3_dot = d*x1*x2*x3 + u1 + u2
x4_dot = x1**2 + x2*x3 + e*x4 + u1

#Define state vector for x values
x_state = sym.Matrix([x1, x2, x3, x4])

#Define state vector for u Control Values
uc_state = [u1, u2]

#Defines Jacobian matrixes
x_star_jacobian = sym.Matrix([x1_dot, x2_dot, x3_dot, x4_dot]).jacobian(x_state)
uc_jacobian = sym.Matrix([x1_dot, x2_dot, x3_dot, x4_dot]).jacobian(uc_state)

#Use to substitute values in Jacobians
equil_state = {x1: 1, x2: 0, x3: -1, x4: 0}
equilibrium_input_control = {u1: -1, u2: 1}

#Solve Jacobians at equilibrium state
x_star_eval = x_star_jacobian.subs(equil_state).subs(equilibrium_input_control) #Matrix A
uc_eval = uc_jacobian.subs(equil_state).subs(equilibrium_input_control) #Matrix B

# Print results obtained
print("Matrix A evaluation at equilibrium:")
sym.pprint(x_star_eval)

print("\nMatrix B Evaluation at equilibrium:")
sym.pprint(uc_eval)

#---------------End Part 1------------------

    #Longitudinal Analysis

B_longitudinal = sym.Matrix([[0.0813, 0.0218],
                             [0, -0.0012],
                             [0, 0],
                             [0, -0.0374]])
C_longitudinal = sym.Matrix([[1, 0, 0, 0],
                             [0, -1, 1, 0],
                             [0, 0, 0, 1]]) 
#Define A
A_longitudinal = sym.Matrix([[0.239, 20.643, -32.193, 0],
                       [-0.0010, -1.0856, 0.0056, 0.9215],
                       [0, 0, 0, 1],
                       [2.1426, 0, -0.2892, -0.6621]])

#Eigenvalues of A
A_longitudinal_eigen = list(A_longitudinal.eigenvals().keys())


print("\nEigenvalues of A (Longitudinal):\n")
sym.pprint(A_longitudinal_eigen)
real_part, imag_part = A_longitudinal_eigen[0].as_real_imag()
omega_longitudinal = sym.sqrt(real_part**2 + imag_part**2)
amortecimento = real_part / omega_longitudinal
T_p = (2*sym.pi)/(omega_longitudinal*(sym.sqrt(1-amortecimento**2)))
omega_p_longitudinal = abs(imag_part)

print("\nValue of omega_n (Natural Frequency) (Longitudinal):\n")
sym.pprint(omega_longitudinal)

print("\nValue of lambda (coef de amortecimento) (Longitudinal):\n")
sym.pprint(amortecimento)

print("\nValue of T_p (Period) (Longitudinal):\n")
sym.pprint(T_p)

print("\nValue of omega_p (Period of oscillation) (Longitudinal)\n")
sym.pprint(omega_p_longitudinal)

AB = A_longitudinal * B_longitudinal
A2B = A_longitudinal**2 * B_longitudinal
A3B = A_longitudinal**3 * B_longitudinal
Delta_matrix = sym.Matrix.hstack(B_longitudinal, AB, A2B, A3B)
print("\nDelta Matrix (Longitudinal) =\n")
sym.pprint(Delta_matrix)

CB = C_longitudinal * B_longitudinal
CAB = C_longitudinal * A_longitudinal * B_longitudinal
CA2B = C_longitudinal * A_longitudinal**2 * B_longitudinal
CA3B = C_longitudinal * A_longitudinal**3 * B_longitudinal
D = sym.zeros(3, 2)
Gama_matrix = sym.Matrix.hstack(CB, CAB, CA2B, CA3B, D)
print("\nGama Matrix (Longitudinal)=\n")
sym.pprint(Gama_matrix)

CA = C_longitudinal * A_longitudinal
CA2 = C_longitudinal * A_longitudinal**2
CA3 = C_longitudinal * A_longitudinal**3

Theta_matrix = sym.Matrix.vstack(C_longitudinal, CA, CA2, CA3)
print("\nTheta Matrix (Longitudinal)=\n")
sym.pprint(Theta_matrix)
    #Lateral Analysis
#Define A
A_lateral = sym.Matrix([[-0.095, 0.129, 0.0643, -0.998],
                        [0, 0, 1, 0.0228],
                        [-4.763, 0, -3.1885, 0.8535],
                        [2.1426, 0, -0.2892, -0.6621]])
B_lateral = sym.Matrix([[0, 0.0006],
                        [0, 0],
                        [0.0137, 0.0069],
                        [0.0009, -0.1031]])
C_lateral = sym.Matrix([[1, 0, 0, 0],
                        [0, 1, 0, 0]])

#Eigenvalues of A
A_lateral_eigen = list(A_lateral.eigenvals().keys())

print("\nEigenvalues of A (Latero-directional):\n")
print(A_lateral_eigen)

real_part_lateral, imag_part_lateral = A_lateral_eigen[1].as_real_imag()
omega_lateral = sym.sqrt(real_part_lateral**2 + imag_part_lateral**2)
amortecimento_lateral = real_part_lateral / omega_lateral
T_p_lateral = (2*sym.pi)/(omega_lateral*(sym.sqrt(1-amortecimento_lateral**2)))
omega_p_lateral = abs(imag_part_lateral)

print("\nValue of omega_n (Natural Frequency) (Latero-directional):\n")
sym.pprint(omega_lateral)

print("\nValue of lambda (coef de amortecimento) (Latero-directional):\n")
sym.pprint(amortecimento_lateral)

print("\nValue of T_p (Period) (Latero-directional):\n")
sym.pprint(T_p_lateral)

print("\nValue of omega_p (Period of oscillation) (Latero-directional)\n")
sym.pprint(omega_p_lateral)

AB_lateral = A_lateral * B_lateral
A2B_lateral = A_lateral**2 * B_lateral
A3B_lateral = A_lateral**3 * B_lateral
Delta_matrix_lateral = sym.Matrix.hstack(B_lateral, AB_lateral, A2B_lateral, A3B_lateral)
print("\nDelta Matrix (Lateral)=\n")
sym.pprint(Delta_matrix_lateral)

CB_lateral = C_lateral * B_lateral
CAB_lateral = C_lateral * A_lateral * B_lateral
CA2B_lateral = C_lateral * A_lateral**2 * B_lateral
CA3B_lateral = C_lateral * A_lateral**3 * B_lateral
D_lateral = sym.zeros(2, 2)
Gama_matrix_lateral = sym.Matrix.hstack(CB_lateral, CAB_lateral, CA2B_lateral, CA3B_lateral, D_lateral)
print("\nGama Matrix (Lateral) =\n")
sym.pprint(Gama_matrix_lateral)

CA_lateral = C_lateral * A_lateral
CA2_lateral = C_lateral * A_lateral**2
CA3_lateral = C_lateral * A_lateral**3

Theta_matrix = sym.Matrix.vstack(C_lateral, CA_lateral, CA2_lateral, CA3_lateral)
print("\nTheta Matrix (Lateral)=\n")
sym.pprint(Theta_matrix)

#------------Simulation----------------------
h=0.01
def simulate(Ad,Bd,x0,u,tsim,h):
 
    num_steps=int(tsim/h)
    x_traj=num.zeros((num_steps,4))
    time=num.arange(0,tsim,h)
 
    x_k=x0
    for k in range(num_steps):
    
        x_k=Ad @ x_k+ Bd @ u
        x_traj[k,:]=x_k
        
    return time,x_traj
 
#Data Storage
 
def evolution(time, estados, labels):
    df = pd.DataFrame(estados, columns=labels)
    df['Time'] = time
    
    return df

def plot_simulate(title,labels,descriptions,time,states,figsize=(12,8)):
  
    plt.figure(figsize=figsize)
 
    for i in range(4):
  
        plt.subplot(2,2,i+1)
        plt.plot(time,states[:,i],label=labels[i])
        plt.title(descriptions[i])
        plt.xlabel('Tempo(s)')
        plt.ylabel(labels[i])
        plt.legend()
    
    plt.suptitle(title,fontsize=16)
    plt.tight_layout()
    plt.show()






def find_Bd(A, B, h, N):
    I = num.eye(A.shape[0])  
    Bd = num.zeros_like(B)  

    for n in range(N):
        term = num.linalg.matrix_power(A, n) * (h ** (n + 1)) / factorial(n + 1)
        Bd += term @ B
    return Bd




A_long = num.array([[0.239, 20.643, -32.193, 0],
                    [-0.0010, -1.0856, 0.0056, 0.9215],
                    [0, 0, 0, 1],
                    [2.1426, 0, -0.2892, -0.6621]])

B_long = num.array([[0.0813, 0.0218],
                    [0, -0.0012],
                    [0, 0],
                    [0, -0.0374]])

Ad_long=expm(A_long*h)
Bd_long=find_Bd(A_long,B_long,0.01,10)


A_lat = num.array([[-0.095, 0.129, 0.0643, -0.998],
                   [0, 0, 1, 0.0228],
                   [-4.763, 0, -3.1885, 0.8535],
                   [2.1426, 0, -0.2892, -0.6621]])

B_lat = num.array([[0, 0.0006],
                   [0, 0],
                   [0.0137, 0.0069],
                   [0.0009, -0.1031]])

Ad_lat=expm(A_lat*h)
Bd_lat=find_Bd(A_lat,B_lat,0.01,10)

x0_long=num.array([40, 0.5, 0.1, 1.5])

x0_lat=num.array([0.2, 0.3, 2, 1.5])

#Control Vectors
u_long=num.array([0.8,0.3])

u_lat=num.array([0.4,0.2])

#Long Simulation

time_long,state_long=simulate(Ad_long,Bd_long,x0_long,u_long,200,0.01)
plot_simulate('Modelo de voo longitudinal',['u','alfa','theta','q'],
                  ['velocidade vertical','ângulo de ataque','ângulo de arfagem','taxa de arfagem'],time_long,state_long,(12,8))

df_long = evolution(time_long, state_long, ['u', 'alfa', 'theta', 'q'])
df_long.to_excel("evolucao_voo_longitudinal.xlsx", index=False)


time_lat,estado_lat=simulate(Ad_lat,Bd_lat,x0_lat,u_lat,20,0.01)
plot_simulate('Modelo de voo latero-direcional',['beta','phi','p','r'],
                  ['ângulo de derrapagem','ângulo de pranchamento','taxa de guinada','taxa de rolamento'],time_lat,estado_lat,(12,8))

df_lat = evolution(time_lat, estado_lat, ['beta', 'phi', 'p', 'r'])
df_lat.to_excel("evolucao_voo_laterodirecional.xlsx", index=False)