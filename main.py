import sympy as sym         #When working with symbols
import matplotlib as matpl  #When working with graphs
import numpy as num         #When working with numerals

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
omega_p_longitudinal = abs((2 * sym.pi)/imag_part)

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

real_part_lateral, imag_part_lateral = A_lateral_eigen[0].as_real_imag()
omega_lateral = sym.sqrt(real_part_lateral**2 + imag_part_lateral**2)
amortecimento_lateral = real_part_lateral / omega_lateral
T_p_lateral = (2*sym.pi)/(omega_lateral*(sym.sqrt(1-amortecimento_lateral**2)))
omega_p_lateral = abs((2 * sym.pi)/imag_part_lateral)

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