import sympy as sym         #When working with symbols
import matplotlib as matpl  #When working with graphs
import numpy as num         #When working with numerals

# Define Initial Values
a = 11
b = 3
c = 4
d = 12
e = 8

#Define symbolic variables
x1, x2, x3, x4 = sym.symbols('x1 x2 x3 x4')
u1, u2 = sym.symbols('u1 u2')

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
x_star_eval = x_star_jacobian.subs(equil_state).subs(equilibrium_input_control)
uc_eval = uc_jacobian.subs(equil_state).subs(equilibrium_input_control)

# Print results obtained
print("x_star Jacobian evaluation at equilibrium:")
sym.pprint(x_star_eval)

print("\nControl Vector Jacobian Evaluation at equilibrium:")
sym.pprint(uc_eval)