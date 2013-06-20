#!/usr/bin/env python
# encoding: utf-8
"""
untitled.py

Created by Andrew Ning on 2013-06-03.
Copyright (c) NREL. All rights reserved.
"""

import numpy as np
from math import sqrt

# omega = np.linspace(40, 80, 200)

# rovert = 10

# Cxb = 6.0

# Cx0 = np.maximum(0.6, 1 + 0.2/Cxb*(1-2.0*omega/rovert))


# ptL = 6*rovert
# ptR = 7*rovert

# A = np.array([[ptL**3, ptL**2, ptL, 1],
#               [ptR**3, ptR**2, ptR, 1],
#               [3*ptL**2, 2*ptL, 1, 0],
#               [3*ptR**2, 2*ptR, 1, 0]])
# b = np.array([1 + 0.2/Cxb*(1-2.0*ptL/rovert), 0.6, -0.4/Cxb/rovert, 0.0])

# coeff = np.linalg.solve(A, b)

# Cx = 0.6*np.ones_like(omega)

# idx = omega < ptL
# Cx[idx] = np.maximum(0.6, 1 + 0.2/Cxb*(1-2.0*omega[idx]/rovert))
# idx = np.logical_and(omega >= ptL, omega <= ptR)
# Cx[idx] = coeff[0]*omega[idx]**3 + coeff[1]*omega[idx]**2 + coeff[2]*omega[idx] + coeff[3]

# import matplotlib.pyplot as plt
# plt.plot(omega, Cx0)
# plt.plot(omega, Cx)
# plt.show()






# lambda0 = 0.2
# beta = 0.6
# eta = 1.0
# alpha = 0.287226983996

# lambdax = np.linspace(0, 3, 2000)

# lambdap = sqrt(alpha/(1-beta))

# chi = np.ones_like(lambdax)

# idx = lambdax >= lambdap
# chi[idx] = alpha/lambdax[idx]**2

# idx = np.logical_and(lambdax > lambda0, lambdax < lambdap)
# chi[idx] = 1 - beta*((lambdax[idx]-lambda0)/(lambdap-lambda0))**eta



# ptL = 0.9*lambda0
# ptR = 1.1*lambda0

# A = np.array([[ptL**3, ptL**2, ptL, 1],
#               [ptR**3, ptR**2, ptR, 1],
#               [3*ptL**2, 2*ptL, 1, 0],
#               [3*ptR**2, 2*ptR, 1, 0]])
# b = np.array([1.0, 1-beta*((ptR-lambda0)/(lambdap-lambda0))**eta, 0.0, -beta*eta*((ptR-lambda0)/(lambdap-lambda0))**(eta-1)/(lambdap-lambda0)])

# coeff = np.linalg.solve(A, b)

# chi2 = np.ones_like(lambdax)

# idx = lambdax >= lambdap
# chi2[idx] = alpha/lambdax[idx]**2

# idx = np.logical_and(lambdax > ptR, lambdax < lambdap)
# chi2[idx] = 1 - beta*((lambdax[idx]-lambda0)/(lambdap-lambda0))**eta

# idx = np.logical_and(lambdax > ptL, lambdax < ptR)
# chi2[idx] = coeff[0]*lambdax[idx]**3 + coeff[1]*lambdax[idx]**2 + coeff[2]*lambdax[idx] + coeff[3]


# import matplotlib.pyplot as plt
# plt.plot(lambdax, chi)
# plt.plot(lambdax, chi2)
# plt.show()





def cubicspline(ptL, ptR, fL, fR, gL, gR, pts):

    A = np.array([[ptL**3, ptL**2, ptL, 1],
                  [ptR**3, ptR**2, ptR, 1],
                  [3*ptL**2, 2*ptL, 1, 0],
                  [3*ptR**2, 2*ptR, 1, 0]])
    b = np.array([fL, fR, gL, gR])

    coeff = np.linalg.solve(A, b)

    value = coeff[0]*pts**3 + coeff[1]*pts**2 + coeff[2]*pts + coeff[3]

    return value







# omega = np.linspace(20, 150, 2000)

# rovert = 50/1.63
# E = 1

# Ctheta = 1.5

# sigma = np.zeros_like(omega)
# sigma2 = np.zeros_like(omega)

# idx = omega/Ctheta < 20
# Cthetas = 1.5 + 10.0/omega[idx]**2 - 5/omega[idx]**3
# sigma[idx] = 0.92*E*Cthetas/omega[idx]/rovert

# offset = (10.0/(20*Ctheta)**2 - 5/(20*Ctheta)**3)
# Cthetas = 1.5 + 10.0/omega[idx]**2 - 5/omega[idx]**3 - offset
# sigma2[idx] = 0.92*E*Cthetas/omega[idx]/rovert


# idx = np.logical_and(omega/Ctheta >= 20, omega/Ctheta <= 1.63*rovert)
# sigma[idx] = 0.92*E*Ctheta/omega[idx]/rovert

# sigma2[idx] = 0.92*E*Ctheta/omega[idx]/rovert

# idx = omega/Ctheta > 1.63*rovert
# sigma[idx] = E*(1.0/rovert)**2*(0.275 + 2.03*(Ctheta/omega[idx]*rovert)**4)


# alpha1 = 0.92/1.63 - 2.03/1.63**4

# ptL = 1.63*rovert*Ctheta - 1
# ptR = 1.63*rovert*Ctheta + 1
# fL = 0.92*E*Ctheta/ptL/rovert
# fR = E*(1.0/rovert)**2*(alpha1 + 2.03*(Ctheta/ptR*rovert)**4)
# gL = -0.92*E*Ctheta/rovert/ptL**2
# gR = -E*(1.0/rovert)*2.03*4*(Ctheta/ptR*rovert)**3*Ctheta/ptR**2
# print ptL, fL, ptR, fR
# coeff = cubicspline(ptL, ptR, fL, fR, gL, gR)

# idx = np.logical_and(omega > ptL, omega < ptR)
# sigma2[idx] = coeff[0]*omega[idx]**3 + coeff[1]*omega[idx]**2 + coeff[2]*omega[idx] + coeff[3]

# idx = omega > ptR
# sigma2[idx] = E*(1.0/rovert)**2*(alpha1 + 2.03*(Ctheta/omega[idx]*rovert)**4)

# # alpha2 = 1.63**3*0.92/4
# # alpha1 = 0.92/1.63 - alpha2/1.63**4
# # sigma2[idx] = E*(1.0/rovert)**2*(alpha1 + alpha2*(Ctheta/omega[idx]*rovert)**4)



# # ptL = 6*rovert
# # ptR = 7*rovert

# # A = np.array([[ptL**3, ptL**2, ptL, 1],
# #               [ptR**3, ptR**2, ptR, 1],
# #               [3*ptL**2, 2*ptL, 1, 0],
# #               [3*ptR**2, 2*ptR, 1, 0]])
# # b = np.array([1 + 0.2/Cxb*(1-2.0*ptL/rovert), 0.6, -0.4/Cxb/rovert, 0.0])

# # coeff = np.linalg.solve(A, b)

# # Cx = 0.6*np.ones_like(omega)

# # idx = omega < ptL
# # Cx[idx] = np.maximum(0.6, 1 + 0.2/Cxb*(1-2.0*omega[idx]/rovert))
# # idx = np.logical_and(omega >= ptL, omega <= ptR)
# # Cx[idx] = coeff[0]*omega[idx]**3 + coeff[1]*omega[idx]**2 + coeff[2]*omega[idx] + coeff[3]

# import matplotlib.pyplot as plt
# plt.plot(omega, sigma)
# plt.plot(omega, sigma2)
# plt.show()






# omegavec = np.linspace(5, 100, 2000)
# ctau = []
# ctau2 = []
# rovert = 10

# for omega in omegavec:

#     ptL1 = 9
#     ptR1 = 11

#     ptL2 = 8.7*rovert - 1
#     ptR2 = 8.7*rovert + 1

#     if omega < ptL1:
#         C_tau2 = sqrt(1.0 + 42.0/omega**3 - 42.0/10**3)

#     elif omega >= ptL1 and omega <= ptR1:
#         fL = sqrt(1.0 + 42.0/ptL1**3 - 42.0/10**3)
#         fR = 1.0
#         gL = -63.0/ptL1**4/fL
#         gR = 0.0
#         C_tau2 = cubicspline(ptL1, ptR1, fL, fR, gL, gR, omega)

#     elif omega > ptR1 and omega < ptL2:
#         C_tau2 = 1.0

#     elif omega >= ptL2 and omega <= ptR2:
#         fL = 1.0
#         fR = 1.0/3.0*sqrt(ptR2/rovert) + 1 - sqrt(8.7)/3
#         gL = 0.0
#         gR = 1.0/6/sqrt(ptR2*rovert)
#         C_tau2 = cubicspline(ptL2, ptR2, fL, fR, gL, gR, omega)

#     else:
#         C_tau2 = 1.0/3.0*sqrt(omega/rovert) + 1 - sqrt(8.7)/3


#     if (omega < 10):
#         C_tau = sqrt(1.0 + 42.0/omega**3)

#     elif (omega > 8.7*rovert):
#         C_tau = 1.0/3.0*sqrt(omega/rovert)

#     else:
#         C_tau = 1.0

#     ctau.append(C_tau)
#     ctau2.append(C_tau2)

# import matplotlib.pyplot as plt
# plt.plot(omegavec, ctau)
# plt.plot(omegavec, ctau2)
# plt.show()







omegavec = np.linspace(.7, 150, 2000)
cx0 = []
cx = []
rovert = 20

constant = 1 + 1.83/1.7 - 2.07/1.7**2
Cxb = 6.0  # clamped-clamped

ptL1 = 1.7-0.25
ptR1 = 1.7+0.25

ptL2 = 0.5*rovert - 1.0
ptR2 = 0.5*rovert + 1.0

ptL3 = (0.5+Cxb)*rovert - 1.0
ptR3 = (0.5+Cxb)*rovert + 1.0


for omega in omegavec:

    if omega < ptL1:
        Cx = constant - 1.83/omega + 2.07/omega**2

    elif omega >= ptL1 and omega <= ptR1:

        fL = constant - 1.83/ptL1 + 2.07/ptL1**2
        fR = 1.0
        gL = 1.83/ptL1**2 - 4.14/ptL1**3
        gR = 0.0
        Cx = cubicspline(ptL1, ptR1, fL, fR, gL, gR, omega)

    elif omega > ptR1 and omega < ptL2:
        Cx = 1.0

    elif omega >= ptL2 and omega <= ptR2:

        fL = 1.0
        fR = 1 + 0.2/Cxb*(1-2.0*ptR2/rovert)
        gL = 0.0
        gR = -0.4/Cxb/rovert
        Cx = cubicspline(ptL2, ptR2, fL, fR, gL, gR, omega)

    elif omega > ptR2 and omega < ptL3:
        Cx = 1 + 0.2/Cxb*(1-2.0*omega/rovert)

    elif omega >= ptL3 and omega <= ptR3:

        fL = 1 + 0.2/Cxb*(1-2.0*ptL3/rovert)
        fR = 0.6
        gL = -0.4/Cxb/rovert
        gR = 0.0
        Cx = cubicspline(ptL3, ptR3, fL, fR, gL, gR, omega)

    else:
        Cx = 0.6




    if omega <= 1.7:
        Cx0 = 1.36 - 1.83/omega + 2.07/omega/omega
    elif omega > 0.5*rovert:
        Cx0 = max(0.6, 1 + 0.2/Cxb*(1-2.0*omega/rovert))
    else:
        Cx0 = 1.0

    cx0.append(Cx0)
    cx.append(Cx)

import matplotlib.pyplot as plt
plt.plot(omegavec, cx0)
plt.plot(omegavec, cx)
plt.show()

