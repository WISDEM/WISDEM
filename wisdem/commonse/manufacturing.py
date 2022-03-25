import numpy as np
from wisdem.commonse import eps

# Compute costs based on "Optimum Design of Steel Structures" by Farkas and Jarmai
# All correlations are based in mm and all inputs are assumed to be in m and then converted within the functions


def steel_cutting_plasma_time(length, thickness):
    # Length input as meters, thickness in mm
    # time = length / (-0.180150943 + 41.03815215/(1e3*thickness+eps)) # minutes
    # Better conditioned polynomial fit to the above correlation
    pp = np.array([2.44908121e02, 1.74461814e01, 7.05214799e-02])
    time = length * np.polyval(pp, thickness)
    return np.sum(time)


def steel_rolling_time(theta, radius, thickness):
    # Radius and thickness input as meters, converted to mm
    time = theta * np.exp(
        6.8582513 - 4.527217 / np.sqrt(1e3 * thickness + eps) + 0.009541996 * np.sqrt(2 * 1e3 * radius)
    )
    return np.sum(time)


def steel_tube_cutgrind_time(theta, radius, thickness, angle):
    # Radius and thickness input as meters, converted to mm
    # time = theta * 2.5 * np.pi * (2.0*1e3*radius) / ((350.0 - 2.0*1e3*thickness)*0.3*np.sin(angle))
    # Better conditioned polynomial fit to the above correlation
    pp = np.array([7.84235859, 0.1428632, 0.07765389])
    time = theta * (2.0 * 1e3 * radius) / (np.polyval(pp, thickness) * np.sin(angle))
    return np.sum(time)


def steel_welding_time(theta, npieces, mtotal, length, thickness, coeff):
    # Length input as meters, thickness as mm
    time = np.sum(theta * np.sqrt(npieces * mtotal))
    time += np.sum(1.3e-3 * coeff * (length) * (1e3 * thickness) ** 1.9358)
    return time


def steel_butt_welding_time(theta, npieces, mtotal, length, thickness):
    return steel_welding_time(theta, npieces, mtotal, length, thickness, 0.152)


def steel_filett_welding_time(theta, npieces, mtotal, length, thickness):
    return steel_welding_time(theta, npieces, mtotal, length, thickness, 0.3394)


def steel_tube_welding_time(theta, npieces, mtotal, length, thickness):
    return steel_welding_time(theta, npieces, mtotal, length, thickness, 0.7889)
