import numpy as np
import openmdao.api as om
from weis.multifidelity.models.base_model import BaseModel


A = 0.5
B = 10.0
C = -5.0


def simple_1D_high(x):
    term1 = A * (6 * x - 2) ** 2 * np.sin(12 * x - 4)
    term2 = B * (x - 0.5)
    return term1 + term2 + C


def simple_1D_medium(x):
    return (6 * x - 2) ** 2 * np.sin(12 * x - 4) + B * (x - 0.5)


def simple_1D_low(x):
    return (6 * x - 2) ** 2 * np.sin(12 * x - 4)


def simple_2D_high(x):
    term1 = A * (6 * x[1] - 2) ** 2
    term2 = B * (x[0] - 0.5)
    return term1 + term2 + C


def simple_2D_low(x):
    term1 = A * (4 * x[1] - 1) ** 2
    term2 = B * x[0]
    return term1 + term2
    
    
class simple_1D_high_model(BaseModel):
    def compute(self, desvars):
        outputs = {}
        outputs["y"] = simple_1D_high(desvars["x"])
        return outputs


class simple_1D_low_model(BaseModel):
    def compute(self, desvars):
        outputs = {}
        outputs["y"] = simple_1D_low(desvars["x"])
        return outputs


class simple_2D_high_model(BaseModel):
    def compute(self, desvars):
        outputs = {}
        outputs["y"] = simple_2D_high(desvars["x"])
        outputs["con"] = np.sum(desvars["x"])
        return outputs


class simple_2D_low_model(BaseModel):
    def compute(self, desvars):
        outputs = {}
        outputs["y"] = simple_2D_low(desvars["x"])
        outputs["con"] = np.sum(desvars["x"]) + desvars["x"][1] ** 2 * outputs["y"]
        return outputs
        
        
class simple_two_variable(BaseModel):
    def compute(self, desvars):
        outputs = {}
        outputs["y"] = np.sum(desvars["x"]**2.1) + desvars["z"]**0.6
        return outputs


class model_with_nd_outputs(BaseModel):
    def compute(self, desvars):
        outputs = {}
        outputs["y"] = np.diag(desvars["x"] ** 2)
        return outputs


class barnes_high_model(BaseModel):
    def compute(self, desvars):
        outputs = {}
        x1 = desvars["x"][0]
        x2 = desvars["x"][1]

        outputs["y"] = (
            -80
            + 3.81 * x1
            - 0.126 * x1 ** 2
            + 6.83 * x2
            - 0.0302 * x1 * x2
            + 1.281e-3 * x2 * x1 ** 2
        )

        outputs["c1"] = -(x1 * x2 / 700 - 1)
        outputs["c2"] = -(x2 / 5 - x1 ** 2 / 625)
        outputs["c3"] = (x2 / 50 - 1) ** 2 - (x1 / 500 - 0.11)

        return outputs


class barnes_low_model(BaseModel):
    def compute(self, desvars):
        outputs = {}
        x1 = desvars["x"][0]
        x2 = desvars["x"][1]

        outputs["y"] = (
            -80
            + 5.1 * x1
            - 0.126 * x1 ** 3
            + 6.83 * x2 ** 2
            - 0.0302 * x1 * x2
            + 7e-3 * x2 * x1 ** 2
        )

        outputs["c1"] = (-x1 - x2 + 50) / 10
        outputs["c2"] = (0.64 * x1 - x2) / 6
        outputs["c3"] = (x2 / 50 - 1) - (x1 / 500 - 0.11)

        return outputs


class eggcrate_high_model(BaseModel):
    def compute(self, desvars):
        outputs = {}
        x1 = desvars["x"][0]
        x2 = desvars["x"][1]

        outputs["y"] = x1 ** 2 + x2 ** 2 + 5 * (np.sin(x1) ** 2 + np.sin(x2) ** 2)

        return outputs


class eggcrate_low_model(BaseModel):
    def compute(self, desvars):
        outputs = {}
        x1 = desvars["x"][0]
        x2 = desvars["x"][1]

        outputs["y"] = (
            (x1 - 1) ** 2 + (x2 + 2) ** 2 + 5 * (np.sin(x1) ** 2 + np.sin(x2) ** 2)
        )

        return outputs
