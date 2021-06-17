import numpy as np


def is_floating(prob):
    return prob.model.options["modeling_options"]["flags"]["floating"]


def get_tower_diameter(prob):
    if is_floating(prob):
        return prob["floatingse.tower.outer_diameter"]
    else:
        return prob["towerse.tower_outer_diameter"]


def get_tower_thickness(prob):
    if is_floating(prob):
        return prob["floatingse.tower.wall_thickness"]
    else:
        return prob["towerse.tower_wall_thickness"]


def get_zpts(prob):
    if is_floating(prob):
        return prob["floatingse.tower.z_param"]
    else:
        return prob["towerse.z_param"]


def get_section_height(prob):
    return np.diff(get_zpts(prob))


def get_transition_height(prob):
    if is_floating(prob):
        return prob["floating.transition_node"][-1]
    else:
        return prob["towerse.transition_piece_height"]


def get_tower_E(prob):
    if is_floating(prob):
        return prob["floatingse.tower.E"]
    else:
        return prob["towerse.E"]


def get_tower_G(prob):
    if is_floating(prob):
        return prob["floatingse.tower.G"]
    else:
        return prob["towerse.G"]


def get_tower_rho(prob):
    if is_floating(prob):
        return prob["floatingse.tower.rho"]
    else:
        return prob["towerse.rho"]


def get_tower_mass(prob):
    if is_floating(prob):
        return prob["floatingse.tower_mass"]
    else:
        return prob["towerse.tower_mass"]


def get_tower_cost(prob):
    if is_floating(prob):
        return prob["floatingse.tower_cost"]
    else:
        return prob["towerse.tower_cost"]


def get_structural_mass(prob):
    if is_floating(prob):
        return prob["floatingse.tower.structural_mass"]
    else:
        return prob["towerse.structural_mass"]


def get_tower_freqs(prob):
    if is_floating(prob):
        return prob["floatingse.tower_freqs"]
    else:
        return prob["towerse.tower.structural_frequencies"]


def get_tower_cm(prob):
    if is_floating(prob):
        return prob["floatingse.tower_center_of_mass"]
    else:
        return prob["towerse.tower_center_of_mass"]


def get_tower_cg(prob):
    return get_tower_cm(prob)
