import numpy as np


def is_floating(prob):
    return prob.model.options["modeling_options"]["flags"]["floating"]


def is_monopile(prob):
    return prob.model.options["modeling_options"]["flags"]["monopile"]


def get_tower_diameter(prob):
    if is_monopile(prob):
        return np.r_[prob["fixedse.monopile_outer_diameter"], prob["towerse.tower_outer_diameter"][1:]]
    else:
        return prob["towerse.tower_outer_diameter"]


def get_tower_thickness(prob):
    if is_monopile(prob):
        return np.r_[prob["fixedse.monopile_wall_thickness"], prob["towerse.tower_wall_thickness"]]
    else:
        return prob["towerse.tower_wall_thickness"]


def get_zpts(prob):
    if is_monopile(prob):
        return np.r_[prob["fixedse.z_param"], prob["towerse.z_param"][1:]]
    else:
        return prob["towerse.z_param"]


def get_section_height(prob):
    return np.diff(get_zpts(prob))


def get_transition_height(prob):
    return prob["towerse.foundation_height"]


def get_tower_E(prob):
    if is_monopile(prob):
        return np.r_[prob["fixedse.member.E"], prob["towerse.member.E"][1:]]
    else:
        return prob["towerse.member.E"]


def get_tower_G(prob):
    if is_monopile(prob):
        return np.r_[prob["fixedse.member.G"], prob["towerse.member.G"][1:]]
    else:
        return prob["towerse.member.G"]


def get_tower_rho(prob):
    if is_monopile(prob):
        return np.r_[prob["fixedse.member.rho"], prob["towerse.member.rho"][1:]]
    else:
        return prob["towerse.member.rho"]


def get_tower_mass(prob):
    return prob["towerse.tower_mass"]


def get_tower_cost(prob):
    return prob["towerse.tower_cost"]


def get_monopile_mass(prob):
    return prob["fixedse.monopile_mass"]


def get_monopile_cost(prob):
    return prob["fixedse.monopile_cost"]


def get_structural_mass(prob):
    if is_monopile(prob):
        return prob["fixedse.structural_mass"]
    else:
        return prob["towerse.tower_mass"]


def get_structural_cost(prob):
    if is_monopile(prob):
        return prob["fixedse.structural_cost"]
    else:
        return prob["towerse.tower_cost"]


def get_tower_freqs(prob):
    if is_monopile(prob):
        return np.r_[prob["fixedse.monopile.structural_frequencies"], prob["towerse.tower.structural_frequencies"]]
    else:
        return prob["towerse.tower.structural_frequencies"]


def get_tower_cm(prob):
    return prob["towerse.tower_center_of_mass"]


def get_tower_cg(prob):
    return get_tower_cm(prob)
