import os
from collections import OrderedDict
import numpy as np
from wisdem.glue_code.runWISDEM import run_wisdem
from wisdem.commonse.mpi_tools import MPI
from weis.multifidelity.models.base_model import BaseModel
from scipy.interpolate import PchipInterpolator
import dill
from wisdem.ccblade import CCBlade as CCBladeOrig


## File management
run_dir = os.path.dirname(os.path.realpath(__file__)) + os.sep
fname_wt_input = run_dir + "../models/IEA-15-240-RWT_WISDEMieaontology4all.yaml"
fname_analysis_options_ccblade = run_dir + "../models/modeling_options_ccblade.yaml"
fname_analysis_options_openfast = run_dir + "../models/modeling_options_openfast.yaml"
fname_opt_options = run_dir + "../models/analysis_options.yaml"
folder_output = run_dir + "it_0/"
fname_wt_output = folder_output + "/temp.yaml"


class FullCCBlade(BaseModel):
    """
    Call the full WISDEM stack and focus on results from CCBlade.
    
    This calls WISDEM using the yaml files and all normal entry points, but
    has the additional overhead of unnecessary analyses.
    """

    def compute(self, desvars):
        wt_opt_ccblade, analysis_options_ccblade, opt_options_ccblade = run_wisdem(
            fname_wt_input,
            fname_analysis_options_ccblade,
            fname_opt_options,
            desvars,
        )

        outputs = {}
        outputs["CP"] = wt_opt_ccblade["ccblade.CP"][0]

        return outputs


class OpenFAST(BaseModel):
    """
    Call the full WISDEM stack and focus on results from OpenFAST.
    
    This calls WISDEM using the yaml files and all normal entry points, but
    has the additional overhead of unnecessary analyses. However, OpenFAST
    is generally more expensive than the other portions of WISDEM, making this
    less important.
    """

    def compute(self, desvars):
        wt_opt_openfast, analysis_options_openfast, opt_options_openfast = run_wisdem(
            fname_wt_input,
            fname_analysis_options_openfast,
            fname_opt_options,
            fname_wt_output,
            folder_output,
            desvars,
        )

        outputs = {}
        outputs["CP"] = wt_opt_openfast["aeroelastic.Cp"][0]

        return outputs


class CCBlade(BaseModel):
    """
    Call only CCBlade as a standalone function using saved inputs.
    
    To ensure we're running the correct geometry, you need to first run
    FullCCBlade to save off some pickle files with info needed for this model.
    However, this model is much faster than the full WISDEM version because it
    doesn't call other analyses unnecessarily. For a quick test, this is
    about 320x faster.
    """

    def __init__(self, desvars_init, warmstart_file=None, n_span=30):
        super().__init__(desvars_init, warmstart_file)
        self.n_span = n_span

    def compute(self, desvars):
        with open(run_dir + f"CCBlade_inputs_{self.n_span}.pkl", "rb") as f:
            saved_dict = dill.load(f)

        chord_opt_gain = desvars["blade.opt_var.chord_opt_gain"]

        chord_original = saved_dict["chord_original"]
        s = saved_dict["s"]
        s_opt_chord = np.linspace(0.0, 1.0, len(chord_opt_gain))

        spline = PchipInterpolator
        chord_spline = spline(s_opt_chord, chord_opt_gain)
        chord = chord_original * chord_spline(s)

        get_cp_cm = CCBladeOrig(
            saved_dict["r"],
            chord,
            saved_dict["twist"],
            saved_dict["af"],
            saved_dict["Rhub"],
            saved_dict["Rtip"],
            saved_dict["nBlades"],
            saved_dict["rho"],
            saved_dict["mu"],
            saved_dict["precone"],
            saved_dict["tilt"],
            saved_dict["yaw"],
            saved_dict["shearExp"],
            saved_dict["hub_height"],
            saved_dict["nSector"],
            saved_dict["precurve"],
            saved_dict["precurveTip"],
            saved_dict["presweep"],
            saved_dict["presweepTip"],
            saved_dict["tiploss"],
            saved_dict["hubloss"],
            saved_dict["wakerotation"],
            saved_dict["usecd"],
        )
        get_cp_cm.inverse_analysis = False
        get_cp_cm.induction = True

        # Compute omega given TSR
        Omega = (
            saved_dict["Uhub"] * saved_dict["tsr"] / saved_dict["Rtip"] * 30.0 / np.pi
        )

        myout, derivs = get_cp_cm.evaluate(
            [saved_dict["Uhub"]], [Omega], [saved_dict["pitch"]], coefficients=True
        )

        outputs = {}
        outputs["CP"] = myout["CP"]

        return outputs
