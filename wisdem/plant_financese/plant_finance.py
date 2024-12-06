import openmdao.api as om


class PlantFinance(om.ExplicitComponent):
    """
    Compute LCOE for the wind plant, formulas from https://doi.org/10.1016/j.tej.2021.106931

    Parameters
    ----------
    machine_rating : float
        Rating of the turbine
    tcc_per_kW : float
        Turbine capital cost
    offset_tcc_per_kW : float
        Offset to turbine capital cost
    turbine_number : int
        Number of turbines at plant
    bos_per_kW : float
        Balance of system costs of the turbine
    opex_per_kW : float
        Average annual operational expenditures of the turbine
    plant_aep_in : float
        Annual Energy Production of the wind plant
    turbine_aep : float
        Annual Energy Production of the wind turbine
    wake_loss_factor : float
        The losses in AEP due to waked conditions
    fixed_charge_rate : float
        Fixed charge rate for coe calculation
    electricity_price : float
        Electricity price
    reserve_margin_price : float
        Reserve margin price
    capacity_credit : float
        Capacity credit
    benchmark_price : float
        Benchmark price

    Returns
    -------
    plant_aep : float
        Annual Energy Production of the wind plant
    capacity_factor : float
        Capacity factor of the wind plant
    lcoe : float
        Levelized cost of energy for the wind plant
    lvoe : float
        Levelized value of energy
    value_factor : float
        Value factor
    nvoc : float
        Net value of capacity
    nvoe : float
        Net value of energy
    slcoe : float
        System LCOE
    bcr : float
        Benefit cost ratio
    cbr : float
        Cost benefit ratio
    roi : float
        Return of investment
    pm : float
        Profit margin
    plcoe : float
        Profitability adjusted LCOE
    """

    def initialize(self):
        self.options.declare("verbosity", default=False)

    def setup(self):
        # Inputs
        self.add_input("machine_rating", val=0.0, units="kW")
        self.add_input("tcc_per_kW", val=0.0, units="USD/kW")
        self.add_input("offset_tcc_per_kW", val=0.0, units="USD/kW")
        self.add_discrete_input("turbine_number", val=0)
        self.add_input("bos_per_kW", val=0.0, units="USD/kW")
        self.add_input("opex_per_kW", val=0.0, units="USD/kW/yr")
        self.add_input("plant_aep_in", val=0.0, units="kW*h")
        self.add_input("turbine_aep", val=0.0, units="kW*h")
        self.add_input("wake_loss_factor", val=0.15)
        self.add_input("fixed_charge_rate", val=0.075)
        self.add_input("electricity_price", val=0.04, units="USD/kW/h")
        self.add_input("reserve_margin_price", val=120.0, units="USD/kW/yr")
        self.add_input("capacity_credit", val=1.0)
        self.add_input("benchmark_price", val=0.071, units="USD/kW/h")

        self.add_output("plant_aep", val=0.0, units="USD/kW/h")
        self.add_output("capacity_factor", val=0.0, desc="Capacity factor of the wind farm")
        self.add_output(
            "lcoe",
            val=0.0,
            units="USD/kW/h",
            desc="Levelized cost of energy: LCOE is the cost that, if assigned to every unit of electricity by an asset over an evaluation period, will equal the total costs during that same period when discounted to the base year.",
        )
        self.add_output(
            "lvoe",
            val=0.0,
            units="USD/kW/h",
            desc="Levelized value of energy: LVOE is the discounted sum of total value divided by the discounted sum of electrical energy generated.",
        )
        self.add_output("value_factor", val=0.0, desc="Value factor is the LVOE divided by a benchmark price.")
        self.add_output(
            "nvoc",
            val=0.0,
            units="USD/kW/yr",
            desc="Net value of capacity: NVOC is the difference in an asset’s total annualized value and annualized cost, divided by the installed capacity of the asset. NVOC ≥ 0 for economic viability.",
        )
        self.add_output(
            "nvoe",
            val=0.0,
            units="USD/kW/h",
            desc="Net value of energy: NVOE is the difference between LVOE and LCOE. NVOE ≥ 0 for economic viability.",
        )
        self.add_output(
            "slcoe",
            val=0.0,
            units="USD/kW/h",
            desc="System LCOE: SLCOE is the negative of NVOE but further adjusted by a benchmark price. System LCOE ≤ benchmark price for economic viability.",
        )
        self.add_output(
            "bcr",
            val=0.0,
            desc="Benefit cost ratio: BCR is the discounted sum of total value divided by the discounted sum of total cost. A higher BCR is more competitive. BCR ≥ 1 for economic viability",
        )
        self.add_output(
            "cbr",
            val=0.0,
            desc="Cost benefit ratio: CBR is the inverse of BCR. CBR ≤ 1 for economic viability. A lower CBR is more competitive.",
        )
        self.add_output(
            "roi",
            val=0.0,
            desc="Return on investment: ROI can also be expressed as BCR – 1. A higher ROI is more competitive. ROI ≥ 0 for economic viability.",
        )
        self.add_output(
            "pm",
            val=0.0,
            desc="Profit margin: PM can also be expressed as 1 - CBR. A higher PM is more competitive. PM ≥ 0 for economic viability.",
        )
        self.add_output(
            "plcoe",
            val=0.0,
            units="USD/kW/h",
            desc="Profitability adjusted PLCOE is the product of a benchmark price and CBR, which is equal to LCOE divided by value factor. A lower PLCOE is more competitive. PLCOE ≤ benchmark price for economic viability.",
        )

        self.declare_partials("*", "*")

    def compute(self, inputs, outputs, discrete_inputs, discrete_outputs):
        # Unpack parameters
        t_rating = inputs["machine_rating"]
        n_turbine = discrete_inputs["turbine_number"]
        tcc_per_kW = inputs["tcc_per_kW"] + inputs["offset_tcc_per_kW"]
        bos_per_kW = inputs["bos_per_kW"]
        opex_per_kW = inputs["opex_per_kW"]
        fcr = inputs["fixed_charge_rate"]
        wlf = inputs["wake_loss_factor"]
        turb_aep = inputs["turbine_aep"]
        electricity_price = inputs["electricity_price"]
        reserve_margin_price = inputs["reserve_margin_price"]
        capacity_credit = inputs["capacity_credit"]
        benchmark_price = inputs["benchmark_price"]
        c_turbine = tcc_per_kW * t_rating
        c_bos_turbine = bos_per_kW * t_rating
        c_opex_turbine = opex_per_kW * t_rating

        # Run a few checks on the inputs
        if n_turbine == 0:
            raise ValueError(
                "ERROR: The number of the turbines in the plant is not initialized correctly and it is currently equal to 0. Check the connections to Plant_FinanceSE"
            )

        if c_turbine == 0:
            raise ValueError(
                "ERROR: The cost of the turbines in the plant is not initialized correctly and it is currently equal to 0 USD. Check the connections to Plant_FinanceSE"
            )

        if c_bos_turbine == 0:
            print(
                "WARNING: The BoS costs of the turbine are not initialized correctly and they are currently equal to 0 USD. Check the connections to Plant_FinanceSE"
            )

        if c_opex_turbine == 0:
            print(
                "WARNING: The Opex costs of the turbine are not initialized correctly and they are currently equal to 0 USD. Check the connections to Plant_FinanceSE"
            )

        if inputs["plant_aep_in"] == 0:
            if turb_aep == 0:
                print(
                    "WARNING: AEP is not computed properly or it is not connected to PlantFinanceSE. Both turbine_aep and park_aep are currently equal to 0 Wh. Check the connections to Plant_FinanceSE"
                )
            park_aep = n_turbine * turb_aep * (1.0 - wlf)
            dpark_dtaep = n_turbine * (1.0 - wlf)
            dpark_dnturb = turb_aep * (1.0 - wlf)
            dpark_dwlf = -n_turbine * turb_aep
            dpark_dpaep = 0.0
        else:
            park_aep = inputs["plant_aep_in"]
            dpark_dpaep = 1.0
            dpark_dtaep = dpark_dnturb = dpark_dwlf = 0.0

        npr = n_turbine * t_rating  # net park rating, used in net energy capture calculation below
        dnpr_dnturb = t_rating
        dnpr_dtrating = n_turbine

        nec = park_aep / npr  # net energy rating, per COE report
        dnec_dwlf = dpark_dwlf / npr
        dnec_dtaep = dpark_dtaep / npr
        dnec_dpaep = dpark_dpaep / npr
        dnec_dnturb = dpark_dnturb / npr - dnpr_dnturb * nec / npr
        dnec_dtrating = -dnpr_dtrating * nec / npr

        icc = (c_turbine + c_bos_turbine) / t_rating  # $/kW, changed per COE report
        c_opex = (c_opex_turbine) / t_rating  # $/kW, changed per COE report

        dicc_dtrating = -icc / t_rating
        dcopex_dtrating = -c_opex / t_rating
        dicc_dcturb = dicc_dcbos = dcopex_dcopex = 1.0 / t_rating

        C = icc * fcr + c_opex
        E = nec
        V = nec * electricity_price + reserve_margin_price * capacity_credit
        lcoe = C / E
        lvoe = V / E
        value_factor = lvoe / benchmark_price
        nvoc = V - C
        nvoe = lvoe - lcoe
        slcoe = benchmark_price - nvoe
        bcr = V / C
        cbr = C / V
        roi = (V - C) / C
        pm = (V - C) / V
        plcoe = C / V * benchmark_price

        # Assign openmdao outputs
        outputs["plant_aep"] = park_aep
        outputs["capacity_factor"] = nec / 8760.0
        outputs["lcoe"] = lcoe
        outputs["lvoe"] = lvoe
        outputs["value_factor"] = value_factor
        outputs["nvoc"] = nvoc
        outputs["nvoe"] = nvoe
        outputs["slcoe"] = slcoe
        outputs["bcr"] = bcr
        outputs["cbr"] = cbr
        outputs["roi"] = roi
        outputs["pm"] = pm
        outputs["plcoe"] = plcoe

        self.J = {}
        self.J["lcoe", "tcc_per_kW"] = dicc_dcturb * fcr / nec
        self.J["lcoe", "turbine_number"] = -dnec_dnturb * lcoe / nec
        self.J["lcoe", "bos_per_kW"] = dicc_dcbos * fcr / nec
        self.J["lcoe", "opex_per_kW"] = dcopex_dcopex / nec
        self.J["lcoe", "fixed_charge_rate"] = icc / nec
        self.J["lcoe", "wake_loss_factor"] = -dnec_dwlf * lcoe / nec
        self.J["lcoe", "turbine_aep"] = -dnec_dtaep * lcoe / nec
        self.J["lcoe", "plant_aep_in"] = -dnec_dpaep * lcoe / nec
        self.J["lcoe", "machine_rating"] = (dicc_dtrating * fcr + dcopex_dtrating) / nec - dnec_dtrating * lcoe / nec

    def compute_partials(self, inputs, J, discrete_inputs):
        J = self.J


# OpenMDAO group to execute the plant finance SE model as a standalone
class StandalonePlantFinanceSE(om.Group):
    def setup(self):
        self.add_subsystem("financese", PlantFinance(), promotes=["*"])


if __name__ == "__main__":
    wt_opt = om.Problem(model=StandalonePlantFinanceSE(), reports=False)
    wt_opt.setup(derivatives=False)
    wt_opt["machine_rating"] = 5.0e3
    wt_opt["tcc_per_kW"] = 1.5e3
    wt_opt["bos_per_kW"] = 446.0
    wt_opt["opex_per_kW"] = 43.0
    wt_opt["turbine_aep"] = 25.0e6
    wt_opt["fixed_charge_rate"] = 0.065
    wt_opt["turbine_number"] = 120
    wt_opt.run_model()

    print("plant_aep ", wt_opt["plant_aep"])
    print("lcoe ", wt_opt["lcoe"])
    print("lvoe ", wt_opt["lvoe"])
    print("value_factor ", wt_opt["value_factor"])
    print("nvoc ", wt_opt["nvoc"])
    print("nvoe ", wt_opt["nvoe"])
    print("slcoe ", wt_opt["slcoe"])
    print("bcr ", wt_opt["bcr"])
    print("cbr ", wt_opt["cbr"])
    print("roi ", wt_opt["roi"])
    print("pm ", wt_opt["pm"])
    print("plcoe ", wt_opt["plcoe"])
