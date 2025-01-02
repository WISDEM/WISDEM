import os
import numpy as np

class PreCompWriter:
    def __init__(self, dir_out, materials, upper, lower, webs, profile, chord, twist, p_le):
        self.dir_out = dir_out

        self.materials = materials
        self.upper = upper
        self.lower = lower
        self.webs = webs
        self.profile = profile

        self.chord = chord
        self.twist = twist
        self.p_le = p_le

        if not os.path.exists(dir_out):
            os.makedirs(dir_out)

    def execute(self):
        flist_layup = self.writePreCompLayup()
        flist_profile = self.writePreCompProfile()
        self.writePreCompMaterials()
        self.writePreCompInput(flist_layup, flist_profile)

    def writePreCompMaterials(self):
        text = []
        text.append("\n")
        text.append("Mat_Id     E1           E2          G12       Nu12     Density      Mat_Name\n")
        text.append(" (-)      (Pa)         (Pa)        (Pa)       (-)      (Kg/m^3)       (-)\n")

        for i, mat in enumerate(self.materials):
            text.append("%d %e %e %e %f %e %s\n" % (i + 1, mat.E1, mat.E2, mat.G12, mat.nu12, mat.rho, mat.name))

        fout = os.path.join(self.dir_out, "materials.inp")
        f = open(fout, "w")
        for outLine in text:
            f.write(outLine)
        f.close()

    def writePreCompLayup(self):
        f_out = []

        def write_layup_sectors(cs, web):
            text = []
            for i, (n_plies, t, theta, mat_idx) in enumerate(zip(cs.n_plies, cs.t, cs.theta, cs.mat_idx)):
                if web:
                    text.extend(["\n", "web_num    no of laminae (N_weblams)    Name of stack:\n"])
                else:
                    text.extend(
                        [
                            "...........................................................................\n",
                            "Sect_num    no of laminae (N_laminas)          STACK:\n",
                        ]
                    )
                n_lamina = len(n_plies)
                text.append("%d %d\n" % (i + 1, n_lamina))
                text.extend(
                    [
                        "\n",
                        "lamina    num of  thickness   fibers_direction  composite_material ID\n",
                        "number    plies   of ply (m)       (deg)               (-)\n",
                    ]
                )
                if web:
                    text.append("wlam_num N_Plies   w_tply       Tht_Wlam            Wmat_Id\n")
                else:
                    text.append("lam_num  N_plies    Tply         Tht_lam            Mat_id\n")

                for j, (plies_j, t_j, theta_j, mat_idx_j) in enumerate(zip(n_plies, t, theta, mat_idx + 1)):
                    text.append("%d %d %e %.1f %d\n" % (j + 1, plies_j, t_j, theta_j, mat_idx_j))
            return text

        for idx, (lower_i, upper_i, webs_i) in enumerate(zip(self.lower, self.upper, self.webs)):
            text = []
            text.append("Composite laminae lay-up inside the blade section\n")
            text.append("\n")
            text.append("*************************** TOP SURFACE ****************************\n")
            # number of sectors
            n_sector = len(upper_i.loc) - 1
            text.append("%d                N_scts(1):  no of sectors on top surface\n" % n_sector)
            text.extend(["\n", "normalized chord location of  nodes defining airfoil sectors boundaries (xsec_node)\n"])
            locU = upper_i.loc
            text.append(" ".join(["%f" % i for i in locU]) + "\n")
            text.extend(write_layup_sectors(upper_i, False))

            text.extend(["\n", "\n", "*************************** BOTTOM SURFACE ****************************\n"])
            n_sector = len(lower_i.loc) - 1
            text.append("%d                N_scts(1):  no of sectors on top surface\n" % n_sector)
            text.extend(["\n", "normalized chord location of  nodes defining airfoil sectors boundaries (xsec_node)\n"])
            locU = lower_i.loc
            text.append(" ".join(["%f" % i for i in locU]) + "\n")
            text.extend(write_layup_sectors(lower_i, False))

            text.extend(
                [
                    "\n",
                    "\n",
                    "**********************************************************************\n",
                    "Laminae schedule for webs (input required only if webs exist at this section):\n",
                ]
            )
            ########## Webs ##########
            text.extend(write_layup_sectors(webs_i, True))

            fname = os.path.join(self.dir_out, "layup_%00d.inp" % idx)
            f = open(fname, "w")
            for outLine in text:
                f.write(outLine)
            f.close()
            f_out.append(fname)

        return f_out

    def writePreCompProfile(self):
        f_out = []
        for idx, profile_i in enumerate(self.profile):
            # idx = 0
            # profile_i = profile[idx]
            text = []

            text.append(
                "%d                      N_af_nodes :no of airfoil nodes, counted clockwise starting\n"
                % len(profile_i.x)
            )
            text.append("                      with leading edge (see users' manual, fig xx)\n")
            text.append("\n")
            text.append(" Xnode      Ynode   !! chord-normalized coordinated of the airfoil nodes\n")

            x_all = np.concatenate((profile_i.x, np.flip(profile_i.x, 0)))
            y_all = np.concatenate((profile_i.yu, np.flip(profile_i.yl, 0)))

            if max(y_all) > 1.0:
                print(idx)

            # import matplotlib.pyplot as plt
            # plt.figure()
            # plt.plot(x_all, y_all)
            # plt.savefig('test.png')

            for x, y in zip(x_all, y_all):
                text.append("%f %f\n" % (x, y))

            fname = os.path.join(self.dir_out, "shape_%d.inp" % idx)
            f = open(fname, "w")
            for outLine in text:
                f.write(outLine)
            f.close()
            f_out.append(fname)

        return f_out

    def writePreCompInput(self, flist_layup, flist_profile):
        for idx in range(0, len(flist_layup)):
            chord = self.chord[idx]
            twist = self.twist[idx]
            p_le = self.p_le[idx]
            webs_i = self.webs[idx]
            layup_i = os.path.split(os.path.abspath(flist_layup[idx]))[1]
            profile_i = os.path.split(os.path.abspath(flist_profile[idx]))[1]

            text = []
            text.append("*****************  main input file for PreComp *****************************\n")
            text.append("Sample Composite Blade Section Properties\n")
            text.append("\n")
            text.append("General information -----------------------------------------------\n")
            text.append("1                Bl_length   : blade length (m)\n")
            text.append("2                N_sections  : no of blade sections (-)\n")
            text.append(
                "%d                N_materials : no of materials listed in the materials table (material.inp)\n"
                % len(self.materials)
            )
            text.append("3                Out_format  : output file   (1: general format, 2: BModes-format, 3: both)\n")
            text.append("f                TabDelim     (true: tab-delimited table; false: space-delimited table)\n")
            text.append("\n")
            text.append("Blade-sections-specific data --------------------------------------\n")
            text.append("Sec span     l.e.     chord   aerodynamic   af_shape    int str layup\n")
            text.append("location   position   length    twist         file          file\n")
            text.append("Span_loc    Le_loc    Chord    Tw_aero   Af_shape_file  Int_str_file\n")
            text.append("  (-)        (-)       (m)    (degrees)       (-)           (-)\n")
            text.append("\n")
            text.append("%.2f %f %e %f %s %s\n" % (0.0, p_le, chord, twist, profile_i, layup_i))
            text.append("%.2f %f %e %f %s %s\n" % (1.0, p_le, chord, twist, profile_i, layup_i))
            text.append("\n")
            text.append("Webs (spars) data  --------------------------------------------------\n")
            text.append("\n")
            text.append(
                "%d                Nweb        : number of webs (-)  ! enter 0 if the blade has no webs\n"
                % len(webs_i.loc)
            )
            text.append(
                "1                Ib_sp_stn   : blade station number where inner-most end of webs is located (-)\n"
            )
            text.append(
                "2                Ob_sp_stn   : blade station number where outer-most end of webs is located (-)\n"
            )
            text.append("\n")
            text.append("Web_num   Inb_end_ch_loc   Oub_end_ch_loc (fraction of chord length)\n")
            for i, loc in enumerate(webs_i.loc):
                text.append("%d %f %f\n" % (i + 1, loc, loc))

            fname = os.path.join(self.dir_out, "input_%d.inp" % idx)
            f = open(fname, "w")
            for outLine in text:
                f.write(outLine)
            f.close()

