import numpy as np
from mayavi import mlab


def sectional2nodal(x):
    return np.r_[x[0], np.convolve(x, [0.5, 0.5], "valid"), x[-1]]


def nodal2sectional(x):
    return 0.5 * (x[:-1] + x[1:])


def set_axes_equal(ax):
    """Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc..  This is one possible solution to Matplotlib's
    ax.set_aspect('equal') and ax.axis('equal') not working for 3D.

    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
    """

    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5 * max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])


class Visualize(object):
    def __init__(self, prob):
        prob.run_model()
        self.prob = prob
        self.fig = None

    def draw_spar(self, fname="spar.png"):
        self.init_figure()

        self.draw_ocean()

        self.draw_mooring(self.prob["mooring_plot_matrix"])

        zcut = 1.0 + self.prob["main_freeboard"]
        self.draw_pontoons(self.prob["plot_matrix"], 0.5 * self.prob["fairlead_support_outer_diameter"], zcut)

        self.draw_column(
            [0.0, 0.0],
            self.prob["main_freeboard"],
            self.prob["main.section_height"],
            0.5 * self.prob["main.outer_diameter"],
            self.prob["main.stiffener_spacing"],
        )

        t_full = sectional2nodal(self.prob["main.wall_thickness"])
        self.draw_ballast(
            [0.0, 0.0],
            self.prob["main_freeboard"],
            self.prob["main.section_height"],
            0.5 * self.prob["main.outer_diameter"] - t_full,
            self.prob["main.permanent_ballast_height"],
            self.prob["variable_ballast_height"],
        )

        self.draw_column(
            [0.0, 0.0],
            self.prob["hub_height"],
            self.prob["tow.tower_section_height"],
            0.5 * self.prob["tow.tower_outer_diameter"],
            None,
            (0.9,) * 3,
        )

        if self.prob["main.buoyancy_tank_mass"] > 0.0:
            self.draw_buoyancy_tank(
                [0.0, 0.0],
                self.prob["main_freeboard"],
                self.prob["main.section_height"],
                self.prob["main.buoyancy_tank_location"],
                0.5 * self.prob["main.buoyancy_tank_diameter"],
                self.prob["main.buoyancy_tank_height"],
            )

        self.set_figure(fname)

    def draw_semi(self, fname="semi.png"):
        self.init_figure()

        self.draw_ocean()

        self.draw_mooring(self.prob["mooring_plot_matrix"])

        pontoonMat = self.prob["plot_matrix"]
        zcut = 1.0 + np.maximum(self.prob["main_freeboard"], self.prob["offset_freeboard"])
        self.draw_pontoons(pontoonMat, 0.5 * self.prob["pontoon_outer_diameter"], zcut)

        self.draw_column(
            [0.0, 0.0],
            self.prob["main_freeboard"],
            self.prob["main.section_height"],
            0.5 * self.prob["main.outer_diameter"],
            self.prob["main.stiffener_spacing"],
        )

        t_full = sectional2nodal(self.prob["main.wall_thickness"])
        self.draw_ballast(
            [0.0, 0.0],
            self.prob["main_freeboard"],
            self.prob["main.section_height"],
            0.5 * self.prob["main.outer_diameter"] - t_full,
            self.prob["main.permanent_ballast_height"],
            self.prob["variable_ballast_height"],
        )

        if self.prob["main.buoyancy_tank_mass"] > 0.0:
            self.draw_buoyancy_tank(
                [0.0, 0.0],
                self.prob["main_freeboard"],
                self.prob["main.section_height"],
                self.prob["main.buoyancy_tank_location"],
                0.5 * self.prob["main.buoyancy_tank_diameter"],
                self.prob["main.buoyancy_tank_height"],
            )

        R_semi = self.prob["radius_to_offset_column"]
        ncolumn = int(self.prob["number_of_offset_columns"])
        angles = np.linspace(0, 2 * np.pi, ncolumn + 1)
        x = R_semi * np.cos(angles)
        y = R_semi * np.sin(angles)
        for k in range(ncolumn):
            self.draw_column(
                [x[k], y[k]],
                self.prob["offset_freeboard"],
                self.prob["off.section_height"],
                0.5 * self.prob["off.outer_diameter"],
                self.prob["off.stiffener_spacing"],
            )

            t_full = sectional2nodal(self.prob["off.wall_thickness"])
            self.draw_ballast(
                [x[k], y[k]],
                self.prob["offset_freeboard"],
                self.prob["off.section_height"],
                0.5 * self.prob["off.outer_diameter"] - t_full,
                self.prob["off.permanent_ballast_height"],
                0.0,
            )

            if self.prob["off.buoyancy_tank_mass"] > 0.0:
                self.draw_buoyancy_tank(
                    [x[k], y[k]],
                    self.prob["offset_freeboard"],
                    self.prob["off.section_height"],
                    self.prob["off.buoyancy_tank_location"],
                    0.5 * self.prob["off.buoyancy_tank_diameter"],
                    self.prob["off.buoyancy_tank_height"],
                )

        self.draw_column(
            [0.0, 0.0],
            self.prob["hub_height"],
            self.prob["tow.tower_section_height"],
            0.5 * self.prob["tow.tower_outer_diameter"],
            None,
            (0.9,) * 3,
        )

        self.set_figure(fname)

    def init_figure(self):
        mysky = np.array([135, 206, 250]) / 255.0
        mysky = tuple(mysky.tolist())
        # fig = plt.figure()
        # ax = fig.add_subplot(111, projection='3d')
        # fig = mlab.figure(bgcolor=(1,)*3, size=(1600,1100))
        # fig = mlab.figure(bgcolor=mysky, size=(1600,1100))
        self.fig = mlab.figure(bgcolor=(0,) * 3, size=(1600, 1100))

    def draw_ocean(self):
        if self.fig is None:
            self.init_figure()
        npts = 100

        # mybrown = np.array([244, 170, 66]) / 255.0
        # mybrown = tuple(mybrown.tolist())
        mywater = np.array([95, 158, 160]) / 255.0  # (0.0, 0.0, 0.8) [143, 188, 143]
        mywater = tuple(mywater.tolist())
        alpha = 0.3

        # Waterplane box
        x = y = 100 * np.linspace(-1, 1, npts)
        X, Y = np.meshgrid(x, y)
        Z = np.sin(100 * X * Y)  # np.zeros(X.shape)
        # ax.plot_surface(X, Y, Z, alpha=alpha, color=mywater)
        mlab.mesh(X, Y, Z, opacity=alpha, color=mywater, figure=self.fig)

        # Sea floor
        Z = -self.prob["water_depth"] * np.ones(X.shape)
        # ax.plot_surface(10*X, 10*Y, Z, alpha=1.0, color=mybrown)
        # mlab.mesh(10*X,10*Y,Z, opacity=1.0, color=mybrown, figure=self.fig)

        # Sides
        # x = 500 * np.linspace(-1, 1, npts)
        # z = self.prob['water_depth'] * np.linspace(-1, 0, npts)
        # X,Z = np.meshgrid(x,z)
        # Y = x.max()*np.ones(Z.shape)
        ##ax.plot_surface(X, Y, Z, alpha=alpha, color=mywater)
        # mlab.mesh(X,Y,Z, opacity=alpha, color=mywater, figure=self.fig)
        # mlab.mesh(X,-Y,Z, opacity=alpha, color=mywater, figure=self.fig)
        # mlab.mesh(Y,X,Z, opacity=alpha, color=mywater, figure=self.fig)
        ##mlab.mesh(-Y,X,Z, opacity=alpha, color=mywater, figure=self.fig)

    def draw_mooring(self, mooring):
        mybrown = np.array([244, 170, 66]) / 255.0
        mybrown = tuple(mybrown.tolist())
        npts = 100

        # Sea floor
        print(self.prob["anchor_radius"])
        r = np.linspace(0, self.prob["anchor_radius"], npts)
        th = np.linspace(0, 2 * np.pi, npts)
        R, TH = np.meshgrid(r, th)
        X = R * np.cos(TH)
        Y = R * np.sin(TH)
        Z = -self.prob["water_depth"] * np.ones(X.shape)
        # ax.plot_surface(X, Y, Z, alpha=1.0, color=mybrown)
        mlab.mesh(X, Y, Z, opacity=1.0, color=mybrown, figure=self.fig)
        cmoor = (0, 0.8, 0)
        nlines = int(self.prob["number_of_mooring_connections"] * self.prob["mooring_lines_per_connection"])
        for k in range(nlines):
            # ax.plot(mooring[k,:,0], mooring[k,:,1], mooring[k,:,2], 'k', lw=2)
            mlab.plot3d(
                mooring[k, :, 0],
                mooring[k, :, 1],
                mooring[k, :, 2],
                color=cmoor,
                tube_radius=0.5 * self.prob["mooring_diameter"],
                figure=self.fig,
            )

    def draw_pontoons(self, truss, R, freeboard):
        nE = truss.shape[0]
        c = (0.5, 0, 0)
        for k in range(nE):
            if np.any(truss[k, 2, :] > freeboard):
                continue
            mlab.plot3d(truss[k, 0, :], truss[k, 1, :], truss[k, 2, :], color=c, tube_radius=R, figure=self.fig)

    def draw_column(self, centerline, freeboard, h_section, r_nodes, spacingVec=None, ckIn=None):
        npts = 20

        nsection = h_section.size
        z_nodes = np.flipud(freeboard - np.r_[0.0, np.cumsum(np.flipud(h_section))])

        th = np.linspace(0, 2 * np.pi, npts)
        for k in range(nsection):
            rk = np.linspace(r_nodes[k], r_nodes[k + 1], npts)
            z = np.linspace(z_nodes[k], z_nodes[k + 1], npts)
            R, TH = np.meshgrid(rk, th)
            Z, _ = np.meshgrid(z, th)
            X = R * np.cos(TH) + centerline[0]
            Y = R * np.sin(TH) + centerline[1]

            # Draw parameters
            if ckIn is None:
                ck = (0.6,) * 3 if np.mod(k, 2) == 0 else (0.4,) * 3
            else:
                ck = ckIn
            # ax.plot_surface(X, Y, Z, alpha=0.5, color=ck)
            mlab.mesh(X, Y, Z, opacity=0.7, color=ck, figure=self.fig)

            if spacingVec is None:
                continue

            z = z_nodes[k] + spacingVec[k]
            while z < z_nodes[k + 1]:
                rk = np.interp(z, z_nodes[k:], r_nodes[k:])
                # ax.plot(rk*np.cos(th), rk*np.sin(th), z*np.ones(th.shape), 'r', lw=0.25)
                mlab.plot3d(
                    rk * np.cos(th) + centerline[0],
                    rk * np.sin(th) + centerline[1],
                    z * np.ones(th.shape),
                    color=(0.5, 0, 0),
                    figure=self.fig,
                )
                z += spacingVec[k]

                """
                # Web
                r   = np.linspace(rk - self.prob['stiffener_web_height'][k], rk, npts)
                R, TH = np.meshgrid(r, th)
                Z, _  = np.meshgrid(z, th)
                X = R*np.cos(TH)
                Y = R*np.sin(TH)
                ax.plot_surface(X, Y, Z, alpha=0.7, color='r')

                # Flange
                r = r[0]
                h = np.linspace(0, self.prob['stiffener_flange_width'][k], npts)
                zflange = z + h - 0.5*self.prob['stiffener_flange_width'][k]
                R, TH = np.meshgrid(r, th)
                Z, _  = np.meshgrid(zflange, th)
                X = R*np.cos(TH)
                Y = R*np.sin(TH)
                ax.plot_surface(X, Y, Z, alpha=0.7, color='r')
                """

    def draw_ballast(self, centerline, freeboard, h_section, r_nodes, h_perm, h_water):
        npts = 40
        th = np.linspace(0, 2 * np.pi, npts)
        z_nodes = np.flipud(freeboard - np.r_[0.0, np.cumsum(np.flipud(h_section))])

        # Permanent ballast
        z_perm = z_nodes[0] + np.linspace(0, h_perm, npts)
        r_perm = np.interp(z_perm, z_nodes, r_nodes)
        R, TH = np.meshgrid(r_perm, th)
        Z, _ = np.meshgrid(z_perm, th)
        X = R * np.cos(TH) + centerline[0]
        Y = R * np.sin(TH) + centerline[1]
        ck = np.array([122, 85, 33]) / 255.0
        ck = tuple(ck.tolist())
        mlab.mesh(X, Y, Z, color=ck, figure=self.fig)

        # Water ballast
        z_water = z_perm[-1] + np.linspace(0, h_water, npts)
        r_water = np.interp(z_water, z_nodes, r_nodes)
        R, TH = np.meshgrid(r_water, th)
        Z, _ = np.meshgrid(z_water, th)
        X = R * np.cos(TH) + centerline[0]
        Y = R * np.sin(TH) + centerline[1]
        ck = (0.0, 0.1, 0.8)  # Dark blue
        mlab.mesh(X, Y, Z, color=ck, figure=self.fig)

    def draw_buoyancy_tank(self, centerline, freeboard, h_section, loc, r_box, h_box):
        npts = 20
        z_nodes = np.flipud(freeboard - np.r_[0.0, np.cumsum(np.flipud(h_section))])
        z_lower = loc * (z_nodes[-1] - z_nodes[0]) + z_nodes[0]

        # Lower and Upper surfaces
        r = np.linspace(0, r_box, npts)
        th = np.linspace(0, 2 * np.pi, npts)
        R, TH = np.meshgrid(r, th)
        X = R * np.cos(TH) + centerline[0]
        Y = R * np.sin(TH) + centerline[1]
        Z = z_lower * np.ones(X.shape)
        ck = (0.9,) * 3
        mlab.mesh(X, Y, Z, opacity=0.5, color=ck, figure=self.fig)
        Z += h_box
        mlab.mesh(X, Y, Z, opacity=0.5, color=ck, figure=self.fig)

        # Cylinder part
        z = z_lower + np.linspace(0, h_box, npts)
        Z, TH = np.meshgrid(z, th)
        R = r_box * np.ones(Z.shape)
        X = R * np.cos(TH) + centerline[0]
        Y = R * np.sin(TH) + centerline[1]
        mlab.mesh(X, Y, Z, opacity=0.5, color=ck, figure=self.fig)

    def set_figure(self, fname=None):
        # ax.set_aspect('equal')
        # set_axes_equal(ax)
        # ax.autoscale_view(tight=True)
        # ax.set_xlim([-125, 125])
        # ax.set_ylim([-125, 125])
        # ax.set_zlim([-220, 30])
        # plt.axis('off')
        # plt.show()
        # mlab.move([-517.16728532, -87.0711504, 5.60826224], [1.35691603e+01, -2.84217094e-14, -1.06547500e+02])
        # mlab.view(-170.68320804213343, 78.220729198686854, 549.40101471336777, [1.35691603e+01,  0.0, -1.06547500e+02])
        if not fname is None:
            fpart = fname.split(".")
            if len(fpart) == 1 or not fpart[-1].lower() in ["jpg", "png", "bmp"]:
                fname += ".png"
            mlab.savefig(fname, figure=self.fig)
        mlab.show()
