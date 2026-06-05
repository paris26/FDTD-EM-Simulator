from utils import init_debye_grid

grid = init_debye_grid(size, sigma, dt, eps_inf, eps_0, n_0, s_c, nt, ed)
ez = grid.ez
hy = grid.hy
jp = grid.jp
ca = grid.ca
cb = grid.cb
cjj = grid.cjj
cje = grid.cje


for i in range(maxTime):
    for mm in range(SIZE - 1):
        hy[mm] = hy[mm] + (ez[mm + 1] - ez[mm]) / imp0

    for mm in range(SIZE):
        old_ez = ez[mm]

        ez[mm] = ca[mm] * ez[mm] + cb[mm] * (
            (hy[mm] - hy[mm - 1]) - 0.5 * (1 + cjj[mm]) * jp[mm]
        )

        jp[mm] = cjj[mm] * jp[mm] + cje[mm] * (ez[mm] - old_ez)
