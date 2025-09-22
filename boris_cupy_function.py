import cupy as cp
# from line_profiler import profile

# @profile
def boris_c_cupy(N_sub_steps, Dtt,
                 B_field, B_skew,
                 xn1, yn1, zn1,
                 vxn1, vyn1, vzn1,
                 Ex_n, Ey_n,
                 Bx_n_custom, By_n_custom, Bz_n_custom,
                 custom_B,
                 N_mp, N_multipoles,
                 charge, mass):

    qm = charge / mass
    Dtt_qm = Dtt * qm

    # Convert all input arrays to CuPy if they aren't already
    # xn1 = cp.asarray(xn1)
    # yn1 = cp.asarray(yn1)
    # zn1 = cp.asarray(zn1)

    # vxn1 = cp.asarray(vxn1)
    # vyn1 = cp.asarray(vyn1)
    # vzn1 = cp.asarray(vzn1)

    # Ex_n = cp.asarray(Ex_n)
    # Ey_n = cp.asarray(Ey_n)

    # B_field = cp.asarray(B_field)
    # B_skew = cp.asarray(B_skew)

    # if custom_B:
    #     Bx_n_custom = cp.asarray(Bx_n_custom)
    #     By_n_custom = cp.asarray(By_n_custom)
    #     Bz_n_custom = cp.asarray(Bz_n_custom)

    for _ in range(N_sub_steps):
        rexy = cp.ones(N_mp)
        imxy = cp.zeros(N_mp)

        By_n = cp.full(N_mp, B_field[0])
        Bx_n = cp.full(N_mp, B_skew[0])
        Bz_n = cp.zeros(N_mp)

        for order in range(1, N_multipoles):
            rexy_0 = rexy
            rexy = rexy_0 * xn1 - imxy * yn1
            imxy = imxy * xn1 + rexy_0 * yn1

            By_n += B_field[order] * rexy - B_skew[order] * imxy
            Bx_n += B_field[order] * imxy + B_skew[order] * rexy

        if custom_B:
            Bx_n += Bx_n_custom
            By_n += By_n_custom
            Bz_n += Bz_n_custom

        # Calculate tB and sB terms
        tBx = 0.5 * Dtt_qm * Bx_n
        tBy = 0.5 * Dtt_qm * By_n
        tBz = 0.5 * Dtt_qm * Bz_n
        tBsq = tBx**2 + tBy**2 + tBz**2

        sBx = 2 * tBx / (1 + tBsq)
        sBy = 2 * tBy / (1 + tBsq)
        sBz = 2 * tBz / (1 + tBsq)

        # v_minus
        vx_minus = vxn1 + 0.5 * Dtt_qm * Ex_n
        vy_minus = vyn1 + 0.5 * Dtt_qm * Ey_n
        vz_minus = vzn1

        # v_prime = v_minus + v_minus x tB
        vx_prime = vy_minus * tBz - vz_minus * tBy + vx_minus
        vy_prime = vz_minus * tBx - vx_minus * tBz + vy_minus
        vz_prime = vx_minus * tBy - vy_minus * tBx + vz_minus

        # v_plus = v_minus + v_prime x sB
        vx_plus = vy_prime * sBz - vz_prime * sBy + vx_minus
        vy_plus = vz_prime * sBx - vx_prime * sBz + vy_minus
        vz_plus = vx_prime * sBy - vy_prime * sBx + vz_minus

        # Final update
        vxn1 = vx_plus + 0.5 * Dtt_qm * Ex_n
        vyn1 = vy_plus + 0.5 * Dtt_qm * Ey_n
        vzn1 = vz_plus

        xn1 += vxn1 * Dtt
        yn1 += vyn1 * Dtt
        zn1 += vzn1 * Dtt

    return xn1, yn1, zn1, vxn1, vyn1, vzn1
