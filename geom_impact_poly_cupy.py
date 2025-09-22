import cupy as cp

def impact_point_and_normal(x_in, y_in, z_in,
                                  x_out, y_out, z_out,
                                  Vx, Vy, Nx, Ny, 
                                  resc_fac):
    N_impacts = x_in.shape[0]

    # Output arrays
    x_int = cp.zeros(N_impacts, dtype=cp.float64)
    y_int = cp.zeros(N_impacts, dtype=cp.float64)
    z_int = cp.zeros(N_impacts, dtype=cp.float64)
    Nx_int = cp.zeros(N_impacts, dtype=cp.float64)
    Ny_int = cp.zeros(N_impacts, dtype=cp.float64)
    i_found = cp.full(N_impacts, -1, dtype=cp.int32)

    # Segment vectors for each edge
    Vx0 = Vx[:-1]
    Vx1 = Vx[1:]
    Vy0 = Vy[:-1]
    Vy1 = Vy[1:]
    # Nx_arr = Nx[:-1]  # assume per-edge normals (length N_edg)
    # Ny_arr = Ny[:-1]
    Nx_arr = Nx
    Ny_arr = Ny
    # print(Vx.shape, Vy.shape, Nx.shape, Ny.shape, Vx0.shape, Vy0.shape, Vx1.shape, Vy1.shape)

    # Broadcast inputs for all (impacts x edges)
    xi = x_in[:, None]
    yi = y_in[:, None]
    xo = x_out[:, None]
    yo = y_out[:, None]
    dx = xo - xi
    dy = yo - yi

    den = (dy * (Vx1 - Vx0) + dx * (Vy1 - Vy0))

    # Avoid division by zero (parallel case)
    den_zero = den == 0.0
    t_border = cp.where(den_zero, -2.0,
                        ((dy * (xi - Vx0)) + dx * (yi - Vy0)) / den)

    # Valid intersections
    valid_border = (t_border >= 0.0) & (t_border <= 1.0)

    # Compute t_ii only for valid borders
    # print(Nx_arr.shape)
    num = Nx_arr * (Vx0 - xi) + Ny_arr * (Vy0 - yi)
    denom = Nx_arr * dx + Ny_arr * dy
    
    t_ii = cp.where(denom != 0.0, num / denom, cp.inf)

    t_ii = cp.where(valid_border, t_ii, cp.inf)

    # Find min t_ii and corresponding edge
    t_min_curr = t_ii.min(axis=1)
    i_found_curr = t_ii.argmin(axis=1)

    # Compute intersection point
    t_scaled = resc_fac * t_min_curr
    x_int = t_scaled * x_out + (1.0 - t_scaled) * x_in
    y_int = t_scaled * y_out + (1.0 - t_scaled) * y_in
    z_int = cp.zeros_like(x_int)

    # Apply normals where valid
    valid_mask = cp.isfinite(t_min_curr)
    Nx_int[valid_mask] = Nx[i_found_curr[valid_mask]]
    Ny_int[valid_mask] = Ny[i_found_curr[valid_mask]]
    i_found[valid_mask] = i_found_curr[valid_mask]

    return x_int, y_int, z_int, Nx_int, Ny_int, i_found

import cupy as cp

def is_outside_convex(x_mp, y_mp, Vx, Vy, cx, cy):
    """
    Vectorized GPU implementation of convex polygon point-outside test using CuPy.
    """
    x_mp = cp.asarray(x_mp)
    y_mp = cp.asarray(y_mp)
    Vx = cp.asarray(Vx)
    Vy = cp.asarray(Vy)

    N_mp = x_mp.shape[0]

    # First, check ellipse containment
    is_inside_ellipse = (x_mp / cx) ** 2 + (y_mp / cy) ** 2 <= 1.0

    # Create edge vectors
    Vx0 = Vx[:-1]
    Vy0 = Vy[:-1]
    Vx1 = Vx[1:]
    Vy1 = Vy[1:]

    # Compute edge vectors
    edge_dx = Vx1 - Vx0
    edge_dy = Vy1 - Vy0

    # Broadcast points (N_mp, 1) and vertices (1, N_edg)
    px = x_mp[:, None]
    py = y_mp[:, None]

    dx = px - Vx0
    dy = py - Vy0

    # Compute cross product for each edge test (half-plane test)
    cross = dy * edge_dx - dx * edge_dy

    # Inside polygon if all cross products > 0 (i.e., left of all edges)
    is_inside_poly = cp.all(cross > 0.0, axis=1)

    # Apply logic: inside polygon OR inside ellipse ⇒ not outside
    is_outside = ~(is_inside_poly | is_inside_ellipse)

    return is_outside
