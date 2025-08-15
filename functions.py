import numpy as np
import matplotlib.pyplot as plt
from random import uniform
import json
import os 
from tqdm import tqdm
import cupy as cp  
import re
import pickle
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm
import matplotlib.colors as mcolors
'''
                Mantina M, Chamberlin AC, Valero R, Cramer CJ, Truhlar DG. 
                Consistent van der Waals radii for the whole main group.
                J Phys Chem A. 2009 May 14;113(19):5806-12.
                doi: 10.1021/jp8111556. PMID: 19382751; PMCID: PMC3658832.
                '''
bondi_radii =  { 
                
                
                "C" : 0.17, 
                "N": 0.155,
                "O": 0.152,
                "F": 0.147,
                "Si": 0.210,
                "P": 0.180,
                "S": 0.180, 
                "Cl": 0.175 ,
                "Br":0.183 ,
                "H": 0.110,
                
}




def transcribe_gro_radius(gro_file_path, crop_edges=False, threshold=0.3, nbins=100):
    import re
    import numpy as np
    
    # You need this globally defined somewhere
    # bondi_radii = {"C": 1.7, "H": 1.2, "O": 1.52, "N": 1.55, ...}

    with open(gro_file_path, "r") as gro_file:
        lines = gro_file.readlines()

    x_coords = []
    y_coords = []
    z_coords = []
    names = []
    full_name = []
    i = 0

    for line in lines[1:]:
        columns = line.split()
        if len(columns) >= 6:
            names.append(re.sub(r'\d+', '', columns[1]))
            full_name.append(str(columns[1]))
            x_coords.append(float(columns[3]))
            y_coords.append(float(columns[4]))
            z_coords.append(float(columns[5]))
            i += 1
        elif len(columns) == 5 and columns[0] != 'wetting':
            names.append(re.sub(r'\d+', '', columns[1]))
            full_name.append(str(columns[1]))
            x_coords.append(float(columns[2]))
            y_coords.append(float(columns[3]))
            z_coords.append(float(columns[4]))
            i += 1

    # Initial length before any cropping
    print("Initial number of atoms:", len(names))
    max_xT = max(x_coords)
    max_yT = max(y_coords)
    max_zT = max(z_coords)
    box_dT = np.array([max_xT, max_yT, max_zT])
    print(f"Box dimensions (original) (nm): {box_dT}")
    

    if crop_edges:
        print("Cropping based on z-slice density...")

        # Convert to arrays
        x_coords = np.array(x_coords)
        y_coords = np.array(y_coords)
        z_coords = np.array(z_coords)
        names = np.array(names)
        full_name = np.array(full_name)

        # Histogram along z-axis
        z_bins = np.linspace(z_coords.min(), z_coords.max(), nbins + 1)
        z_hist, _ = np.histogram(z_coords, bins=z_bins)

        print("Z bin counts:", z_hist)

        # Normalize
        z_density = z_hist / np.max(z_hist)

        # Determine dense slab limits
        def get_bounds(density, bins):
            indices = np.where(density >= threshold)[0]
            return bins[indices[0]], bins[indices[-1] + 1]

        z_min, z_max = get_bounds(z_density, z_bins)

        print(f"Z cropping range: {z_min:.2f} – {z_max:.2f}")

        # Mask to keep atoms within dense z region
        mask = (z_coords > z_min) & (z_coords < z_max)

        # Apply mask to all arrays
        x_coords = x_coords[mask]
        y_coords = y_coords[mask]
        z_coords = z_coords[mask]
        names = names[mask]
        full_name = full_name[mask]

        print("Remaining atoms after z cropping:", len(names))

    # Radii assignment
    radius_list = [bondi_radii[atom_name] for atom_name in names]
    print("Length of Radius list:", len(radius_list))

    # Bounding box based on remaining atoms
    max_x = max(x_coords)
    max_y = max(y_coords)
    max_z = max(z_coords)
    min_x = min(x_coords)
    min_y = min(y_coords)
    min_z = min(z_coords)

    box_d = np.array([max_x, max_y, max_z])
    print(f"Box dimensions (final) (nm): {box_d}")

    # Atom dictionary output
    data = []
    for i in range(len(full_name)):
        dictionary = {
            'name': full_name[i],
            'x': x_coords[i],
            'y': y_coords[i],
            'z': z_coords[i],
            'radius': radius_list[i]
        }
        data.append(dictionary)

    return data, box_d

def max_clear_radius_scores(centers, dots_data, np_to_use, batch_points=2000):
        """
        Score = distance to nearest dot surface (>=0). Batched to keep memory sane.
        """
        scores = np_to_use.zeros(centers.shape[0], dtype=float)
        dot_centers = dots_data[:, :3]
        dot_radii   = dots_data[:, 3]
        N = centers.shape[0]
        for start in range(0, N, batch_points):
            end = min(N, start + batch_points)
            c = centers[start:end]  # (B,3)
            d = np_to_use.linalg.norm(c[:, None, :] - dot_centers[None, :, :], axis=2)  # (B,Ndots)
            surf = d - dot_radii[None, :]
            m = np_to_use.min(surf, axis=1)
            m = np_to_use.where(m > 0, m, 0.0)
            scores[start:end] = m
        return scores

def VSD(data, number_grid_points = 10, box_dimensions_array = [6,6,6], step_size = 0.010, units = "nm",
        diameter=False, Gaussian_Fit = True, GPU =False, all_grid = True):
    
    if GPU:
        np_to_use = cp
    else:
        np_to_use = np
    
    print("Box dimensions (nm)")
    print(box_dimensions_array)
    dots_data = np_to_use.array([(dot["x"], dot["y"], dot["z"], dot["radius"]) for dot in data])
    box_dimensions = {'x': box_dimensions_array[0], 'y': box_dimensions_array[1], 'z': box_dimensions_array[2]}
    n_x = max(1, int(number_grid_points * box_dimensions['x']))
    n_y = max(1, int(number_grid_points * box_dimensions['y']))
    n_z = max(1, int(number_grid_points * box_dimensions['z']))

    cube_length_x = box_dimensions['x'] / n_x
    cube_length_y = box_dimensions['y'] / n_y
    cube_length_z = box_dimensions['z'] / n_z

    x = np_to_use.arange(0, box_dimensions['x'], cube_length_x) + cube_length_x / 2
    y = np_to_use.arange(0, box_dimensions['y'], cube_length_y) + cube_length_y / 2
    z = np_to_use.arange(0, box_dimensions['z'], cube_length_z) + cube_length_z / 2

    x = x[x < box_dimensions['x']]
    y = y[y < box_dimensions['y']]
    z = z[z < box_dimensions['z']]

    grid_x, grid_y, grid_z = np_to_use.meshgrid(x, y, z, indexing='ij')
    parent_centers = np_to_use.column_stack((grid_x.ravel(), grid_y.ravel(), grid_z.ravel()))
    uniform_centers = parent_centers.shape[0]

    if all_grid:
        grid_centers = parent_centers
        grid_cube_volume = cube_length_x * cube_length_y * cube_length_z
        total_calculations = grid_centers.shape[0] * len(data)
        print(f"Uniform centers (ngrid={number_grid_points}): {uniform_centers}")
        print(f"Parent cell size (nm): x:{cube_length_x:.4f}, y:{cube_length_y:.4f}, z:{cube_length_z:.4f}")
        print(f"Grid cube volume (nm^3): {grid_cube_volume:.6f}")
        print(f"Estimated pair checks (centers × dots): {total_calculations}")
    else:
        scores = max_clear_radius_scores(parent_centers, dots_data, np_to_use)
        interesting_percentile = 50
        cut = np_to_use.percentile(scores, interesting_percentile)
        interesting_idx = np_to_use.where(scores >= cut)[0]
        if interesting_idx.size > 0:
            grid_centers = parent_centers[interesting_idx]
        else:
            print("FALLLBACK ERROR")
            rng = np.random.default_rng(42)
            sample = min(max(1, parent_centers.shape[0] // 20), parent_centers.shape[0])
            keep_idx = rng.choice(parent_centers.shape[0], size=sample, replace=False)
            grid_centers = parent_centers[keep_idx]
        grid_cube_volume = cube_length_x * cube_length_y * cube_length_z
        total_calculations = grid_centers.shape[0] * len(data)
        reduction = 100.0 * (1.0 - grid_centers.shape[0] / max(1, uniform_centers))
        print(f"Uniform centers (ngrid={number_grid_points}): {uniform_centers}")
        print(f"Kept interesting centers (top {100 - interesting_percentile}%): {grid_centers.shape[0]}  (↓ {reduction:.1f}% vs uniform)")
        print(f"Parent cell size (nm): x:{cube_length_x:.4f}, y:{cube_length_y:.4f}, z:{cube_length_z:.4f}")
        print(f"Grid cube volume (nm^3): {grid_cube_volume:.6f}")
        print(f"Estimated pair checks (centers × dots): {total_calculations}")
        
    def sphere_overlaps(candidate_centers, candidate_radii, dots_data):
        dot_centers = dots_data[:, :3]
        dot_radii = dots_data[:, 3]
        distances = np_to_use.linalg.norm(candidate_centers[:, None] - dot_centers, axis=2)
        overlaps = distances < (candidate_radii[:, None] + dot_radii)
        return np_to_use.any(overlaps, axis=1)

    def overlapping_indices(new_point, new_radius, sphere_list):
        if not sphere_list:
            return np.array([], dtype=int)
        centers = np.stack([np.asarray(c, dtype=float) for c, _ in sphere_list])
        radii   = np.array([float(r) for _, r in sphere_list], dtype=float)
        d2   = np.sum((centers - np.asarray(new_point, dtype=float))**2, axis=1)
        thr2 = (radii + float(new_radius))**2
        return np.where(d2 < thr2)[0]

    inserted_spheres_radii = []
    radii_list_total = [0.0000, ]
    steo_ = step_size
    sphere_list = []

    for center in tqdm(grid_centers, desc="Processing grid centers"):
        radius_inserted_sphere = 0.010
        ci = 0
        while True:
            overlaps = sphere_overlaps(np_to_use.array([center]), np_to_use.array([radius_inserted_sphere]), dots_data)
            if overlaps.any():
                break
            # (optional) radii_list_total.append(radius_inserted_sphere)
            radius_inserted_sphere += steo_
            ci += 1

        # <<< CHANGED: use the last *valid* (non-overlapping) radius >>>
        last_valid_r = max(0.0, radius_inserted_sphere - steo_)   # maximal inscribed at this grid point
        if last_valid_r <= 0:
            continue

        # <<< CHANGED: store only maximal spheres per region >>>
        new_point = np_to_use.array(center)
        if sphere_list:
            idx = overlapping_indices(new_point, last_valid_r, sphere_list)
            if idx.size > 0:
                # If new sphere is larger than any overlapping stored ones, replace them
                max_old = max(sphere_list[i][1] for i in idx)
                if last_valid_r > max_old:
                    drop = set(int(i) for i in idx)
                    sphere_list = [s for j, s in enumerate(sphere_list) if j not in drop]
                    sphere_list.append((np.asarray(new_point, dtype=float), float(last_valid_r)))
                # else: discard (smaller than an existing maximal sphere)
            else:
                sphere_list.append((np.asarray(new_point, dtype=float), float(last_valid_r)))
        else:
            sphere_list.append((np.asarray(new_point, dtype=float), float(last_valid_r)))
    # -------------------------------------------------------------------------

    print(f"# maximal spheres kept: {len(sphere_list)}")

    # ----------------- Regular binning (avoid np.unique bias) ----------------
    # Plot the histogram of the radii of inserted spheres
    inserted_spheres_radii = [radius for _, radius in sphere_list]
    unique_radii, counts = np_to_use.unique(inserted_spheres_radii, return_counts=True)

    # Pad missing bins before first radius
    other_radii = np_to_use.arange(0, unique_radii[0], steo_)
    unique_radii = np_to_use.concatenate((other_radii[:-1], unique_radii))
    zero_array = np_to_use.zeros_like(other_radii[:-1])
    counts = np_to_use.concatenate((zero_array, counts))
  
    # --- volume-weight the histogram: each accepted grid center = one grid cell volume ---
    scale_factor = 1.0
    if units == 'A':
        scale_factor *= 10

    unique_radii = unique_radii * scale_factor
    # --- Sphere volume weighting ---
    sphere_volumes = (4.0/3.0) * np.pi * (unique_radii ** 3)
    counts = counts * sphere_volumes              # keep using 'counts' below
    # Apply unit/diameter scaling BEFORE derivative
    scale_factor = 1.0
    if diameter:
        scale_factor *= 2
   
    unique_radii = unique_radii * scale_factor
    # Compute CDF/PDF
    counts1 = counts[::-1]
    cdf = np_to_use.cumsum(counts1) / np_to_use.sum(counts1)
    cdf_reversed = cdf[::-1]
    derivative = -1 * np_to_use.gradient(cdf_reversed, unique_radii)
    area = np_to_use.trapz(derivative, unique_radii)
    #Normalize
    # --- robust PDF normalization ---
    # 1) sort & clean
    order = np_to_use.argsort(unique_radii)
    unique_radii = unique_radii[order]
    derivative   = derivative[order]
    mask = np_to_use.isfinite(unique_radii) & np_to_use.isfinite(derivative)
    unique_radii = unique_radii[mask]
    derivative   = np_to_use.maximum(derivative[mask], 0.0)  # clip tiny negatives

    # 2) area normalization (∫ pdf dr = 1)
    area = np_to_use.trapz(derivative, unique_radii)
    if not np_to_use.isfinite(area) or area <= 1e-12:
        # fallback to CDF span; should be ~1 if CDF is a fraction
        area = float(cdf_reversed[0] - cdf_reversed[-1]) if cdf_reversed.size > 1 else 1.0
    derivative = derivative / area  # now a proper PDF (units: 1 / radius-unit)
    area_final = np_to_use.trapz(derivative, unique_radii)
    print(f"The area under the curve is:{area_final}")
    # -------------------------------------------------------------------------

    # ------------------------------ Gaussian fit -----------------------------
    if Gaussian_Fit == True:
        from scipy.optimize import curve_fit
        x_data = unique_radii
        y_data = derivative
        def gaussian(x, amplitude, mean, stddev):
            return amplitude * np.exp(-0.5 * ((x - mean) / stddev) ** 2)
        initial_guess = [y_data.max() if y_data.size else 1.0,
                         x_data[np.argmax(y_data)] if y_data.size else 0.0,
                         (x_data.max()-x_data.min())/6.0 if x_data.size>1 else 1.0]
        try:
            params, covariance = curve_fit(gaussian, x_data, y_data, p0=initial_guess, maxfev=10000)
            x_fit = np.linspace(x_data.min(), x_data.max(), 1000)
            y_fit = gaussian(x_fit, *params)
            y_fit = y_fit/np.max(y_fit) if np.max(y_fit) > 0 else y_fit
        except Exception:
            params, x_fit, y_fit = [], [], []
    else:
        x_fit, y_fit, params = [], [], []
    # -------------------------------------------------------------------------

    return unique_radii, counts, cdf_reversed,  derivative, x_fit, y_fit, params, sphere_list


import numpy as np
import re

BONDI_RADII_NM = {
    "C": 0.17, "N": 0.155, "O": 0.152, "F": 0.147,
    "Si": 0.210, "P": 0.180, "S": 0.180, "Cl": 0.175,
    "Br": 0.183, "H": 0.110,
}

ATOMIC_MASS_G_PER_MOL = {
    "H": 1.0079, "C": 12.011, "N": 14.007, "O": 15.999,
    "F": 18.998, "Si": 28.085, "P": 30.974, "S": 32.06,
    "Cl": 35.45, "Br": 79.904,
}

AVOGADRO = 6.02214076e23

def compute_molecular_volumes_from_gro(
    gro_file_path: str,
    bondi_radii_nm: dict = BONDI_RADII_NM,
    n_samples: int = 1_000_000,
    samples_per_chunk: int = 20_000,
    batch_atoms: int = 1000,
    rng_seed: int = 0,
    density_kg_per_m3: float | None = None,
    default_radius_nm: float | None = None,
    prefix: str = ""
):
    """
    Compute union-of-spheres molecular volume (overlaps handled) and optional bulk volume.
    Keys in output dict will be prefixed with `prefix` if provided.
    """
    # ---------- helpers ----------
    def _load_gro_atoms(path):
        full, names, X, Y, Z = [], [], [], [], []
        with open(path, "r") as fh:
            lines = fh.readlines()
        for line in lines[1:]:
            cols = line.split()
            if len(cols) >= 6:
                names.append(re.sub(r'\d+', '', cols[1]))
                full.append(cols[1])
                X.append(float(cols[3])); Y.append(float(cols[4])); Z.append(float(cols[5]))
            elif len(cols) == 5 and cols[0] != 'wetting':
                names.append(re.sub(r'\d+', '', cols[1]))
                full.append(cols[1])
                X.append(float(cols[2])); Y.append(float(cols[3])); Z.append(float(cols[4]))
        return np.array(full), np.array(names), np.array(X), np.array(Y), np.array(Z)

    def _radii_from_names(names):
        r_default = (min(bondi_radii_nm.values()) if default_radius_nm is None else default_radius_nm)
        return np.array([bondi_radii_nm.get(n, r_default) for n in names], dtype=float)

    def _tight_aabb(centers, radii):
        lo = np.minimum.reduce(centers - radii[:, None])
        hi = np.maximum.reduce(centers + radii[:, None])
        return lo, hi

    def _union_volume_mc(centers, radii, nS, m_chunk, k_batch, seed):
        rng = np.random.default_rng(seed)
        r2 = radii * radii
        lo, hi = _tight_aabb(centers, radii)
        box_len = hi - lo
        V_box = float(np.prod(box_len))
        hits = 0
        remaining = nS

        while remaining > 0:
            m = min(m_chunk, remaining)
            pts = lo + rng.random((m, 3)) * box_len
            inside = np.zeros(m, dtype=bool)
            pts2 = np.einsum('ij,ij->i', pts, pts)[:, None]

            for start in range(0, centers.shape[0], k_batch):
                stop = min(start + k_batch, centers.shape[0])
                C = centers[start:stop, :]
                C2 = np.einsum('ij,ij->i', C, C)[None, :]
                dot = pts @ C.T
                d2 = pts2 + C2 - 2.0 * dot  # (m, k)

                # -------- FIX: reduce over atoms before OR --------
                inside |= np.any(d2 <= r2[None, start:stop], axis=1)
                # ---------------------------------------------------

                if inside.all():
                    break

            hits += int(inside.sum())
            remaining -= m

        p = hits / float(nS)
        V_union = p * V_box
        stderr = np.sqrt(p * (1.0 - p) / nS) * V_box
        return V_union, stderr, V_box, p, lo, hi

    # ---------- main ----------
    full, names, x, y, z = _load_gro_atoms(gro_file_path)
    centers = np.column_stack([x, y, z])
    radii = _radii_from_names(names)

    sum_sphere_vol_nm3 = float((4.0 / 3.0) * np.pi * np.sum(radii**3))

    V_union_nm3, V_se_nm3, V_box_nm3, p_hit, lo, hi = _union_volume_mc(
        centers, radii, n_samples, samples_per_chunk, batch_atoms, rng_seed
    )

    def _as_element(nm):
        if len(nm) >= 2 and nm[:2] in ATOMIC_MASS_G_PER_MOL:
            return nm[:2]
        return nm[:1] if nm[:1] in ATOMIC_MASS_G_PER_MOL else None

    masses = []
    for nm in names:
        el = _as_element(nm)
        if el is None:
            masses.append(0.0)
        else:
            masses.append(ATOMIC_MASS_G_PER_MOL[el])
    total_mass_kg = (np.sum(masses) / AVOGADRO) / 1000.0

    bulk_volume_m3 = None
    if density_kg_per_m3 is not None and density_kg_per_m3 > 0:
        bulk_volume_m3 = total_mass_kg / density_kg_per_m3

    result = {
        f"{prefix}n_atoms": int(len(names)),
        f"{prefix}aabb_lo_nm": lo.tolist(),
        f"{prefix}aabb_hi_nm": hi.tolist(),
        f"{prefix}aabb_volume_nm3": V_box_nm3,
        f"{prefix}hit_fraction": p_hit,
        f"{prefix}union_volume_nm3": V_union_nm3,
        f"{prefix}union_volume_stderr_nm3": V_se_nm3,
        f"{prefix}sum_sphere_volumes_nm3": sum_sphere_vol_nm3,
        f"{prefix}packing_fraction_vs_naive": (V_union_nm3 / sum_sphere_vol_nm3) if sum_sphere_vol_nm3 > 0 else np.nan,
        f"{prefix}total_mass_kg": total_mass_kg,
        f"{prefix}bulk_volume_m3": bulk_volume_m3,
    }
    return result



def plot_VSD(path, unique_radii_L,counts,cdf_reversed, derivative, units = "nm", Diameter = False, save = True, case = "NONE"  ):
    
    
    unique_radii = np.array(unique_radii_L)
    unique_radii = unique_radii.astype(float)
    if Diameter == True:
        x_label = " Probe Diameter ({})".format(units)
    else:
        x_label = "Probe Radius ({})".format(units)
        
    plt.figure(figsize=(10, 6))  # Width: 10 inches, Height: 6 inches
    plt.scatter(unique_radii, counts, edgecolor='black')
    plt.xlabel(x_label)
    plt.ylabel('Frequency')
    plt.title(f' {case} Histogram of Inserted Sphere Radii')
    if save == True:
        plt.savefig(os.path.join(path, "PSD_histogram.png"))
        
    plt.show()

    # Plot the CDF
    plt.figure(figsize=(10, 6))  # Width: 10 inches, Height: 6 inches
    plt.plot(unique_radii, cdf_reversed, marker='o', linestyle='-')
    plt.xlabel(x_label)
    plt.ylabel('Cumulative Void Size Distribution')
    plt.title(f' {case} Cumulative Distribution ')
    plt.grid(True)
    if save == True:
        plt.savefig(os.path.join(path, "PSD_Cumulative.png"))
    plt.show()

    # Plot the derivative of -dV/dr 
   
    max_index = np.argmax(derivative)
    max_value = derivative[max_index]
    max_x = unique_radii[max_index]
    max_string = round(unique_radii[max_index], 3)

    ############3
    plt.figure(figsize=(10, 6))  # Width: 10 inches, Height: 6 inches
    plt.plot(unique_radii, derivative, marker='o', linestyle='-')
    plt.axvline(x=max_x, color='r', linestyle='-.')
    plt.xlabel(x_label)
    plt.ylabel('Void size distribution (1/{})'.format(units))
    plt.title(f' {case} Pore size distribution vs radii ')
    plt.grid(True)
    plt.text(unique_radii[-5], (max_value - (0.20*max_value)), f'{x_label}: {max_string}', fontsize=12, ha='right')
    if save == True:
        plt.savefig(os.path.join(path, "PSD_Derivative.png"))
    plt.show()



def load_and_plot_single_with_colorbar(job_path, file_suffix, ax, title, filter_by_radius=False):
    """
    Loads sphere_list{file_suffix}.pyx from job_path and plots ONE panel on ax.
    Adds a yellow→blue colorbar if filter_by_radius=False.
    """
    file_path = os.path.join(job_path, f"sphere_list{file_suffix}.pyx")
    with open(file_path, 'rb') as f:
        sphere_list = pickle.load(f)

    radii = np.array([r for _, r in sphere_list], dtype=float)
    rmin, rmax = float(radii.min()), float(radii.max())

    # normalize radii to [0.01, 1.0] for visualization
    if rmax > rmin:
        normalized_radii = 0.01 + (radii - rmin) / (rmax - rmin) * (1.0 - 0.01)
    else:
        normalized_radii = np.full_like(radii, 0.5)

    # yellow to blue colormap
    cmap = cm.get_cmap("YlGnBu")  # Yellow→Green→Blue
    norm = mcolors.Normalize(vmin=float(normalized_radii.min()),
                            vmax=float(normalized_radii.max()))

    # plot spheres
    plot_spheres(ax, sphere_list, normalized_radii, cmap, norm, filter_by_radius=filter_by_radius)

    # labels and aesthetics
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)
    ax.set_box_aspect([1, 1, 1])
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])

    # add colorbar if we are mapping radii
    if not filter_by_radius:
        sm = cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])  # required for colorbar
        cb = plt.colorbar(sm, ax=ax, shrink=0.6, pad=0.1)
        cb.set_label("Normalized radius", rotation=270, labelpad=15)



def plot_spheres(ax, sphere_list, normalized_radii, cmap, norm, filter_by_radius=False):
    """
    Plot spheres on ax.
    sphere_list: [(center_xyz, radius_nm), ...]
    normalized_radii: same length as sphere_list, normalized for visualization
    """
    u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
    base_x = np.cos(u) * np.sin(v)
    base_y = np.sin(u) * np.sin(v)
    base_z = np.cos(v)

    for (center, radius), norm_radius in zip(sphere_list, normalized_radii):
        x0, y0, z0 = center

        if filter_by_radius:
            if 0.14 <= radius < 0.22:
                color = 'blue'
            elif radius >= 0.22:
                color = 'green'
            else:
                continue
        else:
            color = cmap(norm(norm_radius))

        xs = x0 + norm_radius * base_x
        ys = y0 + norm_radius * base_y
        zs = z0 + norm_radius * base_z
        ax.plot_surface(xs, ys, zs, color=color, alpha=0.3, linewidth=0)