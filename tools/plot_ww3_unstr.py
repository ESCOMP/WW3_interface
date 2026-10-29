#!/usr/bin/env python3
"""
Plot / animate WW3 *unstructured-grid* fields from CESM history output, or
plot the mesh itself.

Handles both flavours of history a wave run produces:

  * native WW3,      <case>.ww3.hi.YYYY-MM-DD-SSSSS.nc      -- fields are HS,
    USSX, ...  and the mesh connectivity travels with the data in `nconn`.
  * mediator aux,    <case>.cpl.hx.ww3.YYYY-MM-DD-SSSSS.nc  -- fields are
    wavImp_Sw_hs_avg, ...  and only node coordinates are written, so the
    triangles have to come from elsewhere: pass the grid `.msh` (or any native
    history file) with --mesh.  Without it the nodes are Delaunay-triangulated,
    which also spans land and the domain's concavities.

Produces, in --outdir:
  <field>_panels.png  -- one map per time record
  <field>_zoom.png    -- full domain (zoom box outlined) + regional zoom
  <field>.gif         -- animation, optionally with Stokes-drift vectors
  mesh_overview.png   -- the mesh colored by nominal resolution, plus one
                         inset per --zoom showing the triangles (--plots mesh)

Nominal resolution is the edge length of an equilateral triangle with the
same spherical area as the mesh triangle.

Usage:
  plot_ww3_unstr.py RUNDIR [--field HS] [--mesh mesh.msh] [--outdir DIR]
  plot_ww3_unstr.py --mesh mesh.msh [--zoom "Alaska:-170,-130,52,72" ...]

Needs numpy, netCDF4 and matplotlib.
"""
import argparse
import glob
import os
import sys

import numpy as np
import netCDF4
import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Rectangle

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ww3_msh_to_esmf import parse_gmsh, cell_area_rad  # noqa: E402

R_EARTH_KM = 6371.0

# Values outside this range are missing: WW3 writes -999.9, the mediator 1e30.
VALID = (-900.0, 1.0e29)

# Mediator field roots keyed by the native WW3 name, so --field HS works on
# either flavour of file. Matching is otherwise by name alone.
ALIASES = {"HS": "hs", "T01": "Tm1", "TM1": "Tm1", "DIR": "thm", "THM": "thm",
           "USSX": "ustokes", "USSY": "vstokes", "FP": "fp0", "TP": "fp0"}
ALIASES_REV = {v.lower(): k for k, v in ALIASES.items()}

# Vector pair to overlay, by native name; resolved per file like any other field.
VECTOR_PAIR = ("USSX", "USSY")

# Above this many triangles the element edges are a black smear, not a mesh.
EDGE_LIMIT = 20000
# Mesh insets have no field under the edges, so they stay legible for longer.
MESH_EDGE_LIMIT = 50000


def wrap180(lon):
    """Normalize longitude(s) to [-180, 180). Accepts scalars or arrays; makes the
    plotting grid-agnostic to the mesh's 0..360 vs -180..180 convention."""
    return (np.asarray(lon) + 180.0) % 360.0 - 180.0


def normalize(name):
    """Strip the mediator's decoration so wavImp_Sw_hs_avg and HS compare equal."""
    n = name
    for pre in ("wavImp_", "wavExp_", "medImp_", "medExp_"):
        if n.startswith(pre):
            n = n[len(pre):]
    if n.endswith("_avg"):
        n = n[:-4]
    for pre in ("Sw_", "Sa_", "So_", "Si_"):
        if n.startswith(pre):
            n = n[len(pre):]
    return n.lower()


def resolve(ds, want):
    """Variable in ds matching `want` under either naming convention, or None."""
    table = {normalize(v): v for v in ds.variables}
    keys = [normalize(want)]
    if want.upper() in ALIASES:
        keys.append(normalize(ALIASES[want.upper()]))
    if keys[0] in ALIASES_REV:
        keys.append(normalize(ALIASES_REV[keys[0]]))
    for k in keys:
        if k in table:
            return table[k]
    return None


def mask_missing(a):
    return np.ma.masked_outside(np.ma.masked_invalid(a), *VALID)


def find_files(paths):
    """Expand directories and globs into a sorted list of history files."""
    out = []
    for p in paths:
        if os.path.isdir(p):
            hits = (glob.glob(os.path.join(p, "*.ww3.hi.*.nc")) or
                    glob.glob(os.path.join(p, "*.cpl.hx.ww3.*.nc")))
            if not hits:
                sys.exit(f"ERROR: no *.ww3.hi.*.nc or *.cpl.hx.ww3.*.nc under {p}")
            out += hits
        else:
            hits = glob.glob(p)
            if not hits:
                sys.exit(f"ERROR: no such file: {p}")
            out += hits
    return sorted(set(out))


def read_coords(ds):
    """Node lon/lat, from either flavour. Returns arrays of shape (nnode,)."""
    lonv, latv = resolve(ds, "lon"), resolve(ds, "lat")
    if lonv is None or latv is None:
        sys.exit("ERROR: no longitude/latitude variables in the history file")
    lon, lat = ds.variables[lonv], ds.variables[latv]
    # the mediator writes the coordinates once per time record; they are static
    lon = lon[0] if "time" in lon.dimensions else lon[:]
    lat = lat[0] if "time" in lat.dimensions else lat[:]
    return np.asarray(lon).ravel(), np.asarray(lat).ravel()


def read_mesh(path):
    """Node lon/lat and zero-based triangles from a gmsh .msh or a native WW3 history."""
    if path.endswith(".msh"):
        return parse_gmsh(path)
    with netCDF4.Dataset(path) as ds:
        if "nconn" not in ds.variables:
            sys.exit(f"ERROR: {path} has no `nconn`; pass the grid .msh instead")
        lon, lat = read_coords(ds)
        return lon, lat, np.asarray(ds.variables["nconn"][:]) - 1


def read_connectivity(path, nnode):
    """Zero-based triangles from --mesh, checked against the history's node count."""
    _, _, tris = read_mesh(path)
    # A connectivity built for a different grid would place every field on the
    # wrong nodes without any visible error, so refuse rather than guess.
    if int(tris.max()) + 1 != nnode:
        sys.exit(f"ERROR: --mesh has {int(tris.max()) + 1} nodes, history has {nnode}")
    return tris


def build_triangulation(lon, lat, tris):
    """Triangulation with seam-crossing elements masked off."""
    triang = Triangulation(lon, lat, tris)
    # Mask triangles that wrap across the longitude seam (at +/-180). Such a
    # triangle is physically small (nodes adjacent across the seam) but spans the
    # full width in the lon/lat plane and renders as a spurious horizontal streak.
    tlon = lon[triang.triangles]
    triang.set_mask(tlon.max(axis=1) - tlon.min(axis=1) > 180.0)
    return triang


def subset(triang, box):
    """Triangulation restricted to a lon/lat box, plus the node index it used
    and the mask of triangles it kept.

    Clipping with set_xlim still renders every element; on a million-node mesh
    the zoom panel is only affordable if the triangles outside are dropped.
    """
    lon0, lon1, lat0, lat1 = box
    inside = ((triang.x >= lon0) & (triang.x <= lon1) &
              (triang.y >= lat0) & (triang.y <= lat1))
    keep = inside[triang.triangles].any(axis=1)
    if triang.mask is not None:
        keep &= ~triang.mask
    tri = triang.triangles[keep]
    if tri.size == 0:
        return None, None, None
    idx = np.unique(tri)
    remap = np.zeros(triang.x.size, dtype=np.int64)
    remap[idx] = np.arange(idx.size)
    return Triangulation(triang.x[idx], triang.y[idx], remap[tri]), idx, keep


def nominal_resolution_km(lon, lat, tris):
    """Per-triangle edge length of the equilateral triangle of equal spherical area."""
    area_km2 = cell_area_rad(lon[tris], lat[tris]) * R_EARTH_KM**2
    return np.sqrt(4.0 * area_km2 / np.sqrt(3.0))


def parse_zoom(spec):
    """'[NAME:]lon0,lon1,lat0,lat1' -> (name, box), lon wrapped to -180..180."""
    name, _, box = spec.rpartition(":")
    try:
        z = [float(v) for v in box.split(",")]
    except ValueError:
        z = []
    if len(z) != 4:
        sys.exit(f"ERROR: --zoom '{spec}' is not [NAME:]lon0,lon1,lat0,lat1")
    return name.strip(), (float(wrap180(z[0])), float(wrap180(z[1])), z[2], z[3])


def load(files, field, want_vectors, stride):
    """Return triangulation inputs and a list of (label, field, ussx, ussy)."""
    with netCDF4.Dataset(files[0]) as ds:
        lon, lat = read_coords(ds)
        fname = resolve(ds, field)
        if fname is None:
            sys.exit(f"ERROR: no variable matching '{field}'. Available: "
                     + ", ".join(sorted(ds.variables)))
        long_name = getattr(ds.variables[fname], "long_name", fname)
        units_f = getattr(ds.variables[fname], "units", "")
        vecs = [resolve(ds, v) for v in VECTOR_PAIR] if want_vectors else [None, None]
        if want_vectors and None in vecs:
            print(f"note: {VECTOR_PAIR} not in the file; vectors disabled")
            vecs = [None, None]
        tris = ds.variables["nconn"][:] - 1 if "nconn" in ds.variables else None

    frames = []
    seen = 0
    for f in files:
        with netCDF4.Dataset(f) as ds:
            t = ds.variables["time"]
            units, cal = t.units, getattr(t, "calendar", "standard")
            for it in range(ds.dimensions["time"].size):
                if seen % stride:
                    seen += 1
                    continue
                seen += 1
                when = netCDF4.num2date(t[it], units, cal).strftime("%Y-%m-%d %H:%M")
                z = mask_missing(np.asarray(ds.variables[fname][it], dtype=np.float32).ravel())
                ussx = ussy = None
                if vecs[0] is not None:
                    ussx = mask_missing(np.asarray(ds.variables[vecs[0]][it], dtype=np.float32).ravel())
                    ussy = mask_missing(np.asarray(ds.variables[vecs[1]][it], dtype=np.float32).ravel())
                frames.append((when, z, ussx, ussy))
    if not frames:
        sys.exit("ERROR: no time records read")
    return lon, lat, tris, frames, fname, long_name, units_f


def color_range(frames, vmax_arg, vmin_arg):
    """Robust color limits: 0 .. 99th percentile (or user --vmax), so a single
    storm cell does not wash out the rest of the field."""
    allv = np.concatenate([np.ma.asarray(fr[1]).compressed() for fr in frames])
    vmin = float(vmin_arg) if vmin_arg is not None else 0.0
    vmax = float(vmax_arg) if vmax_arg is not None else float(np.percentile(allv, 99))
    return vmin, vmax


def draw_edges(ax, triang, show, lw, alpha, limit=EDGE_LIMIT):
    if show == "off" or (show == "auto" and triang.triangles.shape[0] > limit):
        return
    ax.triplot(triang, color="k", lw=lw, alpha=alpha)


def make_panels(triang, frames, field, long_name, units_f, outpath, clim,
                edges, max_panels):
    if len(frames) > max_panels:
        keep = np.linspace(0, len(frames) - 1, max_panels).round().astype(int)
        frames = [frames[i] for i in keep]
    n = len(frames)
    ncol = min(n, 3)
    nrow = int(np.ceil(n / ncol))
    vmin, vmax = clim
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 3.4 * nrow),
                             squeeze=False, constrained_layout=True)
    last = None
    for k, (when, z, _, _) in enumerate(frames):
        ax = axes[k // ncol][k % ncol]
        last = ax.tripcolor(triang, z, shading="gouraud", cmap="turbo",
                            vmin=vmin, vmax=vmax)
        draw_edges(ax, triang, edges, 0.1, 0.25)
        ax.set_title(when, fontsize=9)
        ax.set_aspect("equal")
        ax.set_xlabel("lon"); ax.set_ylabel("lat")
    for k in range(n, nrow * ncol):
        axes[k // ncol][k % ncol].axis("off")
    cb = fig.colorbar(last, ax=axes, shrink=0.8)
    cb.set_label(f"{long_name} [{units_f}]")
    fig.suptitle(f"WW3 unstructured grid — {field} ({long_name})", fontsize=12)
    fig.savefig(outpath, dpi=130)
    plt.close(fig)
    print("wrote", outpath)


def make_zoom_pair(triang, frames, field, long_name, units_f, outpath, zoom,
                   clim, edges, itime=-1):
    """Two panels for one time: full domain (with the zoom box outlined) + a regional
    zoom that draws the triangular elements prominently so the mesh is visible."""
    vmin, vmax = clim
    when, z = frames[itime][0], frames[itime][1]
    lon0, lon1, lat0, lat1 = zoom
    sub, idx, _ = subset(triang, zoom)
    if sub is None:
        print(f"note: no elements inside {zoom}; skipping the zoom figure")
        return
    fig, (axL, axR) = plt.subplots(2, 1, figsize=(10, 9), constrained_layout=True)
    # full domain (top)
    axL.tripcolor(triang, z, shading="gouraud", cmap="turbo", vmin=vmin, vmax=vmax)
    draw_edges(axL, triang, edges, 0.1, 0.2)
    axL.add_patch(Rectangle((lon0, lat0), lon1 - lon0, lat1 - lat0,
                            fill=False, ec="red", lw=1.4))
    axL.set_aspect("equal"); axL.set_xlabel("lon"); axL.set_ylabel("lat")
    axL.set_title(f"{when}   (full domain)", fontsize=10)
    # regional zoom with visible mesh edges
    tpc = axR.tripcolor(sub, z[idx], shading="gouraud", cmap="turbo",
                        vmin=vmin, vmax=vmax)
    draw_edges(axR, sub, edges, 0.4, 0.6)
    axR.set_xlim(lon0, lon1); axR.set_ylim(lat0, lat1)
    axR.set_aspect("equal"); axR.set_xlabel("lon"); axR.set_ylabel("lat")
    axR.set_title("zoom — triangular elements", fontsize=10)
    cb = fig.colorbar(tpc, ax=[axL, axR], shrink=0.85)
    cb.set_label(f"{long_name} [{units_f}]")
    fig.suptitle(f"WW3 unstructured grid — {field} ({long_name})", fontsize=12)
    fig.savefig(outpath, dpi=140)
    plt.close(fig)
    print("wrote", outpath)


def make_animation(triang, frames, field, long_name, units_f, outpath, zoom,
                   clim, edges, dpi=100, vector_scale=None):
    """Animation of the full domain, plus a regional zoom panel when --zoom is given."""
    vmin, vmax = clim
    sub = idx = None
    if zoom is not None:
        sub, idx, _ = subset(triang, zoom)
    nrow = 2 if sub is not None else 1
    fig, axes = plt.subplots(nrow, 1, figsize=(10, 4.5 * nrow),
                             squeeze=False, constrained_layout=True)
    axL = axes[0][0]
    z0 = frames[0][1]
    tpcL = axL.tripcolor(triang, z0, shading="gouraud", cmap="turbo", vmin=vmin, vmax=vmax)
    draw_edges(axL, triang, edges, 0.1, 0.2)
    axL.set_aspect("equal"); axL.set_xlabel("lon"); axL.set_ylabel("lat")
    tpcR = None
    if sub is not None:
        lon0, lon1, lat0, lat1 = zoom
        axR = axes[1][0]
        axL.add_patch(Rectangle((lon0, lat0), lon1 - lon0, lat1 - lat0,
                                fill=False, ec="red", lw=1.4))
        tpcR = axR.tripcolor(sub, z0[idx], shading="gouraud", cmap="turbo",
                             vmin=vmin, vmax=vmax)
        draw_edges(axR, sub, edges, 0.4, 0.6)
        axR.set_xlim(lon0, lon1); axR.set_ylim(lat0, lat1); axR.set_aspect("equal")
        axR.set_xlabel("lon"); axR.set_ylabel("lat"); axR.set_title("zoom — elements", fontsize=10)
    cb = fig.colorbar(tpcR if tpcR is not None else tpcL, ax=list(axes.ravel()), shrink=0.85)
    cb.set_label(f"{long_name} [{units_f}]")
    sup = fig.suptitle("")

    quiv = None
    if frames[0][2] is not None:
        vidx = np.arange(triang.x.size)[::max(1, triang.x.size // 200)]
        # Scale off the 95th percentile: a fixed scale tuned on one grid leaves the
        # arrows invisible on another, and the handful of shallow-node outliers
        # would swamp matplotlib's own autoscaling.
        mag = np.ma.sqrt(frames[0][2] ** 2 + frames[0][3] ** 2).compressed()
        scale = vector_scale or 20.0 * max(float(np.percentile(mag, 95)), 1e-6)
        quiv = axL.quiver(triang.x[vidx], triang.y[vidx],
                          frames[0][2][vidx], frames[0][3][vidx],
                          color="white", scale=scale, width=0.003, alpha=0.8)

    def update(i):
        when, z, ussx, ussy = frames[i]
        tpcL.set_array(z)                      # gouraud: per-node values
        if tpcR is not None:
            tpcR.set_array(z[idx])
        sup.set_text(f"WW3 unstructured {field} ({long_name}) — {when}")
        if quiv is not None and ussx is not None:
            quiv.set_UVC(ussx[vidx], ussy[vidx])
        return tpcL, sup

    anim = FuncAnimation(fig, update, frames=len(frames), blit=False)
    try:
        anim.save(outpath, writer=PillowWriter(fps=2), dpi=dpi)
        print("wrote", outpath)
    except Exception as e:
        print(f"GIF write failed ({e}); saving frames as PNGs instead")
        for i in range(len(frames)):
            update(i)
            fig.savefig(outpath.replace(".gif", f"_{i:03d}.png"), dpi=120)
    plt.close(fig)


def make_mesh_overview(triang, zooms, outpath, vmin_arg, vmax_arg, title, edges, dpi):
    """Mesh colored by nominal resolution, plus one inset per zoom box."""
    res = nominal_resolution_km(triang.x, triang.y, triang.triangles)
    print(f"nominal resolution (km): min={res.min():.2f} p1={np.percentile(res, 1):.2f} "
          f"median={np.median(res):.2f} p99={np.percentile(res, 99):.2f} max={res.max():.2f}")
    vmin = float(vmin_arg) if vmin_arg is not None else float(np.floor(np.percentile(res, 1)))
    vmax = float(vmax_arg) if vmax_arg is not None else float(np.ceil(np.percentile(res, 99)))

    ncols = max(1, len(zooms))
    fig = plt.figure(figsize=(11, 8.8 if zooms else 5.0), dpi=dpi)
    gs = fig.add_gridspec(2 if zooms else 1, ncols, hspace=0.28, wspace=0.12,
                          height_ratios=[1.0, 0.85] if zooms else None)
    axg = fig.add_subplot(gs[0, :])
    tpc = axg.tripcolor(triang, facecolors=res, cmap="viridis", vmin=vmin, vmax=vmax)
    axg.set_aspect("equal")
    axg.set_xlim(triang.x.min(), triang.x.max()); axg.set_ylim(triang.y.min(), triang.y.max())
    axg.set_xlabel("lon"); axg.set_ylabel("lat")
    axg.set_title(f"{triang.x.size:,} nodes, colored by nominal resolution",
                  fontsize=11, loc="left")
    for name, (lon0, lon1, lat0, lat1) in zooms:
        axg.add_patch(Rectangle((lon0, lat0), lon1 - lon0, lat1 - lat0,
                                fill=False, ec="red", lw=1.3))
        if name:
            axg.annotate(name, (lon1, lat1), textcoords="offset points", xytext=(5, 6),
                         fontsize=8, color="red", ha="left", va="bottom",
                         bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.75))

    for k, (name, box) in enumerate(zooms):
        ax = fig.add_subplot(gs[1, k])
        sub, _, keep = subset(triang, box)
        if sub is None:
            ax.text(0.5, 0.5, f"no mesh in\n{name or box}", ha="center", va="center",
                    transform=ax.transAxes)
            continue
        ax.tripcolor(sub, facecolors=res[keep], cmap="viridis", vmin=vmin, vmax=vmax)
        draw_edges(ax, sub, edges, 0.25, 0.6, limit=MESH_EDGE_LIMIT)
        lon0, lon1, lat0, lat1 = box
        ax.set_xlim(lon0, lon1); ax.set_ylim(lat0, lat1)
        ax.set_aspect("equal")
        ax.set_title(f"{name + '  ' if name else ''}({sub.triangles.shape[0]:,} triangles)",
                     fontsize=10, loc="left")
        ax.set_xlabel("lon")
        if k == 0:
            ax.set_ylabel("lat")

    cb = fig.colorbar(tpc, ax=fig.axes, shrink=0.7, pad=0.015, aspect=30)
    cb.set_label("nominal resolution (km)")
    fig.suptitle(title, fontsize=13, x=0.02, ha="left", y=0.995)
    fig.savefig(outpath, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print("wrote", outpath)


def main():
    p = argparse.ArgumentParser(
        description="Plot WW3 unstructured-grid fields from native or mediator history.")
    p.add_argument("paths", nargs="*",
                   help="history files, globs, or a directory to search "
                        "(*.ww3.hi.*.nc, else *.cpl.hx.ww3.*.nc); "
                        "optional for --plots mesh")
    p.add_argument("--field", default="HS",
                   help="field name in either convention, e.g. HS or wavImp_Sw_hs_avg")
    p.add_argument("--mesh", default=None,
                   help="grid .msh or a native history file, for the triangles; "
                        "required for cpl.hx input unless a Delaunay fill-in will do")
    p.add_argument("--outdir", default=".")
    p.add_argument("--plots", default=None,
                   help="comma-separated subset of panels,zoom,anim,mesh "
                        "(default: panels,zoom,anim with history paths, else mesh)")
    p.add_argument("--vectors", action="store_true",
                   help="overlay surface Stokes-drift vectors on the animation (default: off)")
    p.add_argument("--vmax", default=None,
                   help="fixed colorbar max, in field units or km for mesh (default: 99th pctile)")
    p.add_argument("--vmin", default=None,
                   help="fixed colorbar min (default: 0, or the 1st pctile for mesh)")
    p.add_argument("--zoom", action="append", default=[],
                   help="zoom box '[NAME:]lon0,lon1,lat0,lat1' (lon in either 0-360 or "
                        "-180..180); repeatable for mesh, the first is used for zoom and "
                        "anim; omitted = no zoom panel")
    p.add_argument("--title", default="WW3 unstructured mesh", help="mesh plot title")
    p.add_argument("--stride", type=int, default=1, help="use every Nth time record")
    p.add_argument("--max-panels", type=int, default=12, dest="max_panels",
                   help="cap on the number of maps in the panels figure")
    p.add_argument("--edges", choices=["auto", "on", "off"], default="auto",
                   help=f"draw element edges; auto = only under {EDGE_LIMIT} triangles")
    p.add_argument("--vector-scale", type=float, default=None, dest="vector_scale",
                   help="quiver scale (default: from the 95th percentile magnitude)")
    p.add_argument("--gif-dpi", type=int, default=100, dest="gif_dpi", help="GIF resolution (dpi)")
    # argparse takes a value like '-82,-60,24,46' for an option; attach it to --zoom
    argv, it = [], iter(sys.argv[1:])
    for a in it:
        argv.append(f"--zoom={next(it, '')}" if a == "--zoom" else a)
    args = p.parse_args(argv)
    zooms = [parse_zoom(z) for z in args.zoom]

    plots = args.plots or ("panels,zoom,anim" if args.paths else "mesh")
    want = {s.strip() for s in plots.split(",")}
    unknown = want - {"panels", "zoom", "anim", "mesh"}
    if unknown:
        p.error(f"unknown --plots: {', '.join(sorted(unknown))}")
    field_plots = want - {"mesh"}
    if field_plots and not args.paths:
        p.error(f"--plots {','.join(sorted(field_plots))} needs history paths")
    if not args.paths and not args.mesh:
        p.error("give history paths, or --mesh for the mesh plot")
    os.makedirs(args.outdir, exist_ok=True)

    if args.paths:
        files = find_files(args.paths)
        print(f"{len(files)} file(s), first: {files[0]}")
        lon, lat, tris, frames, fname, long_name, units_f = load(
            files, args.field, args.vectors, max(1, args.stride))
        if tris is None and args.mesh:
            tris = read_connectivity(args.mesh, lon.size)
    else:
        lon, lat, tris = read_mesh(args.mesh)
    if tris is None:
        if "mesh" in want:
            sys.exit("ERROR: the mesh plot needs the real triangles; pass --mesh")
        print("note: no connectivity in the history and no --mesh; "
              "Delaunay-triangulating the nodes, which also spans land")
    # Grid-agnostic: normalize longitude to [-180, 180) regardless of whether the
    # mesh stored 0..360 or -180..180, so the same zoom box works for any grid.
    lon = wrap180(lon)
    triang = build_triangulation(
        lon, lat, tris if tris is not None else Triangulation(lon, lat).triangles)

    nmask = int(triang.mask.sum()) if triang.mask is not None else 0
    nrec = f"{len(frames)} time records, " if args.paths else ""
    print(f"{nrec}{triang.x.size} nodes, "
          f"{triang.triangles.shape[0]} triangles ({nmask} seam-crossing masked)")

    zoom = zooms[0][1] if zooms else None
    if "mesh" in want:
        make_mesh_overview(triang, zooms, os.path.join(args.outdir, "mesh_overview.png"),
                           args.vmin, args.vmax, args.title, args.edges, dpi=170)
    if not field_plots:
        return
    clim = color_range(frames, args.vmax, args.vmin)
    tag = args.field

    if "panels" in want:
        make_panels(triang, frames, tag, long_name, units_f,
                    os.path.join(args.outdir, f"{tag}_panels.png"), clim,
                    args.edges, args.max_panels)
    if "zoom" in want:
        if zoom is None:
            print("note: --zoom not given; skipping the zoom figure")
        else:
            make_zoom_pair(triang, frames, tag, long_name, units_f,
                           os.path.join(args.outdir, f"{tag}_zoom.png"), zoom,
                           clim, args.edges)
    if "anim" in want:
        make_animation(triang, frames, tag, long_name, units_f,
                       os.path.join(args.outdir, f"{tag}.gif"), zoom, clim,
                       args.edges, dpi=args.gif_dpi, vector_scale=args.vector_scale)


if __name__ == "__main__":
    main()
