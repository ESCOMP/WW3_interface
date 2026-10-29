#!/usr/bin/env python3
"""Convert a WW3 unstructured gmsh mesh into an ESMF mesh for CESM coupling.

The CESM WW3 cap builds an arbitrary DistGrid with one coupling point per WW3
node and reads the coupling geometry from a pre-built ESMF mesh (the mesh_wav
attribute). The ESMF mesh must therefore have one *element* per WW3 node, in
node order. In the .msh the nodes are triangle vertices, so we build the median
dual: each node becomes a polygon whose corners are the midpoints of its
incident edges and the centroids of its incident triangles.

Usage:
    ww3_msh_to_esmf.py <in.msh> <out_esmf.nc> [--title TITLE]
"""

import argparse
import sys
from datetime import datetime

import numpy as np
from netCDF4 import Dataset

D2R = np.pi / 180.0
R2D = 180.0 / np.pi


# --- spherical polygon area -------------------------------------------------
# Vendored verbatim from NCAR/mom6_forge (mom6_forge/utils.py) hash: 5c28a57

def _spherical_angle(a, b):
    """Compute angle between unit vectors a and b on a sphere.
    This function is used for area computation.

    Parameters
    ----------
    a : np.ndarray, shape (..., 3)
        First unit vector (x, y, z).
    b : np.ndarray, shape (..., 3)
        Second unit vector (x, y, z).

    Returns
    -------
    angle : np.ndarray
        Angle in radians.
    """

    cross = np.cross(a, b)
    sin_angle = np.linalg.norm(cross, axis=-1)
    cos_angle = np.sum(a * b, axis=-1)
    return np.arctan2(sin_angle, cos_angle)


def _tri_area(u, v, w):
    """Vectorized spherical triangle area using L'Huilier's theorem.

    Parameters:
    -----------
    u, v, w: np.ndarray of shape (..., 3)
        Arrays of unit vectors representing triangle vertices.

    Returns:
    --------
    np.ndarray of shape (...,): Areas of the spherical triangles.
    """
    a = _spherical_angle(v, w)
    b = _spherical_angle(u, w)
    c = _spherical_angle(u, v)
    s = 0.5 * (a + b + c)

    t = (
        np.tan(0.5 * s)
        * np.tan(0.5 * (s - a))
        * np.tan(0.5 * (s - b))
        * np.tan(0.5 * (s - c))
    )

    area = np.abs(4.0 * np.arctan(np.sqrt(np.abs(t))))
    return area


def _great_circle_area(polygon_unitvecs):
    """
    Compute area of multiple spherical polygons using triangulation from vertex 0.

    Parameters:
    -----------
    polygon_unitvecs: np.ndarray of shape (n_poly, n_verts, 3)

    Returns:
    --------
    np.ndarray of shape (n_poly,): Areas of the spherical polygons
    """
    n_poly, n_verts, _ = polygon_unitvecs.shape
    if n_verts < 3:
        return np.zeros(n_poly)

    pnt0 = polygon_unitvecs[:, 0:1, :]  # shape (n_poly, 1, 3)
    pnt1 = polygon_unitvecs[:, 1:-1, :]  # shape (n_poly, n_verts-2, 3)
    pnt2 = polygon_unitvecs[:, 2:, :]  # shape (n_poly, n_verts-2, 3)

    u = np.broadcast_to(pnt0, pnt1.shape)  # broadcast pnt0
    areas = _tri_area(u, pnt1, pnt2)  # shape (n_poly, n_verts-2)
    return np.sum(areas, axis=1)  # sum over triangles per polygon


def cell_area_rad(xv_coords, yv_coords):
    """Compute the area of cell(s) using spherical polygons.

    Parameters
    ----------
    xv_coords : np.ndarray
        X-coordinate(s) of cell vertices in degrees.
    yv_coords : np.ndarray
        Y-coordinate(s) of cell vertices in degrees.

    Returns
    -------
    area : np.ndarray
        Area of the cell(s) in square radians.
    """

    # Convert coordinates to radians
    xv_rad = np.deg2rad(xv_coords)
    yv_rad = np.deg2rad(yv_coords)

    # Construct unit vectors
    x = np.cos(yv_rad) * np.cos(xv_rad)
    y = np.cos(yv_rad) * np.sin(xv_rad)
    z = np.sin(yv_rad)

    area = _great_circle_area(np.stack([x, y, z], axis=-1))
    return area

# --- end vendored -----------------------------------------------------------


def ll2xyz(lon, lat):
    lo, la = lon * D2R, lat * D2R
    return np.stack([np.cos(la) * np.cos(lo),
                     np.cos(la) * np.sin(lo),
                     np.sin(la)], axis=-1)


def xyz2ll(v):
    v = v / np.linalg.norm(v, axis=-1, keepdims=True)
    lat = np.arcsin(np.clip(v[..., 2], -1.0, 1.0)) * R2D
    lon = np.arctan2(v[..., 1], v[..., 0]) * R2D
    return lon, lat


def parse_gmsh(path):
    """Return node lon, lat (N,) and zero-based triangle vertices (T,3).

    Streamed: reading the whole file into a list of lines costs several times
    the file size in Python string objects.
    """
    with open(path) as f:
        for line in f:
            if line.strip() == "$Nodes":
                break
        nn = int(f.readline())
        nodes = np.loadtxt(f, max_rows=nn, usecols=(0, 1, 2))
        if not np.array_equal(nodes[:, 0], np.arange(1, nn + 1)):
            sys.exit("ERROR: node ids are not 1..N in file order; WW3 requires this")
        lon, lat = nodes[:, 1].copy(), nodes[:, 2].copy()
        del nodes

        for line in f:
            if line.strip() == "$Elements":
                break
        ne = int(f.readline())
        tris = np.empty((ne, 3), dtype=np.int64)
        k = 0
        for _ in range(ne):
            p = f.readline().split()
            if p[1] == "2":                   # gmsh element type 2 = triangle
                nt = int(p[2])
                tris[k] = p[3 + nt:3 + nt + 3]
                k += 1
    return lon, lat, tris[:k] - 1             # to zero-based


def build_dual(lon, lat, tris):
    """Median-dual cells, one per node, in node order.

    Returns node_coords (M,2), element_conn (N,maxc) one-based with -1 fill,
    and num_element_conn (N,).
    """
    N, T = lon.size, len(tris)
    xyz = ll2xyz(lon, lat)

    # unique undirected edges, encoded as lo*N + hi so np.unique is 1-D
    pairs = np.concatenate([tris[:, [0, 1]], tris[:, [1, 2]], tris[:, [2, 0]]])
    code = pairs.min(axis=1) * N + pairs.max(axis=1)
    del pairs
    ucode, einv = np.unique(code, return_inverse=True)
    del code
    E = ucode.size
    ecount = np.bincount(einv, minlength=E)
    ea, eb = ucode // N, ucode % N
    del ucode
    # edge id of each triangle's (0,1), (1,2), (2,0) edge -- replaces a dict
    # with one entry per edge, which costs hundreds of MB on large meshes
    etri = einv.reshape(3, T).T.astype(np.int64)
    del einv

    # dual vertices: [0,E) edge midpoints, [E,E+T) centroids, then boundary nodes
    mid_lon, mid_lat = xyz2ll(xyz[ea] + xyz[eb])
    cen_lon, cen_lat = xyz2ll(xyz[tris[:, 0]] + xyz[tris[:, 1]] + xyz[tris[:, 2]])

    bedge = ecount == 1
    bnode = np.zeros(N, dtype=bool)
    bnode[ea[bedge]] = True
    bnode[eb[bedge]] = True
    del ea, eb, bedge
    bidx = np.where(bnode)[0]
    bslot = np.full(N, -1, dtype=np.int64)
    bslot[bidx] = E + T + np.arange(bidx.size)

    node_coords = np.empty((E + T + bidx.size, 2))
    node_coords[:E, 0], node_coords[:E, 1] = mid_lon, mid_lat
    node_coords[E:E + T, 0], node_coords[E:E + T, 1] = cen_lon, cen_lat
    node_coords[E + T:, 0], node_coords[E + T:, 1] = lon[bidx], lat[bidx]
    del mid_lon, mid_lat, cen_lon, cen_lat

    # CSR adjacency of triangles per node
    corner_tri = np.repeat(np.arange(T), 3)
    order = np.argsort(tris.ravel(), kind="stable")
    cn, ct = tris.ravel()[order], corner_tri[order]
    start = np.searchsorted(cn, np.arange(N), side="left")
    stop = np.searchsorted(cn, np.arange(N), side="right")
    del corner_tri, order, cn

    # corner count is exact up front: two per incident triangle, plus the
    # midpoint and the node itself when the fan is open. Lets element_conn be
    # preallocated instead of built from a list of N Python lists.
    ntri = (stop - start).astype(np.int32)
    nconn = 2 * ntri + 2 * bnode.astype(np.int32)
    nconn[ntri == 0] = 0
    maxc = int(nconn.max())
    element_conn = np.full((N, maxc), -1, dtype=np.int32)

    dual_xyz = ll2xyz(node_coords[:, 0], node_coords[:, 1])
    verts = np.empty(maxc, dtype=np.int64)

    def edge_of(t, u, w):
        """Edge id of the edge of triangle t joining nodes u and w."""
        a, b, c = tris[t]
        if (u == a and w == b) or (u == b and w == a):
            return etri[t, 0]
        if (u == b and w == c) or (u == c and w == b):
            return etri[t, 1]
        return etri[t, 2]

    pinched = []
    for P in range(N):
        s0, s1 = start[P], stop[P]
        if s1 == s0:
            continue
        tp = ct[s0:s1]
        others = {}
        for t in tp:
            a, b, c = tris[t]
            others[int(t)] = [int(v) for v in (a, b, c) if v != P]
        setP = set(int(t) for t in tp)
        isb = bool(bnode[P])

        # iterate tp, not setP: set order is arbitrary and would make the
        # starting corner -- and so elementConn -- vary between runs
        o0 = t0 = None
        if isb:
            for t in tp:
                for o in others[int(t)]:
                    if ecount[edge_of(int(t), P, o)] == 1:
                        o0, t0 = o, int(t)
                        break
                if o0 is not None:
                    break
        if o0 is None:
            t0 = int(tp[0]); o0 = others[t0][0]

        n = 0
        cur_o, cur_t, seen = o0, t0, set()
        while cur_t is not None and cur_t not in seen:
            seen.add(cur_t)
            verts[n] = edge_of(cur_t, P, cur_o); n += 1
            verts[n] = E + cur_t; n += 1
            a, b = others[cur_t]
            nxt = a if b == cur_o else b
            cand = [t for t in setP if t != cur_t and nxt in others[t]]
            cur_t, cur_o = (cand[0] if cand else None), nxt
        if isb:
            last = next(t for t in tp if cur_o in others[int(t)])
            verts[n] = edge_of(int(last), P, cur_o); n += 1
            verts[n] = int(bslot[P]); n += 1

        if len(seen) != tp.size:
            pinched.append(P)

        # orientation on the sphere: counter-clockwise seen from outside means
        # sum(v_i x v_i+1) points the same way as the cell centre. A planar
        # lon/lat test would misbehave for a cell enclosing a pole.
        v = dual_xyz[verts[:n]]
        if np.dot(np.cross(v, np.roll(v, -1, axis=0)).sum(axis=0), xyz[P]) < 0:
            element_conn[P, :n] = verts[n - 1::-1] + 1
        else:
            element_conn[P, :n] = verts[:n] + 1

    if pinched:
        sys.exit(f"ERROR: {len(pinched)} node(s) have triangles in more than one fan "
                 f"(e.g. node {pinched[0] + 1}); a single ESMF element cannot "
                 f"represent them. Fix the mesh at these pinch points.")

    return node_coords, element_conn, nconn


def cell_areas(node_coords, element_conn, nconn, center_lon, center_lat,
               chunk=200000):
    """Spherical areas (steradians), fanned from the cell centre.

    Median-dual cells are star-shaped about their WW3 node but not convex, so
    fanning from a polygon vertex makes the fan triangles overlap; with unsigned
    triangle areas those overlaps add and concave cells come out too large.
    Fanning from the node is exact, and the dual then tiles the triangulation.

    Short cells are padded by repeating the last corner, which contributes
    zero-area triangles and closes the ring correctly. Computed in chunks: the
    (ncell, ncorner, 3) intermediates are several hundred MB per million cells.
    """
    n = element_conn.shape[0]
    out = np.empty(n)
    for lo in range(0, n, chunk):
        hi = min(lo + chunk, n)
        idx = element_conn[lo:hi].copy()
        for i in range(hi - lo):
            idx[i, nconn[lo + i]:] = idx[i, nconn[lo + i] - 1]
        idx -= 1
        xv = np.concatenate([center_lon[lo:hi, None], node_coords[idx, 0],
                             node_coords[idx, 0][:, :1]], axis=1)
        yv = np.concatenate([center_lat[lo:hi, None], node_coords[idx, 1],
                             node_coords[idx, 1][:, :1]], axis=1)
        out[lo:hi] = cell_area_rad(xv, yv)
    return out


def write_esmf(path, node_coords, element_conn, nconn, center_coords,
               title=None, source=None):
    ds = Dataset(path, "w", format="NETCDF3_64BIT_DATA")
    ds.createDimension("nodeCount", node_coords.shape[0])
    ds.createDimension("elementCount", element_conn.shape[0])
    ds.createDimension("maxNodePElement", element_conn.shape[1])
    ds.createDimension("coordDim", 2)
    v = ds.createVariable("nodeCoords", "f8", ("nodeCount", "coordDim"))
    v.units = "degrees"; v[:] = node_coords
    v = ds.createVariable("elementConn", "i4", ("elementCount", "maxNodePElement"),
                          fill_value=-1)
    v.long_name = "Node indices that define the element connectivity"
    v.start_index = np.int32(1); v[:] = element_conn
    v = ds.createVariable("numElementConn", "i4", ("elementCount",))
    v.long_name = "Number of nodes per element"; v[:] = nconn
    v = ds.createVariable("centerCoords", "f8", ("elementCount", "coordDim"))
    v.units = "degrees"; v[:] = center_coords
    ds.createVariable("elementMask", "i4", ("elementCount",))[:] = 1
    ds.gridType = "unstructured mesh"
    ds.date_created = datetime.now().isoformat()
    if title:
        ds.title = title
    if source:
        ds.inputFile = source
    ds.close()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("msh")
    ap.add_argument("out")
    ap.add_argument("--title")
    a = ap.parse_args()

    lon, lat, tris = parse_gmsh(a.msh)
    print(f"parsed {lon.size} nodes, {len(tris)} triangles")
    node_coords, element_conn, nconn = build_dual(lon, lat, tris)
    centers = np.column_stack([lon, lat])
    print(f"dual: nodeCount={node_coords.shape[0]} elementCount={element_conn.shape[0]} "
          f"corners {nconn.min()}-{nconn.max()}")

    # The dual must tile the triangulation exactly, so the two areas agree.
    # This is the real correctness check: it catches concave cells fanned the
    # wrong way, dropped triangles and mis-ordered corners alike.
    xyz = ll2xyz(lon, lat)
    primal = _tri_area(xyz[tris[:, 0]], xyz[tris[:, 1]], xyz[tris[:, 2]]).sum()
    areas = cell_areas(node_coords, element_conn, nconn, lon, lat)
    dual = areas.sum()
    rel = abs(dual - primal) / primal
    print(f"area: primal {primal:.7f} sr, dual {dual:.7f} sr, rel diff {rel:.3e} "
          f"({dual / (4.0 * np.pi):.4f} of the sphere)")
    if rel > 1e-8:
        sys.exit("ERROR: dual cells do not tile the triangulation")
    if (areas <= 0).any():
        sys.exit(f"ERROR: {int((areas <= 0).sum())} degenerate cells")

    write_esmf(a.out, node_coords, element_conn, nconn, centers, a.title, a.msh)
    print(f"wrote {a.out}")

    with Dataset(a.out) as chk:
        d = np.abs(chk["centerCoords"][:] - centers).max()
    print(f"check: centerCoords vs .msh nodes, max diff = {d:g}")
    if d != 0.0:
        sys.exit("ERROR: centerCoords do not match the input node order")


if __name__ == "__main__":
    main()
