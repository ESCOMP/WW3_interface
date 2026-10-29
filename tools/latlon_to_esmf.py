#!/usr/bin/env python3
"""Build an ESMF mesh from the lon/lat metadata of a regular lat-lon file.

CDEPS needs an ESMF mesh for every stream, but a forcing file carries only its
1-D longitude/latitude centres. This derives the cell edges from those centres
and writes the matching mesh, with one element per grid cell in lat-major order
-- the order the data flattens to, and the order CDEPS assumes.

Cells are quadrilaterals bounded by great circles. A row of cells that reaches
a pole keeps a full ring of nodes there rather than collapsing onto one: the
cells are then zero-width at the pole, which is what ESMF needs to be able to
build the mesh dual and regrid bilinearly.

Usage:
    latlon_to_esmf.py <in.nc> <out_esmf.nc> [--lon-name L] [--lat-name L]
"""

import argparse
import os
import sys

import numpy as np
from netCDF4 import Dataset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ww3_msh_to_esmf import write_esmf, cell_area_rad, ll2xyz  # noqa: E402

LON_NAMES = ("longitude", "lon", "x", "XLONG")
LAT_NAMES = ("latitude", "lat", "y", "XLAT")


def find_coord(ds, explicit, candidates, what):
    if explicit:
        if explicit not in ds.variables:
            raise SystemExit(f"ERROR: no variable {explicit!r} in the file")
        return explicit
    for name in candidates:
        if name in ds.variables and ds.variables[name].ndim == 1:
            return name
    raise SystemExit(f"ERROR: no 1-D {what} variable found; pass --{what}-name")


def edges_from_centres(c):
    """Cell interfaces midway between centres, with the ends extrapolated."""
    e = np.empty(c.size + 1)
    e[1:-1] = 0.5 * (c[:-1] + c[1:])
    e[0] = c[0] - 0.5 * (c[1] - c[0])
    e[-1] = c[-1] + 0.5 * (c[-1] - c[-2])
    return e


def build_mesh(lon, lat, extend_to_poles="auto"):
    """Nodes, connectivity and centres for the grid these centres describe."""
    nlon, nlat = lon.size, lat.size
    lon_e = edges_from_centres(lon)
    lat_e = edges_from_centres(lat)

    # a grid that wraps shares its first and last meridian, so drop the duplicate
    periodic = abs((lon_e[-1] - lon_e[0]) - 360.0) < 1e-6
    ncol = nlon if periodic else nlon + 1

    # never let the extrapolated end edges run past the pole
    lat_e = np.clip(lat_e, -90.0, 90.0)

    # A grid that wraps in longitude and whose cells already span all but a
    # sliver of the 180 degrees of latitude is meant to be global: the sliver
    # is an artifact of extrapolating the end cell from its neighbour, which
    # under-reaches on a Gaussian grid. Stretch the end cells onto the poles,
    # because a mesh that leaves the caps uncovered yields undefined values
    # there for anything mapped from it. Judged against the whole sphere, not
    # against the end cell: a coarse regional grid's half-cell is large, so
    # comparing the two is not a reliable signal.
    lat_span = abs(lat_e[0] - lat_e[-1])
    reaches_poles = periodic and lat_span >= 0.98 * 180.0
    if extend_to_poles == "yes" or (extend_to_poles == "auto" and reaches_poles):
        north = 0 if lat_e[0] > lat_e[-1] else -1
        extended = abs(abs(lat_e[north]) - 90.0) > 1e-9 or abs(abs(lat_e[-1 - north]) - 90.0) > 1e-9
        lat_e[north], lat_e[-1 - north] = 90.0, -90.0
        if extended:
            print(f"extended the end cells onto the poles "
                  f"(latitude span was {lat_span:.4f} of 180)")
    pole = np.isclose(np.abs(lat_e), 90.0, atol=1e-9)
    if pole[1:-1].any():
        raise SystemExit("ERROR: an interior latitude edge sits on a pole")

    # Every latitude edge row carries a full ring of nodes, a row on a pole
    # included. Collapsing a polar row onto one shared node looks tidier but
    # ESMF then cannot build the mesh dual -- that node's dual cell would be
    # made of coincident element centres -- and bilinear regridding fails.
    # CAM's fv *_polemod meshes keep the ring for the same reason.
    nnode = (nlat + 1) * ncol
    node = np.empty((nnode, 2))
    node[:, 0] = np.tile(lon_e[:ncol], nlat + 1)
    node[:, 1] = np.repeat(lat_e, ncol)

    def nodes_of(edge_row, col):
        """Global node indices for column indices on given latitude edge rows."""
        return edge_row * ncol + (col % ncol)

    jj, ii = np.meshgrid(np.arange(nlat), np.arange(nlon), indexing="ij")
    jj, ii = jj.ravel(), ii.ravel()
    descending = lat[0] > lat[-1]
    south, north = (jj + 1, jj) if descending else (jj, jj + 1)

    # SW, SE, NE, NW -- counter-clockwise seen from outside the sphere
    corners = np.stack([nodes_of(south, ii), nodes_of(south, ii + 1),
                        nodes_of(north, ii + 1), nodes_of(north, ii)], axis=1)

    conn = corners
    nconn = np.full(corners.shape[0], corners.shape[1], dtype=np.int32)

    centre = np.stack([np.broadcast_to(lon, (nlat, nlon)).ravel(),
                       np.repeat(lat, nlon)], axis=1)
    return node, conn.astype(np.int32) + 1, nconn, centre, periodic


def polygon_areas(node, conn, nconn, chunk=200000):
    """Great-circle area of each cell, in steradians."""
    out = np.empty(conn.shape[0])
    for lo in range(0, conn.shape[0], chunk):
        hi = min(lo + chunk, conn.shape[0])
        idx = conn[lo:hi].copy()
        for k in range(hi - lo):
            idx[k, nconn[lo + k]:] = idx[k, nconn[lo + k] - 1]  # pad, zero area
        idx -= 1
        out[lo:hi] = cell_area_rad(node[idx, 0], node[idx, 1])
    return out


def orientations(node, conn, nconn, centre, chunk=200000):
    """Positive where the ring winds counter-clockwise seen from outside."""
    out = np.empty(conn.shape[0])
    cxyz = ll2xyz(centre[:, 0], centre[:, 1])
    for lo in range(0, conn.shape[0], chunk):
        hi = min(lo + chunk, conn.shape[0])
        idx = conn[lo:hi].copy()
        for k in range(hi - lo):
            idx[k, nconn[lo + k]:] = idx[k, nconn[lo + k] - 1]
        idx -= 1
        v = ll2xyz(node[idx, 0], node[idx, 1])
        nrm = np.cross(v, np.roll(v, -1, axis=1)).sum(axis=1)
        out[lo:hi] = (nrm * cxyz[lo:hi]).sum(axis=1)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src")
    ap.add_argument("out")
    ap.add_argument("--lon-name")
    ap.add_argument("--lat-name")
    ap.add_argument("--title")
    ap.add_argument("--extend-to-poles", choices=("auto", "yes", "no"), default="auto",
                    help="stretch the end cells onto the poles; auto-detects a global grid")
    args = ap.parse_args()

    with Dataset(args.src) as ds:
        lon_name = find_coord(ds, args.lon_name, LON_NAMES, "lon")
        lat_name = find_coord(ds, args.lat_name, LAT_NAMES, "lat")
        lon = np.asarray(ds[lon_name][:], dtype=float)
        lat = np.asarray(ds[lat_name][:], dtype=float)

    print(f"{lon_name}: {lon.size} points, {lon[0]:.4f} .. {lon[-1]:.4f}")
    print(f"{lat_name}: {lat.size} points, {lat[0]:.4f} .. {lat[-1]:.4f}")

    node, conn, nconn, centre, periodic = build_mesh(lon, lat, args.extend_to_poles)
    print(f"periodic in longitude: {periodic}")
    npolar = int((np.abs(centre[:, 1]) >= 90.0 - 1e-9).sum())
    print(f"nodes {node.shape[0]}, elements {conn.shape[0]} "
          f"({npolar} cells centred on a pole)")

    if conn.shape[0] != lon.size * lat.size:
        raise SystemExit("ERROR: element count does not match the grid")

    area = polygon_areas(node, conn, nconn)
    total, sphere = area.sum(), 4.0 * np.pi
    rel = abs(total - sphere) / sphere
    print(f"total area {total:.10f} sr vs 4pi {sphere:.10f} (rel {rel:.2e})")
    if rel > 1e-8:
        raise SystemExit("ERROR: cells do not tile the sphere; if this is a global grid "
                         "whose end cells stop short of the poles, pass --extend-to-poles yes")
    if area.min() <= 0:
        raise SystemExit(f"ERROR: {int((area <= 0).sum())} cell(s) have zero area")

    orient = orientations(node, conn, nconn, centre)
    if (orient <= 0).any():
        raise SystemExit(f"ERROR: {int((orient <= 0).sum())} cell(s) wind clockwise")

    if conn.max() > node.shape[0] or conn[conn > 0].min() < 1:
        raise SystemExit("ERROR: connectivity indexes outside the node list")

    title = args.title or (f"{lon.size}x{lat.size} regular lat-lon mesh from "
                           f"{os.path.basename(args.src)}")
    write_esmf(args.out, node, conn, nconn, centre, title=title,
               source=os.path.abspath(args.src))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
