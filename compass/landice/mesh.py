import os
import re
import sys
import time
from collections import deque
from shutil import copyfile

import json

import jigsawpy
import meshio
import mpas_tools.io
import numpy as np
import xarray
from geometric_features import FeatureCollection, GeometricFeatures
from mpas_tools.io import write_netcdf
from mpas_tools.landice.projections import projections as landice_projections
from mpas_tools.logging import check_call
from mpas_tools.mesh.conversion import convert, cull
from mpas_tools.mesh.creation import build_planar_mesh
from mpas_tools.mesh.creation.sort_mesh import sort_mesh
from netCDF4 import Dataset
from pyproj import Transformer
from shapely.geometry import shape
from shapely.vectorized import contains
from skimage.measure import find_contours
from scipy.interpolate import NearestNDInterpolator, interpn
from scipy import ndimage

def mpas_flood_fill(seed_mask, grow_mask, cellsOnCell, nEdgesOnCell,
                    grow_iters=sys.maxsize):
    """
    Flood-fill for mpas meshes using mpas cells.

    Parameters
    ----------
    seed_mask : numpy.ndarray
        Integer array of locations from which to flood fill
        0 = invalid, 1 = valid

    grow_mask : numpy.ndarray
        Integer array of locations valid for growing into
        0 = invalid, 1 = valid

    cellsOnCell : numpy.ndarray
        cellsOnCell array from the mpas mesh

    nEdgesOnCell : numpy.ndarray
        nEdgesOnCell array from the mpas mesh

    grow_iters : integer
        optional argument limiting the number of iterations
        over which to extend the mask

    Returns
    -------
    keep_mask : numpy.ndarray
        mask calculated by the flood fill routine,
        where cells connected to seed_mask
        are 1 and everything else is 0.
    """

    iter = 0
    keep_mask = seed_mask.copy()
    n_mask_cells = keep_mask.sum()
    for iter in range(grow_iters):
        mask_ind = np.nonzero(keep_mask == 1)[0]
        print(f'iter={iter}, keep_mask size={keep_mask.sum()}')
        new_keep_mask = keep_mask.copy()
        for iCell in mask_ind:
            neighs = cellsOnCell[iCell, :nEdgesOnCell[iCell]] - 1
            neighs = neighs[neighs >= 0]  # drop garbage cell
            for jCell in neighs:
                if grow_mask[jCell] == 1:
                    new_keep_mask[jCell] = 1
        keep_mask = new_keep_mask.copy()
        n_mask_cells_new = keep_mask.sum()
        if n_mask_cells_new == n_mask_cells:
            break
        n_mask_cells = n_mask_cells_new
        iter += 1
    return keep_mask


def gridded_flood_fill(field, iStart=None, jStart=None):
    """
    Generic flood-fill routine to create mask of connected elements
    in the desired input array (field) from a gridded dataset. This
    is generally used to remove glaciers and ice-fields that are not
    connected to the ice sheet. Note that there may be more efficient
    algorithms.

    Parameters
    ----------
    field : numpy.ndarray
        Array from gridded dataset to use for flood-fill.
        Usually ice thickness.

    iStart : int
        x index from which to start flood fill for field.
        Defaults to the center x coordinate.

    jStart : int
        y index from which to start flood fill.
        Defaults to the center y coordinate.

    Returns
    -------
    flood_mask : numpy.ndarray
        mask calculated by the flood fill routine,
        where cells connected to the ice sheet (or main feature)
        are 1 and everything else is 0.
    """

    sz = field.shape
    searched_mask = np.zeros(sz)
    flood_mask = np.zeros(sz)
    if iStart is None and jStart is None:
        iStart = sz[0] // 2
        jStart = sz[1] // 2
    flood_mask[iStart, jStart] = 1

    neighbors = np.array([[1, 0], [-1, 0], [0, 1], [0, -1]])

    lastSearchList = np.ravel_multi_index([[iStart], [jStart]],
                                          sz, order='F')

    cnt = 0
    while len(lastSearchList) > 0:
        cnt += 1
        newSearchList = np.array([], dtype='i')

        for iii in range(len(lastSearchList)):
            [i, j] = np.unravel_index(lastSearchList[iii], sz, order='F')
            # search neighbors
            for n in neighbors:
                ii = min(i + n[0], sz[0] - 1)  # don't go out of bounds
                jj = min(j + n[1], sz[1] - 1)  # subscripts to neighbor
                # only consider unsearched neighbors
                if searched_mask[ii, jj] == 0:
                    searched_mask[ii, jj] = 1  # mark as searched

                    if field[ii, jj] > 0.0:
                        flood_mask[ii, jj] = 1  # mark as ice
                        # add to list of newly found  cells
                        newSearchList = np.append(newSearchList,
                                                  np.ravel_multi_index(
                                                      [[ii], [jj]], sz,
                                                      mode='clip',
                                                      order='F')[0])
        lastSearchList = newSearchList

    return flood_mask


def set_rectangular_geom_points_and_edges(xmin, xmax, ymin, ymax):
    """
    Set node and edge coordinates to pass to
    :py:func:`mpas_tools.mesh.creation.build_mesh.build_planar_mesh()`.

    Parameters
    ----------
    xmin : int or float
        Left-most x-coordinate in region to mesh

    xmax : int or float
        Right-most x-coordinate in region to mesh

    ymin : int or float
        Bottom-most y-coordinate in region to mesh

    ymax : int or float
        Top-most y-coordinate in region to mesh

    Returns
    -------
    geom_points : jigsawpy.jigsaw_msh_t.VERT2_t
        xy node coordinates to pass to ``build_planar_mesh()``

    geom_edges : jigsawpy.jigsaw_msh_t.EDGE2_t
        xy edge coordinates between nodes to pass to ``build_planar_mesh()``
    """

    geom_points = np.array([  # list of xy "node" coordinates
        ((xmin, ymin), 0),
        ((xmax, ymin), 0),
        ((xmax, ymax), 0),
        ((xmin, ymax), 0)],
        dtype=jigsawpy.jigsaw_msh_t.VERT2_t)

    geom_edges = np.array([  # list of "edges" between nodes
        ((0, 1), 0),
        ((1, 2), 0),
        ((2, 3), 0),
        ((3, 0), 0)],
        dtype=jigsawpy.jigsaw_msh_t.EDGE2_t)

    return geom_points, geom_edges


def clip_mesh_to_bounding_box(mask_ds, base_ds, bounding_box):
    """
    Set cells to culled if they lay outside the bounding box

    Parameters
    ----------
    mask_ds: xr.Dataset
        mask dataset, generated by ``compute_mpas_region_mask``
    base_ds: xr.Dataset
        unculled mesh dataset
    bounding_box: list of 4 ints
        Bounding box [x_min, x_max, y_min, y_max] to cull mesh outside of

    Returns
    -------
    mask_ds: xarray.Dataset
        mask dataset with updated masks based on bounding box
    """

    if len(bounding_box) != 4:
        msg = f"bounding box must be len 4, instead is len {len(bounding_box)}"
        raise ValueError(msg)

    x_min, x_max, y_min, y_max = bounding_box

    if (x_max < x_min) or (y_max < y_min):
        msg = "Bounding box must be ordered: [x_min, x_max, y_min, y_max]"
        msg += (
            f"\n x_max < x_min ({x_max:.2f} < {x_min:.2f})"
            if x_max < x_min else ""
        )
        msg += (
            f"\n y_max < y_min ({y_max:.2f} < {y_min:.2f})"
            if y_max < y_min else ""
        )
        raise ValueError(msg)

    for loc in ["Cell", "Edge", "Vertex"]:
        mask_var = f"region{loc}Masks"

        if mask_var not in mask_ds:
            continue

        mask = (
            (base_ds[f"x{loc}"] < x_min) |
            (base_ds[f"x{loc}"] > x_max) |
            (base_ds[f"y{loc}"] < y_min) |
            (base_ds[f"y{loc}"] > y_max)
        )

        mask_ds[mask_var] = xarray.where(mask, 0, mask_ds[mask_var])

    return mask_ds


def rasterize_geojson_mask(geojson_file, x1, y1, projection):
    """
    Rasterize a single-polygon geojson region (defined in lon/lat) onto a
    structured x1/y1 grid, so the region can be intersected with other
    gridded fields (e.g. ``dist_to_edge``) before contour extraction,
    rather than being applied as a separate cull pass on the mesh.

    Parameters
    ----------
    geojson_file : str
        Path to a geojson file containing a single Polygon feature in
        lon/lat (CRS84).

    x1 : numpy.ndarray
        x coordinates of the structured grid (planar projection).

    y1 : numpy.ndarray
        y coordinates of the structured grid (planar projection).

    projection : str
        Key into ``mpas_tools.landice.projections.projections`` giving the
        proj4 string of the planar projection used by ``x1``/``y1``.

    Returns
    -------
    mask : numpy.ndarray, shape (len(y1), len(x1)), dtype bool
        ``True`` where the grid point falls inside the geojson polygon.
    """

    with open(geojson_file) as fh:
        geojson_dict = json.load(fh)

    polygon_lonlat = shape(geojson_dict['features'][0]['geometry'])

    transformer = Transformer.from_crs(
        'EPSG:4326', landice_projections[projection], always_xy=True)
    lon, lat = polygon_lonlat.exterior.coords.xy
    x_proj, y_proj = transformer.transform(np.asarray(lon), np.asarray(lat))

    polygon_planar = shape({
        'type': 'Polygon',
        'coordinates': [list(zip(x_proj, y_proj))],
    })

    xx, yy = np.meshgrid(x1, y1)
    return contains(polygon_planar, xx, yy)


def set_cell_width(self, section_name, thk, bed=None, vx=None, vy=None,
                   dist_to_edge=None, dist_to_grounding_line=None,
                   flood_fill_iStart=None, flood_fill_jStart=None):
    """
    Set cell widths based on settings in config file to pass to
    :py:func:`mpas_tools.mesh.creation.build_mesh.build_planar_mesh()`.

    Parameters
    ----------
    section_name : str
        Section of the config file from which to read parameters. The
        following options to be set in the given config section:
        ``levels``, ``x_min``, ``x_max``, ``y_min``, ``y_max``,
        ``min_spac``, ``max_spac``, ``high_log_speed``, ``low_log_speed``,
        ``high_dist``, ``low_dist``, ``high_dist_bed``, ``low_dist_bed``,
        ``high_bed``, ``low_bed``, ``cull_distance``, ``use_speed``,
        ``use_dist_to_edge``, ``use_dist_to_grounding_line``, and ``use_bed``.
        See the Land-Ice Framework section of the Users or Developers guide
        for more information about these options and their uses.

    thk : numpy.ndarray
        Ice thickness field from gridded dataset,
        usually after trimming to flood fill mask

    bed : numpy.ndarray
        Bed topography from gridded dataset

    vx : numpy.ndarray, optional
        x-component of ice velocity from gridded dataset,
        usually after trimming to flood fill mask. Can be set to ``None``
        if ``use_speed == False`` in config file.

    vy : numpy.ndarray, optional
        y-component of ice velocity from gridded dataset,
        usually after trimming to flood fill mask. Can be set to ``None``
        if ``use_speed == False`` in config file.

    dist_to_edge : numpy.ndarray, optional
        Distance from each cell to ice edge, calculated in separate function.
        Can be set to ``None`` if ``use_dist_to_edge == False`` in config file
        and you do not want to set large ``cell_width`` where cells will be
        culled anyway, but this is not recommended.

    dist_to_grounding_line : numpy.ndarray, optional
        Distance from each cell to grounding line, calculated in separate
        function.  Can be set to ``None`` if
        ``use_dist_to_grounding_line == False`` in config file.

    flood_fill_iStart : int, optional
        x-index location to start flood-fill when using bed topography

    flood_fill_jStart : int, optional
        y-index location to start flood-fill when using bed topography

    Returns
    -------
    cell_width : numpy.ndarray
        Desired width of MPAS cells based on mesh desnity functions to pass to
        :py:func:`mpas_tools.mesh.creation.build_mesh.build_planar_mesh()`.
    """

    logger = self.logger
    section = self.config[section_name]

    # Get config inputs for cell spacing functions
    min_spac = section.getfloat('min_spac')
    max_spac = section.getfloat('max_spac')
    high_log_speed = section.getfloat('high_log_speed')
    low_log_speed = section.getfloat('low_log_speed')
    high_dist = section.getfloat('high_dist')
    low_dist = section.getfloat('low_dist')
    high_dist_bed = section.getfloat('high_dist_bed')
    low_dist_bed = section.getfloat('low_dist_bed')
    low_bed = section.getfloat('low_bed')
    high_bed = section.getfloat('high_bed')

    # convert km to m
    cull_distance = section.getfloat('cull_distance') * 1.e3

    # Cell spacing function based on union of masks
    if section.get('use_bed') == 'True':
        logger.info('Using bed elevation for spacing.')
        if flood_fill_iStart is not None and flood_fill_jStart is not None:
            logger.info('calling gridded_flood_fill to find \
                        bedTopography <= low_bed connected to the ocean.')
            tic = time.time()
            # initialize mask to low bed topography
            in_mask = (bed <= low_bed)
            # Do not let flood fill reach further than high_dist_bed into
            # the ice sheet interior.
            in_mask[np.logical_and(
                thk > 0, dist_to_grounding_line >= high_dist_bed)] = 0
            low_bed_mask = gridded_flood_fill(in_mask,
                                              iStart=flood_fill_iStart,
                                              jStart=flood_fill_jStart)
            toc = time.time()
            logger.info(f'Flood fill finished in {toc - tic} seconds.')
        # Use a logistics curve for bed topography spacing.
        k = 0.05  # This works well, but could try other values
        spacing_bed = min_spac + (max_spac - min_spac) / (1.0 + np.exp(
            -k * (bed - np.mean([high_bed, low_bed]))))
        # We only want bed topography to influence spacing within high_dist_bed
        # from the ice margin. In the region between high_dist_bed and
        # low_dist_bed, use a linear ramp to damp influence of bed topo.
        spacing_bed[dist_to_grounding_line >= low_dist_bed] = (
            (1.0 - (dist_to_grounding_line[
                dist_to_grounding_line >= low_dist_bed] -
                low_dist_bed) / (high_dist_bed - low_dist_bed)) *
            spacing_bed[dist_to_grounding_line >= low_dist_bed] +
            (dist_to_grounding_line[dist_to_grounding_line >=
                                    low_dist_bed] - low_dist_bed) /
            (high_dist_bed - low_dist_bed) * max_spac)
        spacing_bed[dist_to_grounding_line >= high_dist_bed] = max_spac
        if flood_fill_iStart is not None and flood_fill_jStart is not None:
            spacing_bed[low_bed_mask == 0] = max_spac
            # Do one more flood fill to eliminate isolated pockets
            # of high resolution that were separated when we set
            # spacing_bed[dist_to_grounding_line >= high_dist_bed] = max_spac
            in_mask2 = (bed <= low_bed)
            in_mask2[np.logical_and(
                thk > 0, spacing_bed > (2. * min_spac))] = 0
            low_bed_mask2 = gridded_flood_fill(in_mask2,
                                               iStart=flood_fill_iStart,
                                               jStart=flood_fill_jStart)
            spacing_bed[low_bed_mask2 == 0] = max_spac
    else:
        spacing_bed = max_spac * np.ones_like(thk)

    # Make cell spacing function mapping from log speed to cell spacing
    if section.get('use_speed') == 'True':
        logger.info('Using speed for cell spacing')
        speed = (vx ** 2 + vy ** 2) ** 0.5
        lspd = np.log10(speed)
        spacing_speed = np.interp(lspd, [low_log_speed, high_log_speed],
                                  [max_spac, min_spac], left=max_spac,
                                  right=min_spac)

        # Clean up where we have missing velocities. These are usually nans
        # or the default netCDF _FillValue of ~10.e36
        missing_data_mask = np.logical_or(
            np.logical_or(np.isnan(vx), np.isnan(vy)),
            np.logical_or(np.abs(vx) > 1.e5,
                          np.abs(vy) > 1.e5))
        spacing_speed[missing_data_mask] = max_spac
        logger.info(f'Found {np.sum(missing_data_mask)} points in input '
                    f'dataset with missing velocity values. Setting '
                    f'velocity-based spacing to maximum value.')

        spacing_speed[thk == 0.0] = min_spac
    else:
        spacing_speed = max_spac * np.ones_like(thk)

    # Make cell spacing function mapping from distance to ice edge
    if section.get('use_dist_to_edge') == 'True':
        logger.info('Using distance to ice edge for cell spacing')
        spacing_edge = np.interp(dist_to_edge, [low_dist, high_dist],
                                 [min_spac, max_spac], left=min_spac,
                                 right=max_spac)
        spacing_edge[thk == 0.0] = min_spac
    else:
        spacing_edge = max_spac * np.ones_like(thk)

    # Make cell spacing function mapping from distance to grounding line
    if section.get('use_dist_to_grounding_line') == 'True':
        logger.info('Using distance to grounding line for cell spacing')
        spacing_gl = np.interp(dist_to_grounding_line, [low_dist, high_dist],
                               [min_spac, max_spac], left=min_spac,
                               right=max_spac)
        spacing_gl[thk == 0.0] = min_spac
    else:
        spacing_gl = max_spac * np.ones_like(thk)

    # Merge cell spacing methods
    cell_width = max_spac * np.ones_like(thk)
    for width in [spacing_bed, spacing_speed, spacing_edge, spacing_gl]:
        cell_width = np.minimum(cell_width, width)

    # Set large cell_width in areas we are going to cull anyway (speeds up
    # whole process). Use 3x the cull_distance to avoid this affecting
    # cell size in the final mesh. There may be a more rigorous way to set
    # that distance.
    if dist_to_edge is not None:
        # Interior cells have negative dist_to_edge, so the threshold
        # comparison alone is sufficient to identify far exterior cells.
        mask = dist_to_edge > np.abs(3. * cull_distance)
        logger.info('Setting cell_width in outer regions to max_spac '
                    f'for {mask.sum()} cells')
        cell_width[mask] = max_spac

    return cell_width

def writeContoursToVtk(contour, file):
    """Writes a VTK mesh file from the given contour
       (chain of vertices that form edges)."""
    def add_zero_z_coord(pt):
        return np.concatenate((pt, [0]))
    points = []
    line_indices = []
    first_point = 0
    for i in range(len(contour) - 1):
        points.append(add_zero_z_coord(contour[i]))
        line_indices.append([first_point + i, first_point + i + 1])
    points.append(add_zero_z_coord(contour[-1]))
    first_point = len(points)

    cells = [("line", line_indices)]

    mesh = meshio.Mesh(points, cells)

    mesh.write(file)

def writeToVtk(points, edges, filename, point_data=None,
               point_data_name='vertexId'):
    """Writes a VTK mesh file from the given contour (points and edges).

    Parameters
    ----------
    points : array-like, shape (N+1, 2)
        Closed contour; the last point (repeat of first) is dropped.
    edges : list of (int, int)
        Local point-index pairs forming polyline edges.
    filename : str
        Output file path.
    point_data : array-like of int, optional
        Per-point scalar values (length N, matching the written points).
        Written as VTK POINT_DATA SCALARS with dtype int.
    point_data_name : str, optional
        VTK scalar name for ``point_data`` (default ``'vertexId'``).
    """
    n = len(points[:-1])
    lines = ['# vtk DataFile Version 3.0\n',
             f'{filename} written by compass/landice/mesh.py\n',
             'ASCII\n',
             'DATASET POLYDATA\n\n',
             f'POINTS {n} float\n']

    for pt in points[:-1]:
        lines.append(f'{pt[0]} {pt[1]} 0.0\n')
    lines.append(f'\nLINES {len(edges)} {len(edges)*3}\n')
    for edge in edges:
        lines.append(f'2 {edge[0]} {edge[1]}\n')
    if point_data is not None:
        lines.append(f'\nPOINT_DATA {n}\n')
        lines.append(f'SCALARS {point_data_name} int 1\n')
        lines.append('LOOKUP_TABLE default\n')
        for val in point_data:
            lines.append(f'{val}\n')
    with open(filename, "w") as f:
        f.writelines(lines)

def writeTrianglesToVtk(points, triangles, filename, boundary_order,
                        point_data_name='boundaryOrder'):
    """Writes a VTK triangle mesh, marking boundary vertices and their
    ordering around the boundary loop.

    Parameters
    ----------
    points : array-like, shape (N, 2)
        xy coordinates of all mesh vertices.
    triangles : array-like, shape (M, 3)
        Local point-index triples forming each triangle.
    filename : str
        Output file path.
    boundary_order : array-like of int, shape (N,)
        Per-point scalar: -1 for interior (non-boundary) vertices, or the
        0-based position of the vertex in the ordered (CW or CCW) boundary
        loop.
    point_data_name : str, optional
        VTK scalar name for ``boundary_order``
        (default ``'boundaryOrder'``).
    """
    n = len(points)
    lines = ['# vtk DataFile Version 3.0\n',
             f'{filename} written by compass/landice/mesh.py\n',
             'ASCII\n',
             'DATASET POLYDATA\n\n',
             f'POINTS {n} float\n']

    for pt in points:
        lines.append(f'{pt[0]} {pt[1]} 0.0\n')
    lines.append(f'\nPOLYGONS {len(triangles)} {len(triangles) * 4}\n')
    for tri in triangles:
        lines.append(f'3 {tri[0]} {tri[1]} {tri[2]}\n')
    lines.append(f'\nPOINT_DATA {n}\n')
    lines.append(f'SCALARS {point_data_name} int 1\n')
    lines.append('LOOKUP_TABLE default\n')
    for val in boundary_order:
        lines.append(f'{val}\n')
    with open(filename, "w") as f:
        f.writelines(lines)


def remove_triangles(contour, name, debug=False):
    """ find sequences of four points where the first
    and last point are the same and remove the second
    and third points """

    # assert that there is a loop
    assert (contour[0] == contour[-1]).all()
    assert (len(contour) > 4)
    fn_name = "remove_triangles"
    tic = time.time()
    clean = []
    a = 0
    b = 1
    c = 2
    d = 3
    clean.append(contour[a])
    last_idx = len(contour) - 1
    tri_removed = 0
    while d <= last_idx:
        if np.allclose(contour[a], contour[d]):
            if debug:
                print("triangle found near pt {:.4E} {:.4E}"
                      .format(contour[a][0], contour[a][1]))
            tri_removed += 1
            # skip middle points
            a = d
            b = a + 1
            c = b + 1
            d = c + 1
        else:
            clean.append(contour[b])
            a += 1
            b += 1
            c += 1
            d += 1
    # add the last points
    while c <= last_idx:
        clean.append(contour[c])
        c += 1
    # close the loop if it isn't already
    # i.e., if the last edge was removed
    if not (clean[0] == clean[-1]).all():
        clean.append(clean[0])
    toc = time.time()
    print("{} {} removed {} triangles"
          .format(fn_name, name, tri_removed))
    print("{} {} done: {:.2f} seconds\n"
          .format(fn_name, name, toc - tic))
    writeContoursToVtk(clean,
                       "{}PrimaryContourNoTri.vtk".format(name))
    return clean

def remove_coincident_edges(contour, name, debug=False):

    class Edge:
        def __init__(self, v0, v1):
            assert (len(v0) == 2)
            assert (len(v1) == 2)
            self.v0 = v0
            self.v1 = v1

    def are_edges_coincident(e1, e2):
        assert isinstance(e1, Edge)
        assert isinstance(e2, Edge)
        if np.allclose(e1.v0, e2.v0) and np.allclose(e1.v1, e2.v1):
            return True
        elif np.allclose(e1.v0, e2.v1) and np.allclose(e1.v1, e2.v0):
            return True
        else:
            return False

    # assert that there is a loop
    assert (contour[0] == contour[-1]).all()
    assert (len(contour) > 3)
    coincident_count = 0
    fn_name = "remove_coincident_edges"
    tic = time.time()
    clean = []
    left = 0
    middle = 1
    right = 2
    clean.append(contour[left])
    last_idx = len(contour) - 1
    while left < last_idx and middle <= last_idx and right <= last_idx:
        e1 = Edge(contour[left], contour[middle])
        e2 = Edge(contour[middle], contour[right])
        if are_edges_coincident(e1, e2):
            if debug:
                print("coincident edges found near pt {:.4E} {:.4E}"
                      "left {} mid {} right {}"
                      .format(contour[middle][0], contour[middle][1],
                              left, middle, right))
            # skip both edges
            coincident_count += 1
            left += 2
            middle += 2
            right += 2
        else:
            clean.append(contour[middle])
            left += 1
            middle += 1
            right += 1
    # close the loop if it isn't already
    # i.e., if the last edge was removed
    if not (clean[0] == clean[-1]).all():
        clean.append(clean[0])
    toc = time.time()
    print("{} {} removed {} edges"
          .format(fn_name, name, coincident_count))
    print("{} {} done: {:.2f} seconds\n"
          .format(fn_name, name, toc - tic))
    writeContoursToVtk(clean,
                       "{}PrimaryContourClean.vtk".format(name))
    return clean

def collapse_small_edges(contour, small, name, debug=False):
    # assert that there is a loop
    assert (contour[0] == contour[-1]).all()

    collapsed_count = 0
    tic = time.time()
    collapsed = []
    current = 0
    next = 1
    collapsed.append(contour[current])
    last_idx = len(contour) - 1
    while current < last_idx and next <= last_idx:
        pt = contour[current]
        next_pt = contour[next]
        dist = np.linalg.norm(pt - next_pt)
        if dist <= small:
            if debug:
                print("pt[{}] {} {} pt[{}] {} {}".
                      format(current, pt[0], pt[1],
                             next, next_pt[0], next_pt[1]))
                print("points are {} apart, which is less than {}".
                      format(dist, small))
            next += 1  # advance 'next' for the next evaluation
            collapsed_count += 1
        else:
            collapsed.append(next_pt)
            current = next
            next += 1
    # close the loop if it isn't already
    # i.e., if the last edge was collapsed
    if not (collapsed[0] == collapsed[-1]).all():
        collapsed.append(collapsed[0])
    toc = time.time()
    print("collapse_small_edges {} removed {} edges"
          .format(name, collapsed_count))
    print("collapse_small_edges {} done: {:.2f} seconds\n"
          .format(name, toc - tic))
    writeContoursToVtk(collapsed,
                       "{}PrimaryContourCollapsed.vtk".format(name))
    return collapsed

def rho_i():
    return np.double(910.0)


def rho_w():
    return np.double(1028.0)

def get_phi(thk, topg, x, y):
    print("get phi start\n")
    assert (thk.shape == (len(y), len(x)))
    tic = time.time()

    max_distance = max(max(x), max(y))

    phi = np.where(np.isclose(thk, 0, atol=1e-10),
                   max_distance,
                   rho_i() * thk + rho_w() * topg)
    toc = time.time()
    print("get_phi done: {:.2f} seconds\n".format(toc - tic))
    return phi

def get_ice_surface_height(phi, topg, thk):
    # upper ice surface height where ice is floating
    s_floating = (1 - (rho_i() / rho_w())) * thk
    s_grounded = topg + thk
    # set the height to s_floating where phi < 0 and s_grounded otherwise
    s_height = np.where(phi < 0, s_floating, s_grounded)
    return s_height


def transform_max_contour(contours, x, y, name):
    max_contour = max(contours, key=len)
    max_contour_len = len(max_contour)
    print("max sized contour lenth {}\n".format(max_contour_len))
    # transform the contour points back to the original coordinate system
    cell_size = x[1] - x[0]  # assumed constant and equal in x and y
    min_x = np.min(x)
    min_y = np.min(y)
    transformed_pts = [(pt * cell_size) + (min_x, min_y) for pt in max_contour]
    writeContoursToVtk(transformed_pts, "{}Contours.vtk".format(name))
    return transformed_pts

def extract_contour(field, x, y, name):
    """ the field is expected to be either 0 or 1 at
    each grid point """

    assert np.all((field == 0) | (field == 1))
    assert (field.shape == (len(y), len(x)))
    tic = time.time()
    contours = find_contours(field.T, 0.5)
    contours_xform = transform_max_contour(contours, x, y, name)
    toc = time.time()
    print("{} extract_contour done: {:.2f} seconds\n".format(name, toc - tic))
    return contours_xform

def get_dist_to_edge_and_gl(self, thk, topg, x, y,
                            section_name, window_size=None):
    """
    Calculate distance from each point to ice edge and grounding line,
    to be used in mesh density functions in
    :py:func:`compass.landice.mesh.set_cell_width()`. In future development,
    this should be updated to use a faster package such as ``scikit-fmm``.

    Parameters
    ----------
    thk : numpy.ndarray
        Ice thickness field from gridded dataset,
        usually after trimming to flood fill mask

    topg : numpy.ndarray
        Bed topography field from gridded dataset

    x : numpy.ndarray
        x coordinates from gridded dataset

    y : numpy.ndarray
        y coordinates from gridded dataset

    section_name : str
        Section of the config file from which to read parameters. The
        following options to be set in the given config section:
        ``levels``, ``x_min``, ``x_max``, ``y_min``, ``y_max``,
        ``min_spac``, ``max_spac``, ``high_log_speed``, ``low_log_speed``,
        ``high_dist``, ``low_dist``, ``high_dist_bed``, ``low_dist_bed``,
        ``high_bed``, ``low_bed``, ``cull_distance``, ``use_speed``,
        ``use_dist_to_edge``, ``use_dist_to_grounding_line``, and ``use_bed``.
        See the Land-Ice Framework section of the Users or Developers guide
        for more information about these options and their uses.

    window_size : int or float
        Size (in meters) of a search 'box' (one-directional) to use
        to calculate the distance from each cell to the ice margin.
        Bigger number makes search slower, but if too small, the transition
        zone could get truncated. We usually want this calculated as the
        maximum of ``high_dist`` and ``high_dist_bed``, but there may be cases
        in which it is useful to set it manually. However, it should never be
        smaller than either ``high_dist`` or ``high_dist_bed``.

    Returns
    -------
    dist_to_edge : numpy.ndarray
        Distance from each cell to the ice edge

    dist_to_grounding_line : numpy.ndarray
        Distance from each cell to the grounding line
    """
    logger = self.logger
    section = self.config[section_name]
    tic = time.time()

    high_dist = float(section.get('high_dist'))
    high_dist_bed = float(section.get('high_dist_bed'))

    if window_size is None:
        window_size = max(high_dist, high_dist_bed)
    elif window_size < min(high_dist, high_dist_bed):
        logger.info('WARNING: window_size was set to a value smaller'
                    ' than high_dist and/or high_dist_bed. Resetting'
                    f' window_size to {max(high_dist, high_dist_bed)},'
                    ' which is max(high_dist, high_dist_bed)')
        window_size = max(high_dist, high_dist_bed)

    dx = x[1] - x[0]  # assumed constant and equal in x and y
    nx = len(x)
    ny = len(y)
    sz = thk.shape

    # Create masks to define ice edge and grounding line
    neighbors = np.array([[1, 0], [-1, 0], [0, 1], [0, -1],
                          [1, 1], [-1, 1], [1, -1], [-1, -1]])

    ice_mask = thk > 0.0
    grounded_mask = thk > (-1028.0 / 910.0 * topg)
    margin_mask = np.zeros(sz, dtype='i')
    grounding_line_mask = np.zeros(sz, dtype='i')

    for n in neighbors:
        not_ice_mask = np.logical_not(np.roll(ice_mask, n, axis=[0, 1]))
        margin_mask = np.logical_or(margin_mask, not_ice_mask)

        not_grounded_mask = np.logical_not(np.roll(grounded_mask,
                                                   n, axis=[0, 1]))
        grounding_line_mask = np.logical_or(grounding_line_mask,
                                            not_grounded_mask)

    # where ice exists and neighbors non-ice locations
    margin_mask = np.logical_and(margin_mask, ice_mask)
    # optional - plot mask
    # plt.pcolor(margin_mask); plt.show()

    # Calculate dist to margin and grounding line
    [XPOS, YPOS] = np.meshgrid(x, y)
    dist_to_edge = np.zeros(sz)
    dist_to_grounding_line = np.zeros(sz)

    d = int(np.ceil(window_size / dx))
    rng = np.arange(-1 * d, d, dtype='i')
    max_dist = float(d) * dx

    # just look over areas with ice
    # ind = np.where(np.ravel(thk, order='F') > 0)[0]
    ind = np.where(np.ravel(thk, order='F') >= 0)[0]  # do it everywhere
    for iii in range(len(ind)):
        [i, j] = np.unravel_index(ind[iii], sz, order='F')

        irng = i + rng
        jrng = j + rng

        # only keep indices in the grid
        irng = irng[np.nonzero(np.logical_and(irng >= 0, irng < ny))]
        jrng = jrng[np.nonzero(np.logical_and(jrng >= 0, jrng < nx))]

        dist_to_here = ((XPOS[np.ix_(irng, jrng)] - x[j]) ** 2 +
                        (YPOS[np.ix_(irng, jrng)] - y[i]) ** 2) ** 0.5

        dist_to_here_edge = dist_to_here.copy()
        dist_to_here_grounding_line = dist_to_here.copy()

        dist_to_here_edge[margin_mask[np.ix_(irng, jrng)] == 0] = max_dist
        dist_to_here_grounding_line[grounding_line_mask
                                    [np.ix_(irng, jrng)] == 0] = max_dist

        dist_to_edge[i, j] = dist_to_here_edge.min()
        dist_to_grounding_line[i, j] = dist_to_here_grounding_line.min()

    toc = time.time()
    logger.info('compass.landice.mesh.get_dist_to_edge_and_gl() took {:0.2f} '
                'seconds'.format(toc - tic))

    return dist_to_edge, dist_to_grounding_line


def build_cell_width(self, section_name, gridded_dataset,
                     flood_fill_start=[None, None]):
    """
    Determine MPAS mesh cell size based on user-defined density function.

    Parameters
    ----------
    section_name : str
        Section of the config file from which to read parameters. The
        following options to be set in the given config section:
        ``levels``, ``x_min``, ``x_max``, ``y_min``, ``y_max``,
        ``min_spac``, ``max_spac``, ``high_log_speed``, ``low_log_speed``,
        ``high_dist``, ``low_dist``, ``high_dist_bed``, ``low_dist_bed``,
        ``high_bed``, ``low_bed``, ``cull_distance``, ``use_speed``,
        ``use_dist_to_edge``, ``use_dist_to_grounding_line``, and ``use_bed``.
        See the Land-Ice Framework section of the Users or Developers guide
        for more information about these options and their uses.

    gridded_dataset : str
        name of NetCDF file used to define cell spacing

    flood_fill_start : list of ints
        ``i`` and ``j`` indices used to define starting location for flood
        fill. Most cases will use ``[None, None]``, which will just start the
        flood fill in the center of the gridded dataset.

    Returns
    -------
    cell_width : numpy.ndarray
        Desired width of MPAS cells based on mesh desnity functions to pass to
        :py:func:`mpas_tools.mesh.creation.build_mesh.build_planar_mesh()`.

    x1 : float
        x coordinates from gridded dataset

    y1 : float
        y coordinates from gridded dataset

    geom_points : jigsawpy.jigsaw_msh_t.VERT2_t
        xy node coordinates to pass to ``build_planar_mesh()``

    geom_edges : jigsawpy.jigsaw_msh_t.EDGE2_t
        xy edge coordinates between nodes to pass to ``build_planar_mesh()``

    flood_mask : numpy.ndarray
        mask calculated by the flood fill routine,
        where cells connected to the ice sheet (or main feature)
        are 1 and everything else is 0.

    dist_to_edge : numpy.ndarray
        Distance from each grid point to the ice margin (m), on the
        same grid as the gridded dataset.
    """

    section = self.config[section_name]
    # get needed fields from gridded dataset
    f = Dataset(gridded_dataset, 'r')
    f.set_auto_mask(False)  # disable masked arrays

    x1 = f.variables['x1'][:]
    y1 = f.variables['y1'][:]
    thk = f.variables['thk'][0, :, :]
    topg = f.variables['topg'][0, :, :]
    vx = f.variables['vx'][0, :, :]
    vy = f.variables['vy'][0, :, :]

    f.close()

    # Get bounds defined by user, or use bound of gridded dataset
    bnds = [np.min(x1), np.max(x1), np.min(y1), np.max(y1)]
    bnds_options = ['x_min', 'x_max', 'y_min', 'y_max']
    for index, option in enumerate(bnds_options):
        bnd = section.get(option)
        if bnd != 'None':
            bnds[index] = float(bnd)

    geom_points, geom_edges = set_rectangular_geom_points_and_edges(*bnds)

    # Remove ice not connected to the ice sheet.
    flood_mask = gridded_flood_fill(thk)
    thk[flood_mask == 0] = 0.0
    vx[flood_mask == 0] = 0.0
    vy[flood_mask == 0] = 0.0

    # Calculate distance from each grid point to ice edge
    # and grounding line, for use in cell spacing functions.
    distToEdge, distToGL = get_dist_to_edge_and_gl(
        self, thk, topg, x1,
        y1, section_name=section_name)

    print("using distToEdge for contour extraction\n")
    distToEdge_ff = gridded_flood_fill(distToEdge)
    edge_contour = extract_contour(distToEdge_ff, x1, y1, "edge")
    edge_coarsened_contour = collapse_small_edges(edge_contour,
                                                  small=500, name="edge")
    edge_nocoin_contour = remove_coincident_edges(edge_coarsened_contour,
                                                  name="edge")
    edge_notri_contour = remove_triangles(edge_nocoin_contour, name="edge")

    # Write the ice-margin contour and the domain bounding box as separate
    # nested contours (inner and outer, respectively) for generate2dModel,
    # rather than merging them into a single combined contour.
    bbox_points = [pt[0] for pt in geom_points]
    bbox_points.append(bbox_points[0])  # close the loop
    bbox_edges = [pt[0] for pt in geom_edges]
    writeToVtk(np.array(bbox_points), bbox_edges, "bbox.vtk")
    n_edge = len(edge_notri_contour) - 1  # last point duplicates the first
    edge_edges = [(i, (i + 1) % n_edge) for i in range(n_edge)]
    writeToVtk(edge_notri_contour, edge_edges, "edge.vtk")

    # Sign distToEdge: interior (flood_mask==1) is negative, exterior positive.
    # Must come after distToEdge_ff, which requires unsigned values as barriers.
    distToEdge[flood_mask == 1] *= -1

    phi = get_phi(thk, topg, x1, y1)
    s_height = get_ice_surface_height(phi, topg, thk)
    s_height_ff = gridded_flood_fill(s_height)

    out_file = Dataset("s_height.nc", 'w')
    out_file.createDimension("x1", len(x1))
    out_file.createDimension("y1", len(y1))
    x1_var = out_file.createVariable("x1", "f8", ("x1"))
    x1_var[:] = x1[:]
    y1_var = out_file.createVariable("y1", "f8", ("y1"))
    y1_var[:] = y1[:]
    thk_var = out_file.createVariable("thk", "f8", ("y1", "x1"))
    thk_var[:, :] = thk[:, :]
    phi_var = out_file.createVariable("phi", "f8", ("y1", "x1"))
    phi_var[:, :] = phi[:, :]
    distToEdge_var = out_file.createVariable("distToEdge", "f8", ("y1", "x1"))
    distToEdge_var[:, :] = distToEdge[:, :]
    s_var = out_file.createVariable("s_height", "f8", ("y1", "x1"))
    s_var[:, :] = s_height[:, :]
    f_var = out_file.createVariable("s_height_ff", "f8", ("y1", "x1"))
    f_var[:, :] = s_height_ff[:, :]
    out_file.close()
    # Set cell widths based on mesh parameters set in config file
    cell_width = set_cell_width(self, section_name=section_name,
                                thk=thk, bed=topg, vx=vx, vy=vy,
                                dist_to_edge=distToEdge,
                                dist_to_grounding_line=distToGL,
                                flood_fill_iStart=flood_fill_start[0],
                                flood_fill_jStart=flood_fill_start[1])

    return (cell_width.astype('float64'), x1.astype('float64'),
            y1.astype('float64'), geom_points, geom_edges, flood_mask,
            distToEdge)


# Per-triangle tags used by the seam partition below: a triangle is
# either preserved verbatim from the input MPAS mesh or generated by
# Simmetrix. ``producer`` is also the VTK scalar name written for these,
# so the tags can be inspected per triangle in ParaView.
PRODUCER_PRESERVED = 1
PRODUCER_SIMMETRIX = 2


def boundary_edge_loops(dsMesh):
    """
    Walk the mesh's boundary edges into closed loops.

    Walking *edges* rather than cells is deliberate. A boundary cell does
    not necessarily own exactly two boundary edges: on the GIS dehorned
    mesh 418 cells own three and 27 own four, while 796 own just one. Any
    walk that steps cell-to-cell assuming a degree of two mis-traverses at
    those cells. Every boundary *edge*, by contrast, has exactly two
    neighbours -- one through each endpoint -- so the boundary edge graph
    is a disjoint union of simple cycles and the walk is unambiguous.

    Parameters
    ----------
    dsMesh : xarray.Dataset
        MPAS mesh dataset containing ``cellsOnEdge`` and ``verticesOnEdge``.

    Returns
    -------
    loops : list of numpy.ndarray
        One array of 0-based boundary-edge indices per closed loop. A
        domain with interior voids yields more than one loop.
    """
    cellsOnEdge = dsMesh['cellsOnEdge'].values
    verticesOnEdge = dsMesh['verticesOnEdge'].values

    bnd = np.where(np.any(cellsOnEdge == 0, axis=1))[0]
    if len(bnd) == 0:
        return []

    # dual vertex -> incident boundary edges
    vtx_edges = {}
    for e in bnd:
        for v in verticesOnEdge[e]:
            vtx_edges.setdefault(int(v), []).append(int(e))

    loops = []
    unvisited = set(int(e) for e in bnd)
    while unvisited:
        start = min(unvisited)
        loop = [start]
        unvisited.discard(start)
        # arbitrarily pick one endpoint as the direction of travel
        prev_vtx = int(verticesOnEdge[start][0])
        curr = start
        while True:
            a, b = (int(v) for v in verticesOnEdge[curr])
            next_vtx = b if a == prev_vtx else a
            nxt = [e for e in vtx_edges[next_vtx]
                   if e != curr and e in unvisited]
            if not nxt:
                break
            curr = nxt[0]
            unvisited.discard(curr)
            loop.append(curr)
            prev_vtx = next_vtx
        loops.append(np.array(loop, dtype=np.int64))
    return loops


def get_ordered_boundary_cells(dsMesh):
    """
    Ordered boundary cells, as one open sequence per boundary loop.

    In MPAS's dual-mesh layout, ``xCell``/``yCell`` are the actual primal
    (triangular) mesh vertex (corner) points -- ``xVertex``/``yVertex``
    are the triangle circumcenters (dual/Voronoi vertices) -- so this
    traces the boundary of the surviving primal mesh in terms of its true
    corner points.

    Each loop is the sequence of owner cells along its boundary edge walk
    with consecutive duplicates collapsed, left *open* (the first cell is
    not repeated at the end). A cell owning several non-adjacent boundary
    edges legally appears more than once in a loop -- at a pinch point the
    boundary really does pass through the cell twice -- so callers must not
    assume the entries are unique.

    Parameters
    ----------
    dsMesh : xarray.Dataset
        MPAS mesh dataset containing ``cellsOnEdge`` and ``verticesOnEdge``.

    Returns
    -------
    loops : list of numpy.ndarray
        0-based cell indices, one array per boundary loop.
    """
    cellsOnEdge = dsMesh['cellsOnEdge'].values
    loops = []
    for edge_loop in boundary_edge_loops(dsMesh):
        c0 = cellsOnEdge[edge_loop, 0]
        c1 = cellsOnEdge[edge_loop, 1]
        owner = np.where(c0 == 0, c1, c0) - 1

        collapsed = [int(owner[0])]
        for c in owner[1:]:
            if int(c) != collapsed[-1]:
                collapsed.append(int(c))
        # the walk wraps, so drop a trailing repeat of the first cell
        while len(collapsed) > 1 and collapsed[-1] == collapsed[0]:
            collapsed.pop()
        loops.append(np.array(collapsed, dtype=np.int64))
    return loops


def get_cell_adjacency(dsMesh):
    """
    Per-cell neighbour lists (0-based) from ``cellsOnCell``/``nEdgesOnCell``.

    Parameters
    ----------
    dsMesh : xarray.Dataset
        MPAS mesh dataset.

    Returns
    -------
    adjacency : list of list of int
        ``adjacency[i]`` are the 0-based indices of cell ``i``'s neighbours.
    """
    cellsOnCell = dsMesh['cellsOnCell'].values
    nEdgesOnCell = dsMesh['nEdgesOnCell'].values
    return [[int(c) - 1 for c in cellsOnCell[i, :nEdgesOnCell[i]] if c > 0]
            for i in range(dsMesh.sizes['nCells'])]


def get_boundary_distance(dsMesh, adjacency=None):
    """
    Ring distance of every cell from the domain boundary: 0 for a boundary
    cell, 1 for its neighbours, and so on. Cells unreachable from the
    boundary get -1 (does not occur on a connected mesh).

    Parameters
    ----------
    dsMesh : xarray.Dataset
        MPAS mesh dataset.

    adjacency : list of list of int, optional
        Precomputed neighbour lists from
        :py:func:`get_cell_adjacency()`.

    Returns
    -------
    dist : numpy.ndarray
        Ring distance per cell, shape ``(nCells,)``.
    """
    nCells = dsMesh.sizes['nCells']
    adj = adjacency if adjacency is not None else get_cell_adjacency(dsMesh)
    cellsOnEdge = dsMesh['cellsOnEdge'].values

    dist = np.full(nCells, -1, dtype=np.int64)
    queue = deque()
    for e in np.where(np.any(cellsOnEdge == 0, axis=1))[0]:
        for c in cellsOnEdge[e]:
            if c > 0 and dist[int(c) - 1] != 0:
                dist[int(c) - 1] = 0
                queue.append(int(c) - 1)
    while queue:
        c = queue.popleft()
        for nb in adj[c]:
            if dist[nb] < 0:
                dist[nb] = dist[c] + 1
                queue.append(nb)
    return dist


class SeamPartition:
    """
    The seam/interface/interior partition of an MPAS mesh, and the
    per-triangle MPAS-preserved/Simmetrix classification it implies.

    Each polygon is reconstructed by ``MpasMeshConverter.x`` purely from
    the fan of incident triangles (``cellsOnVertex`` rows), and it closes
    only if that fan is a complete cycle (interior polygon) or a complete
    open fan terminated by two boundary edges (boundary polygon). So every
    fan must come from a single source -- either wholly preserved from the
    input MPAS mesh, or wholly generated by Simmetrix. Never mixed.

    This partition guarantees that by construction. Cells are classified by
    ring distance from the domain boundary:

    * **seam** -- within ``k`` rings of the boundary. Kept verbatim and
      handed to Simmetrix via ``MS_specifyFace``.
    * **interior** -- everything else. Simmetrix remeshes this freely.
    * **interface** -- the subset of seam cells having an interior
      neighbour, i.e. the innermost ring of the seam annulus.

    Only at an *interface* cell is a split fan legitimate: such a cell gets
    preserved triangles outboard and Simmetrix triangles inboard, and that
    is safe precisely because the two groups meet along the two interface
    edges the cell owns, closing into a single cycle.

    ``k=2`` is the operating point. At ``k=1`` every seam cell is also an
    interface cell -- the annulus is one ring thick with no insulating
    layer -- and MPAS-preserved and Simmetrix triangles interleave around
    interface fans (458 offending cells on the GIS dehorned mesh).
    ``k >= 2`` passes.

    Attributes
    ----------
    k : int
        Seam width in rings.

    dist : numpy.ndarray
        Ring distance of each cell from the boundary, shape ``(nCells,)``.

    is_seam, is_interface, is_interior : numpy.ndarray of bool
        Per-cell classification, shape ``(nCells,)``.

    cells_on_vertex : numpy.ndarray
        0-based ``cellsOnVertex``, with -1 for a null corner.

    producer : numpy.ndarray
        Per-triangle tag, shape ``(nVertices,)``: ``PRODUCER_PRESERVED``
        (kept verbatim from the input MPAS mesh) if every real corner is a
        seam cell, else ``PRODUCER_SIMMETRIX`` (generated by Simmetrix).
    """

    def __init__(self, dsMesh, k=2):
        self.k = k
        self.nCells = dsMesh.sizes['nCells']
        self.nVertices = dsMesh.sizes['nVertices']

        adj = get_cell_adjacency(dsMesh)
        self.adjacency = adj
        self.dist = get_boundary_distance(dsMesh, adj)

        # kept for the interface-contiguity check
        self._cell_xy = (dsMesh['xCell'].values, dsMesh['yCell'].values)
        cellsOnEdge = dsMesh['cellsOnEdge'].values
        self._boundary_cells = np.zeros(self.nCells, dtype=bool)
        for e in np.where(np.any(cellsOnEdge == 0, axis=1))[0]:
            for c in cellsOnEdge[e]:
                if c > 0:
                    self._boundary_cells[int(c) - 1] = True

        self.is_seam = (self.dist >= 0) & (self.dist < k)
        self.is_interior = ~self.is_seam

        self.is_interface = np.zeros(self.nCells, dtype=bool)
        for c in np.where(self.is_seam)[0]:
            if any(self.is_interior[nb] for nb in adj[c]):
                self.is_interface[c] = True

        # A triangle is preserved iff all of its real corners are seam
        # cells. Incomplete (null-corner) boundary rows qualify on their
        # real corners alone -- they are how MPAS terminates a boundary fan
        # and must travel with the preserved annulus.
        cov = dsMesh['cellsOnVertex'].values - 1
        self.cells_on_vertex = cov
        self.producer = np.full(self.nVertices, PRODUCER_SIMMETRIX,
                                dtype=np.int32)
        for v in range(self.nVertices):
            corners = [int(c) for c in cov[v] if c >= 0]
            if corners and all(self.is_seam[c] for c in corners):
                self.producer[v] = PRODUCER_PRESERVED

    @property
    def preserved_triangles(self):
        """Triangle (``nVertices``) indices kept verbatim from the input."""
        return np.where(self.producer == PRODUCER_PRESERVED)[0]

    @property
    def interior_triangles(self):
        """Triangle (``nVertices``) indices Simmetrix generates."""
        return np.where(self.producer == PRODUCER_SIMMETRIX)[0]

    def counts(self):
        """Summary counts, for logging."""
        return {
            'cells': int(self.nCells),
            'seam': int(self.is_seam.sum()),
            'interface': int(self.is_interface.sum()),
            'interior': int(self.is_interior.sum()),
            'triangles': int(self.nVertices),
            'preserved': int(len(self.preserved_triangles)),
            'generated': int(len(self.interior_triangles)),
        }

    def check_invariant(self):
        """
        Verify no incident-triangle fan is split between MPAS-preserved
        and Simmetrix triangles, except at interface cells where a split
        is expected by design.

        Returns
        -------
        problems : list of str
            Empty if the partition is sound.
        """
        problems = []
        tris_of_cell = {}
        for v in range(self.nVertices):
            for c in self.cells_on_vertex[v]:
                if c >= 0:
                    tris_of_cell.setdefault(int(c), []).append(v)

        for c in range(self.nCells):
            tris = tris_of_cell.get(c, [])
            if not tris:
                problems.append(f'cell {c}: no incident triangles')
                continue
            tags = {int(self.producer[v]) for v in tris}
            if len(tags) > 1 and not self.is_interface[c]:
                kind = ('interior' if self.is_interior[c]
                        else 'seam(non-interface)')
                problems.append(f'cell {c} ({kind}): fan split between '
                                'MPAS-preserved and Simmetrix triangles')

        # Every interior cell's fan must be wholly Simmetrix-generated;
        # otherwise the annulus does not actually separate the two.
        for c in np.where(self.is_interior)[0]:
            tris = tris_of_cell.get(c, [])
            if any(self.producer[v] == PRODUCER_PRESERVED for v in tris):
                problems.append(f'interior cell {c} has an MPAS-preserved '
                                'triangle')

        problems.extend(self._check_interface_contiguity(tris_of_cell))
        return problems

    def _check_interface_contiguity(self, tris_of_cell):
        """
        At an interface cell the fan is *expected* to be split, so the
        agreement test above is vacuous there -- which is exactly where
        the real defect lives. What must hold instead is that the two
        groups form **contiguous arcs** that meet: walking the fan around
        the cell, the MPAS-preserved triangles must be consecutive and the
        Simmetrix triangles must be consecutive, giving exactly two
        changes around the cycle (or zero, if one side is empty).

        Three or more changes means the two interleave, so they cannot be
        joined by a single pair of shared edges and the polygon will not
        close -- the failure this whole design exists to prevent.

        The fan is ordered here by angle about the cell centre, which is
        well-defined for the star-shaped Voronoi neighbourhoods MPAS
        produces.
        """
        problems = []
        for c in np.where(self.is_interface)[0]:
            tris = tris_of_cell.get(c, [])
            if len(tris) < 3:
                continue
            tags = [int(self.producer[v]) for v in tris]
            if len(set(tags)) < 2:
                continue  # wholly one producer: nothing to interleave

            order = np.argsort(self._fan_angles(c, tris))
            ring = [tags[i] for i in order]
            changes = sum(1 for i in range(len(ring))
                          if ring[i] != ring[(i + 1) % len(ring)])
            # a boundary cell's fan is an open arc, so it may show one
            # fewer transition than a closed interior fan
            limit = 3 if self._boundary_cells[c] else 2
            if changes > limit:
                problems.append(
                    f'interface cell {c}: producers interleave around the '
                    f'fan ({changes} transitions, expected <= {limit})')
        return problems

    def _fan_angles(self, cell, tris):
        """Angle of each incident triangle's centroid about the cell centre."""
        xc, yc = self._cell_xy
        cov = self.cells_on_vertex
        angles = []
        for v in tris:
            corners = [int(t) for t in cov[v] if t >= 0]
            cx = float(np.mean(xc[corners]))
            cy = float(np.mean(yc[corners]))
            angles.append(np.arctan2(cy - yc[cell], cx - xc[cell]))
        return np.array(angles)


def get_boundary_order(dsMesh, nCells):
    """
    Per-cell 0-based boundary traversal position, -1 for non-boundary cells.

    ``readVtkGeom`` (``modelGen2d.cc``) scatters the points of a
    ``boundary-triangles`` contour into contour slots by this value, and
    ``specifyBoundaryTriangleMesh`` exits if any slot is unfilled, so the
    values must cover ``0..nBoundary-1`` exactly once.

    Parameters
    ----------
    dsMesh : xarray.Dataset
        MPAS mesh dataset.

    nCells : int
        Number of cells, i.e. the length of the returned array.

    Returns
    -------
    order : numpy.ndarray
        Boundary traversal position per cell, shape ``(nCells,)``.

    Raises
    ------
    ValueError
        If the boundary is not a single simple loop of distinct cells,
        which is what the ``boundary-triangles`` VTK format can represent.
        A cell has only one ``boundaryOrder`` slot, so a boundary that
        passes through a cell twice (a pinch point) or splits into several
        loops (an interior void) cannot be expressed.
    """
    loops = get_ordered_boundary_cells(dsMesh)
    if len(loops) != 1:
        raise ValueError(
            f'boundary has {len(loops)} loops; the boundary-triangles VTK '
            'format represents a single closed contour only')
    loop = loops[0]
    unique = np.unique(loop)
    if len(unique) != len(loop):
        raise ValueError(
            f'boundary loop visits {len(loop) - len(unique)} cell(s) more '
            'than once (pinch point); each cell has only one boundaryOrder '
            'slot so this cannot be expressed in the VTK format')

    order = np.full(nCells, -1, dtype=np.int64)
    for pos, cell in enumerate(loop):
        order[int(cell)] = pos
    return order


def stitch_simmetrix_output(dsMesh, partition, sim_file, out_file,
                            seam_cells):
    r"""
    Combine ``generate2dModel``'s mesh with the input's boundary
    terminator rows, writing a complete minimal MPAS mesh for
    ``MpasMeshConverter.x``.

    ``generate2dModel`` emits every triangle of its mesh, but it cannot
    emit the incomplete (null-corner) ``cellsOnVertex`` rows: it meshes an
    *area*, so every face it makes has three real corners. Those rows are
    how MPAS terminates a boundary fan, and without them the boundary
    polygons do not close. This function appends them.

    In the primal mesh below, every point is a cell and every triangle is
    a ``cellsOnVertex`` row. ``A``, ``B`` and ``C`` are boundary cells;
    nothing is meshed above them::

            A ─────────── B ─────────── C
             \           / \           /
              \   T1    /   \   T2    /       T1 = (A, B, D)
               \       /     \       /        T2 = (B, C, D)
                \     /       \     /
                 \   /         \   /
                  \ /           \ /
                   D ─────────── D'

    ``B``'s fan is ``T1`` and ``T2``, which meet along ``B``-``D``. Past
    ``T1`` there is nothing beyond ``A``-``B``, and past ``T2`` nothing
    beyond ``B``-``C``: those two are *boundary edges*, MPAS edges whose
    ``cellsOnEdge`` has one null slot because only the inward side has a
    real cell. The fan is an open arc, so ``B``'s polygon has no way to
    close.

    MPAS terminates it with one extra ``cellsOnVertex`` row per boundary
    edge, naming that edge's two real cells and a null third corner::

            R1 = (A, B, null)     for boundary edge A-B
            R2 = (B, C, null)     for boundary edge B-C

    Each stands in for the triangle that would exist if the mesh continued
    outward. The null corner is what tells ``MpasMeshConverter.x`` the fan
    terminates rather than wraps.

    Appending a row grows ``nVertices``, so ``cellsOnVertex`` and the
    vertex coordinates ``xVertex``/``yVertex``/``zVertex`` all grow
    together; ``nCells`` is untouched. Every such row is preserved-tagged
    by construction, since its real corners are boundary cells and so are
    seam cells for any ``k >= 1``.

    Only the seam cells are handed to Simmetrix, so an input cell index is
    *not* an output cell index. The mapping between them rests on three
    things:

    * ``specifyBoundaryTriangleMesh`` tags each specified mesh vertex with
      its ``all_vertices`` index (``MS_specifyVertex(..., i)``,
      ``simModelGen2d.cc``), which is the seam cell's position in the
      point list :py:func:`fit_boundary_splines()` wrote;
    * ``numberUnspecifiedVertices`` renumbers mesher-created vertices to
      ``>= numAllVtx``, so a specified vertex keeps its position and
      everything Simmetrix added lands above the seam;
    * ``writeMeshSimToNetCDF`` writes *both* ``cellsOnVertex`` and
      ``xCell``/``yCell`` indexed by ``EN_id`` (``netcdfWriter.cc``).

    Together those make output cell ``i`` the same point as seam cell
    ``i`` for ``i < len(seam_cells)``, which is the map ``seam_cells``
    inverts below.

    Underneath those, ``M_write`` renumbers every mesh entity by iteration
    position, discarding the tags, so it must run *after* the netcdf is
    written.

    Parameters
    ----------
    dsMesh : xarray.Dataset
        The input MPAS mesh the partition was built from.

    partition : SeamPartition
        Supplies the preserved-triangle set and ``cells_on_vertex``.

    sim_file : str
        ``generate2dModel``'s NetCDF output.

    out_file : str
        Minimal MPAS mesh to write, for ``MpasMeshConverter.x``.

    seam_cells : numpy.ndarray
        The 0-based input cell indices written as points, in the order
        they were written, i.e. the inverse of the local index space the
        preserved triangles reference.

    Returns
    -------
    report : dict
        Counts, plus ``preserved_missing`` (preserved triangles absent
        from the Simmetrix output) and ``unmapped_cells``.
    """
    dsSim = xarray.open_dataset(sim_file)
    dsSim.load()

    x_sim = dsSim['xCell'].values
    y_sim = dsSim['yCell'].values
    cov_sim = dsSim['cellsOnVertex'].values  # 1-based, 0 = null

    n_sim = int(dsSim.sizes['nCells'])
    n_seam = len(seam_cells)

    # Input cell -> output cell. A seam cell keeps its position in the
    # point list; every other input cell has no counterpart, since
    # Simmetrix meshed that region afresh.
    mapping = np.full(dsMesh.sizes['nCells'], -1, dtype=np.int64)
    mapping[seam_cells] = np.arange(n_seam)

    # A specified vertex the mesher dropped would leave its seam cell
    # without an output counterpart, which breaks the preserved triangles
    # that reference it.
    unmapped = [int(c) for i, c in enumerate(seam_cells) if i >= n_sim]

    # Did the preserved triangles survive MS_specifyFace? Compare as
    # corner sets in the *output* index space.
    sim_tris = set()
    for row in cov_sim:
        corners = [int(c) - 1 for c in row if c > 0]
        if len(corners) == 3:
            sim_tris.add(frozenset(corners))

    cov_in = partition.cells_on_vertex
    preserved_missing = []
    preserved_expected = 0
    for v in partition.preserved_triangles:
        corners = [int(c) for c in cov_in[v] if c >= 0]
        if len(corners) != 3:
            continue  # a terminator row, appended below rather than meshed
        preserved_expected += 1
        mapped = [int(mapping[c]) for c in corners]
        if any(m < 0 for m in mapped) or frozenset(mapped) not in sim_tris:
            preserved_missing.append(int(v))

    # Append the terminator rows. Coordinates need no conversion: the
    # input is in metres and generate2dModel is told units=m, so it
    # converts to its internal km on the way in and back on the way out.
    extra_rows = []
    extra_x = []
    extra_y = []
    xv_in = dsMesh['xVertex'].values
    yv_in = dsMesh['yVertex'].values
    for v in range(partition.nVertices):
        row = cov_in[v]
        if not (row < 0).any():
            continue
        out_row = []
        for c in row:
            m = -1 if c < 0 else int(mapping[int(c)])
            out_row.append(0 if m < 0 else m + 1)  # 0 = null, MPAS is 1-based
        extra_rows.append(out_row)
        extra_x.append(float(xv_in[v]))
        extra_y.append(float(yv_in[v]))

    if extra_rows:
        cov_out = np.concatenate(
            [cov_sim, np.array(extra_rows, dtype=np.int32)])
        xv_out = np.concatenate([dsSim['xVertex'].values, np.array(extra_x)])
        yv_out = np.concatenate([dsSim['yVertex'].values, np.array(extra_y)])
    else:
        cov_out = cov_sim
        xv_out = dsSim['xVertex'].values
        yv_out = dsSim['yVertex'].values

    dsOut = xarray.Dataset()
    dsOut['xCell'] = xarray.DataArray(x_sim, dims=['nCells'])
    dsOut['yCell'] = xarray.DataArray(y_sim, dims=['nCells'])
    dsOut['zCell'] = xarray.DataArray(np.zeros_like(x_sim), dims=['nCells'])
    dsOut['xVertex'] = xarray.DataArray(xv_out, dims=['nVertices'])
    dsOut['yVertex'] = xarray.DataArray(yv_out, dims=['nVertices'])
    dsOut['zVertex'] = xarray.DataArray(
        np.zeros_like(xv_out), dims=['nVertices'])
    dsOut['cellsOnVertex'] = xarray.DataArray(
        cov_out.astype(np.int32), dims=['nVertices', 'vertexDegree'])
    dsOut['meshDensity'] = xarray.DataArray(
        np.ones(len(x_sim)), dims=['nCells'])
    dsOut.attrs['on_a_sphere'] = 'NO'
    dsOut.attrs['sphere_radius'] = 0.0
    dsOut.to_netcdf(out_file)

    report = {
        'sim_cells': n_sim,
        'sim_triangles': int(dsSim.sizes['nVertices']),
        'appended': len(extra_rows),
        'preserved_expected': preserved_expected,
        'preserved_found': preserved_expected - len(preserved_missing),
        'preserved_missing': preserved_missing,
        'unmapped_cells': unmapped,
    }
    dsSim.close()
    return report


def fit_boundary_splines(self, dsMesh, section):
    """
    Fit splines to the domain boundary contour of ``dsMesh`` and generate
    the final MALI mesh from the resulting geometric model. The mesh is
    built by Simmetrix's ``generate2dModel`` from the original ice-margin
    contour (inner, which carries the interior geometric model entities)
    and the preserved seam annulus of ``dsMesh`` (outer), so it already
    has the correct (dehorned/culled) domain shape and requires no
    further culling.

    The handoff is structured around an explicit *seam*: a band of cells
    hugging the domain boundary is kept verbatim from ``dsMesh`` and
    handed to Simmetrix via ``MS_specifyFace``, which holds it fixed,
    while Simmetrix remeshes the interior freely. See
    :py:class:`SeamPartition` for why the band is needed -- in short, a
    cell's Voronoi polygon is reconstructed by ``MpasMeshConverter.x``
    purely from the fan of triangles incident to that cell, so splitting
    one cell's fan between the two producers leaves its polygon unclosed.

    In MPAS's dual-mesh layout, ``xCell``/``yCell`` (``nCells``) are the
    primal (triangular) mesh corner points, and a triangle
    is indexed by ``nVertices`` with its 3 corner cells given by
    ``cellsOnVertex``; ``xVertex``/``yVertex`` are the triangle
    circumcenters (dual/Voronoi vertices).

    Parameters
    ----------
    self : compass step
        Provides ``logger`` and ``config``.

    dsMesh : xarray.Dataset
        MPAS mesh dataset supplying the domain boundary shape and the
        preserved seam annulus.

    section : configparser section
        Config section used to read the Simmetrix parameters, including
        ``simmetrix_seam_width`` (the seam width ``k`` in cell rings,
        default 2).

    Returns
    -------
    dsNewMesh : xarray.Dataset
        The new MPAS mesh produced from the boundary-fitted geometric
        model, with ``geomModelIdCell``, ``geomModelDimCell``,
        ``geomModelIdVertex``, and ``geomModelDimVertex`` classification
        arrays attached.
    """
    logger = self.logger

    xCell = dsMesh['xCell'].values
    yCell = dsMesh['yCell'].values
    nCells = dsMesh.sizes['nCells']

    # Partition the mesh into a preserved seam annulus hugging the domain
    # boundary and an interior region for Simmetrix to remesh. See
    # SeamPartition for why every cell's triangle fan must have a single
    # producer, and why k=2 is the operating point.
    seam_width = section.getint('simmetrix_seam_width', 2)
    partition = SeamPartition(dsMesh, k=seam_width)
    counts = partition.counts()
    logger.info(f'Seam partition (k={seam_width}): '
                f'{counts["seam"]} seam cells '
                f'({counts["interface"]} interface), '
                f'{counts["interior"]} interior cells; '
                f'{counts["preserved"]} preserved triangles, '
                f'{counts["generated"]} to be generated by Simmetrix')

    problems = partition.check_invariant()
    if problems:
        preview = '\n  '.join(problems[:10])
        raise RuntimeError(
            f'The seam partition at k={seam_width} violates the '
            f'single-producer invariant at {len(problems)} location(s); '
            'the resulting mesh would have unclosed cell polygons. First '
            f'few:\n  {preview}\n'
            'Increasing simmetrix_seam_width may resolve this.')

    # Write the boundary-triangles VTK that generate2dModel consumes
    # (readVtkGeom, modelGen2d.cc):
    #   POINTS     one per *seam* cell -- this is the `all_vertices` index
    #              space MS_specifyVertex tags each mesh vertex with
    #              (EN_id), and the space writeMeshSimToNetCDF maps back
    #              to output cell indices.
    #   POLYGONS   the preserved triangles, as triples of point indices
    #   POINT_DATA boundaryOrder, the 0-based boundary traversal position
    #
    # Only seam cells may appear: specifyBoundaryTriangleMesh specifies
    # every point in the file as a mesh vertex, and a vertex in the region
    # Simmetrix meshes that no specified face or contour connects is an
    # isolated point, which SurfaceMesher_execute rejects.
    seam_cells = np.where(partition.is_seam)[0]
    global_to_local = np.full(nCells, -1, dtype=np.int64)
    global_to_local[seam_cells] = np.arange(len(seam_cells))

    boundary_order = get_boundary_order(dsMesh, nCells)[seam_cells]

    # Preserved triangles only. Incomplete (null-corner) rows carry no
    # triangle to specify -- they are boundary *terminators*, and are
    # re-appended after meshing rather than handed to Simmetrix.
    cellsOnVertexZB = partition.cells_on_vertex
    preserved_tri = []
    for v in partition.preserved_triangles:
        corners = [int(c) for c in cellsOnVertexZB[v] if c >= 0]
        if len(corners) == 3:
            preserved_tri.append([int(global_to_local[c]) for c in corners])
    preserved_tri = np.array(preserved_tri, dtype=np.int64).reshape(-1, 3)

    # Every corner of a preserved triangle is a seam cell by construction
    # (that is what PRODUCER_PRESERVED means), so all of them must have
    # resolved to a local index.
    if preserved_tri.size and preserved_tri.min() < 0:
        raise RuntimeError(
            'a preserved triangle references a cell outside the seam')

    bdry_tri_vtk = 'boundary_cells.vtk'
    writeTrianglesToVtk(
        np.column_stack([xCell[seam_cells], yCell[seam_cells]]),
        preserved_tri, bdry_tri_vtk, boundary_order)
    logger.info(f'Wrote {bdry_tri_vtk}: {len(seam_cells)} seam-cell points, '
                f'{len(preserved_tri)} preserved triangles, '
                f'{int((boundary_order >= 0).sum())} boundary contour points')

    logger.info('Using Simmetrix generate2dModel to fit splines to the '
                'domain boundary')
    # Get Simmetrix parameters from config (with defaults for GIS)
    coincident_tol = section.getfloat('simmetrix_coincident_tolerance',
                                      1.0)
    angle_tol = section.getfloat('simmetrix_angle_tolerance', 100.0)
    oncurve_angle_tol = section.getfloat(
            'simmetrix_oncurve_angle_tolerance', 100.0)
    units = section.get('simmetrix_units', 'm')

    # Get path to generate2dModel binary
    # Default to 'generate2dModel' (assumes it's in PATH)
    # Can be overridden with full path in config
    simmetrix_binary = section.get('simmetrix_binary',
                                   'generate2dModel')

    # The original ice-margin contour (inner) and the preserved seam
    # annulus of the dehorned mesh (outer).
    edge_vtk = 'edge.vtk'
    output_prefix = 'boundary_contour'

    if not os.path.exists(edge_vtk):
        raise FileNotFoundError(
            f'Input file {edge_vtk} not found. Ensure '
            'build_cell_width() was called first to create this file.')

    # Both contours are required. generate2dModel maps order=0 to
    # features.inner and order=1 to features.outer, and runs createEdges
    # on each, so the inner (ice-margin) contour is where the interior
    # geometric model entities -- its splines, point and boundary
    # classification -- are defined. It also selects the two-loop form of
    # createFaces (simModelGen2d.cc), giving a face bounded by the outer
    # contour and the margin. Passing only the triangles would take the
    # hasSingleContour path and discard that inner model entirely.
    #
    # The boundary triangles must be the *outer* contour: features.outer
    # is what createMesh meshes and what writeMeshSimToNetCDF indexes by
    # EN_id, so the seam annulus has to travel in that slot.
    #
    # fail-if-cleaned: every point in the triangles contour is a cell
    # centre the partition depends on, and the triangle corner indices
    # reference those point positions. If cleaning drops a point, those
    # indices no longer mean what was written, so a silent removal would
    # corrupt the handoff rather than merely perturb it. Better to stop.
    args = [simmetrix_binary,
            '--contour', f'file={edge_vtk},order=0,units={units}',
            '--contour', f'file={bdry_tri_vtk},order=1,units={units},'
                         f'boundary-triangles,fail-if-cleaned',
            output_prefix,
            str(coincident_tol), str(angle_tol), str(oncurve_angle_tol),
            '1']  # createMesh = 1 (generate mesh)

    check_call(args, logger=logger)

    # generate2dModel writes a minimal MPAS mesh (boundary_contour.nc)
    # with geomModelIdCell/geomModelDimCell/geomModelIdVertex/
    # geomModelDimVertex classification arrays. Read those before
    # MpasMeshConverter.x, which does not preserve unrecognized variables.
    minimal_mesh_file = output_prefix + '.nc'
    dsMinimal = xarray.open_dataset(minimal_mesh_file)
    geom_id_cell = dsMinimal['geomModelIdCell'].values.copy()
    geom_dim_cell = dsMinimal['geomModelDimCell'].values.copy()
    geom_id_vertex = dsMinimal['geomModelIdVertex'].values.copy()
    geom_dim_vertex = dsMinimal['geomModelDimVertex'].values.copy()
    dsMinimal.close()

    # Re-attach the incomplete (null-corner) boundary rows, which
    # Simmetrix cannot produce, and verify the preserved triangles
    # survived MS_specifyFace.
    stitched_mesh_file = output_prefix + '_stitched.nc'
    report = stitch_simmetrix_output(dsMesh, partition, minimal_mesh_file,
                                     stitched_mesh_file, seam_cells)
    logger.info(f'Simmetrix produced {report["sim_cells"]} cells and '
                f'{report["sim_triangles"]} triangles; '
                f'{report["preserved_found"]}/'
                f'{report["preserved_expected"]} preserved triangles '
                f'survived, {report["appended"]} boundary terminator rows '
                'appended')
    if report['preserved_missing']:
        raise RuntimeError(
            f'{len(report["preserved_missing"])} preserved triangle(s) are '
            'absent from the Simmetrix output, so the seam annulus was not '
            'held fixed; the affected cells\' polygons will not close')
    if report['unmapped_cells']:
        raise RuntimeError(
            f'{len(report["unmapped_cells"])} seam cell(s) have no '
            'Simmetrix counterpart, so the EN_id index space shared by the '
            'two sides of the handoff was not preserved')

    logger.info('Converting boundary-fitted triangular mesh to MPAS mesh')
    converted_mesh_file = output_prefix + '_converted.nc'
    args = ['MpasMeshConverter.x', stitched_mesh_file, converted_mesh_file]
    check_call(args, logger=logger)

    dsNewMesh = xarray.open_dataset(converted_mesh_file)
    dsNewMesh.load()

    # Stitching appends the boundary terminator rows, so the per-triangle
    # classification arrays generate2dModel wrote are short by that many
    # entries. Those appended rows came from the input mesh rather than
    # from the geometric model, so they have no classification; pad with
    # -1 to keep the arrays dimensioned on nVertices.
    n_appended = report['appended']
    if n_appended > 0:
        geom_id_vertex = np.concatenate(
            [geom_id_vertex, np.full(n_appended, -1,
                                     dtype=geom_id_vertex.dtype)])
        geom_dim_vertex = np.concatenate(
            [geom_dim_vertex, np.full(n_appended, -1,
                                      dtype=geom_dim_vertex.dtype)])

    dsNewMesh['geomModelIdCell'] = xarray.DataArray(
        geom_id_cell, dims=['nCells'])
    dsNewMesh['geomModelDimCell'] = xarray.DataArray(
        geom_dim_cell, dims=['nCells'])
    dsNewMesh['geomModelIdVertex'] = xarray.DataArray(
        geom_id_vertex, dims=['nVertices'])
    dsNewMesh['geomModelDimVertex'] = xarray.DataArray(
        geom_dim_vertex, dims=['nVertices'])

    return dsNewMesh


def build_mali_mesh(self, cell_width, x1, y1, geom_points,
                    geom_edges, mesh_name, section_name,
                    gridded_dataset, projection, geojson_file=None,
                    cores=1, bounding_box=None, dist_to_edge=None):
    """
    Create the MALI mesh based on final cell widths determined by
    :py:func:`compass.landice.mesh.build_cell_width()`, using Jigsaw or
    Simmetrix and MPAS-Tools functions. Culls the mesh based on config
    options, interpolates all available fields from the gridded dataset to
    the MALI mesh using the bilinear method, and marks domain boundaries as
    Dirichlet cells.

    Parameters
    ----------
    cell_width : numpy.ndarray
        Desired width of MPAS cells calculated by :py:func:`build_cell_width()`
        based on mesh density functions define in :py:func:`set_cell_width()`
        to pass to
        :py:func:`mpas_tools.mesh.creation.build_mesh.build_planar_mesh()`.

    x1 : float
        x coordinates from gridded dataset

    y1 : float
        y coordinates from gridded dataset

    geom_points : jigsawpy.jigsaw_msh_t.VERT2_t
        xy node coordinates to pass to ``build_planar_mesh()``

    geom_edges : jigsawpy.jigsaw_msh_t.EDGE2_t
        xy edge coordinates between nodes to pass to ``build_planar_mesh()``

    mesh_name : str
        Filename to be used for final MALI NetCDF mesh file.

    section_name : str
        Section of the config file from which to read parameters. The
        following options to be set in the given config section:
        ``levels``, ``x_min``, ``x_max``, ``y_min``, ``y_max``,
        ``min_spac``, ``max_spac``, ``high_log_speed``, ``low_log_speed``,
        ``high_dist``, ``low_dist``, ``high_dist_bed``, ``low_dist_bed``,
        ``high_bed``, ``low_bed``, ``cull_distance``, ``use_speed``,
        ``use_dist_to_edge``, ``use_dist_to_grounding_line``, ``use_bed``,
        ``mesh_generator`` (default: 'jigsaw', can be 'simmetrix'),
        and Simmetrix-specific options: ``simmetrix_binary``,
        ``simmetrix_coincident_tolerance``, ``simmetrix_angle_tolerance``,
        ``simmetrix_oncurve_angle_tolerance``, ``simmetrix_units``.
        See the Land-Ice Framework section of the Users or Developers guide
        for more information about these options and their uses.

    gridded_dataset : str
        Name of gridded dataset file to be used for interpolation to MALI mesh

    projection : str
        Projection to be used for setting lat-long fields.
        Likely ``'gis-gimp'`` or ``'ais-bedmap2'``

    geojson_file : str, optional
        Name of geojson file that defines regional domain extent.

    cores : int, optional
        The number of cores to use for mask creation

    bounding_box : array_like of float, shape (4,), optional
        Bounding box [x_min, x_max, y_min, y_max] to cull mesh outside of

    dist_to_edge : numpy.ndarray, optional
        Distance from each gridded dataset point to the ice margin (m),
        on the x1/y1 grid. When provided, cells are culled based on
        whether their nearest gridded ``dist_to_edge`` value exceeds
        ``cull_distance``, rather than using the mesh-based
        ``define_landice_cull_mask`` approach.
    """

    if bounding_box is not None and geojson_file is None:
        msg = (
            "Bounding box clipping can only be applied to an existing cull"
            "mask. You must provide a geojson file for this to work."
        )
        raise ValueError(msg)

    logger = self.logger
    section = self.config[section_name]

    # Check which mesh generator to use
    mesh_generator = section.get('mesh_generator', 'jigsaw').lower()

    if mesh_generator == 'simmetrix':
        logger.info('Using Simmetrix generate2dModel for mesh generation')

        # Get Simmetrix parameters from config (with defaults for GIS)
        coincident_tol = section.getfloat('simmetrix_coincident_tolerance',
                                          1.0)
        angle_tol = section.getfloat('simmetrix_angle_tolerance', 100.0)
        oncurve_angle_tol = section.getfloat(
            'simmetrix_oncurve_angle_tolerance', 100.0)
        units = section.get('simmetrix_units', 'm')

        # Get path to generate2dModel binary
        # Default to 'generate2dModel' (assumes it's in PATH)
        # Can be overridden with full path in config
        simmetrix_binary = section.get('simmetrix_binary',
                                        'generate2dModel')

        # Call generate2dModel with the ice-margin contour (inner) and the
        # domain bounding box (outer) as two separate nested contours.
        edge_vtk = 'edge.vtk'
        bbox_vtk = 'bbox.vtk'
        output_prefix = 'edge_wBbox'

        for input_vtk in (edge_vtk, bbox_vtk):
            if not os.path.exists(input_vtk):
                raise FileNotFoundError(
                    f'Input file {input_vtk} not found. Ensure '
                    'build_cell_width() was called first to create this '
                    'file.')

        args = [simmetrix_binary,
                '--contour', f'file={edge_vtk},order=0,units={units}',
                '--contour', f'file={bbox_vtk},order=1,units={units}',
                output_prefix,
                str(coincident_tol), str(angle_tol), str(oncurve_angle_tol),
                '1']  # createMesh = 1 (generate mesh)

        check_call(args, logger=logger)

        # generate2dModel now directly outputs edge_wBbox.nc in MPAS format
        # Convert to full MPAS mesh using MpasMeshConverter.x
        logger.info('Converting triangular mesh to MPAS mesh')
        args = ['MpasMeshConverter.x', output_prefix + '.nc', 'base_mesh.nc']
        check_call(args, logger=logger)

    else:  # Default to Jigsaw
        logger.info('calling build_planar_mesh')
        build_planar_mesh(cell_width, x1, y1, geom_points,
                          geom_edges, logger=logger)

    dsMesh = xarray.open_dataset('base_mesh.nc')
    logger.info('culling mesh')
    dsMesh = cull(dsMesh, logger=logger)
    logger.info('converting to MPAS mesh')
    dsMesh = convert(dsMesh, logger=logger)
    logger.info('writing grid_converted.nc')
    write_netcdf(dsMesh, 'grid_converted.nc')
    levels = section.get('levels')
    args = ['create_landice_grid_from_generic_mpas_grid',
            '-i', 'grid_converted.nc',
            '-o', 'grid_preCull.nc',
            '-l', levels, '-v', 'glimmer']

    check_call(args, logger=logger)

    args = ['interpolate_to_mpasli_grid', '-s',
            gridded_dataset, '-d',
            'grid_preCull.nc', '-m', 'b', '-t']

    check_call(args, logger=logger)

    # Set when the geojson region mask (and bounding box) have already been
    # folded into the gridded-distance ``cullCell`` mask below, so the
    # separate mesh-space geojson cull further down can be skipped for
    # Simmetrix.
    geojson_folded_into_cullCell = False

    cullDistance = section.get('cull_distance')
    if float(cullDistance) > 0.:
        if mesh_generator == 'simmetrix' and dist_to_edge is not None:
            logger.info('Defining cull mask from gridded dist_to_edge field')
            dsMeshPreCull = xarray.open_dataset('grid_preCull.nc')
            xCell = dsMeshPreCull['xCell'].values
            yCell = dsMeshPreCull['yCell'].values
            # Interpolate dist_to_edge onto mesh cell centers
            dist_interp = interpn(
                (y1, x1), dist_to_edge, (yCell, xCell),
                method='linear', bounds_error=False,
                fill_value=np.max(dist_to_edge))
            cull_dist_m = float(cullDistance) * 1.0e3
            # Interior cells have negative dist_interp, so the threshold
            # comparison alone excludes them from culling.
            cullCell = (dist_interp > cull_dist_m).astype(np.int32)

            if geojson_file is not None:
                logger.info(
                    'Combining gridded dist_to_edge cull mask with '
                    'rasterized geojson region mask')
                geojson_mask = rasterize_geojson_mask(
                    geojson_file, x1, y1, projection)
                geojson_mask_interp = interpn(
                    (y1, x1), geojson_mask.astype(np.float64),
                    (yCell, xCell), method='nearest',
                    bounds_error=False, fill_value=0.0).astype(bool)
                cullCell = (cullCell.astype(bool) |
                            ~geojson_mask_interp).astype(np.int32)

            if bounding_box is not None:
                outside_bbox = (
                    (xCell < bounding_box[0].item()) |
                    (xCell > bounding_box[1].item()) |
                    (yCell < bounding_box[2].item()) |
                    (yCell > bounding_box[3].item())
                )
                cullCell = (cullCell.astype(bool) |
                            outside_bbox).astype(np.int32)

            # Ensure culled region is topologically connected to the
            # domain boundary. Isolated culled patches within the buffer
            # region are un-culled so there are no holes in the retained
            # mesh.
            cellsOnCell = dsMeshPreCull['cellsOnCell'].values
            nEdgesOnCell = dsMeshPreCull['nEdgesOnCell'].values
            maxEdges = cellsOnCell.shape[1]
            col_idx = np.arange(maxEdges)
            valid_edge = (col_idx[np.newaxis, :] <
                          nEdgesOnCell[:, np.newaxis])
            boundary_mask = np.any(
                (cellsOnCell == 0) & valid_edge,
                axis=1).astype(np.int32)
            seed_mask = (boundary_mask & cullCell).astype(np.int32)
            cullCell = mpas_flood_fill(seed_mask, cullCell,
                                       cellsOnCell, nEdgesOnCell)

            dsMeshPreCull['cullCell'] = xarray.DataArray(
                cullCell, dims=['nCells'])
            write_netcdf(dsMeshPreCull, 'grid_preCull.nc')
            dsMeshPreCull.close()
            geojson_folded_into_cullCell = True
        else:
            args = ['define_landice_cull_mask', '-f',
                    'grid_preCull.nc', '-m',
                    'distance', '-d', cullDistance]

            check_call(args, logger=logger)
    else:
        logger.info('cullDistance <= 0 in config file. '
                    'Will not cull by distance to margin. \n')

    if geojson_file is not None and not geojson_folded_into_cullCell:
        # This step is only necessary because the GeoJSON region
        # is defined by lat-lon. For Simmetrix, this is skipped when the
        # geojson region has already been rasterized onto the structured
        # grid and folded into the gridded-distance ``cullCell`` mask
        # above.
        args = ['set_lat_lon_fields_in_planar_grid', '-f',
                'grid_preCull.nc', '-p', projection]

        check_call(args, logger=logger)

        args = ['compute_mpas_region_masks',
                '-m', 'grid_preCull.nc',
                '-o', 'mask.nc',
                '-g', geojson_file,
                '--process_count', f'{cores}',
                '--format', mpas_tools.io.default_format,
                '--engine', mpas_tools.io.default_engine]

        check_call(args, logger=logger)

        logger.info('culling to geojson file')

    dsMesh = xarray.open_dataset('grid_preCull.nc')
    if geojson_file is not None and not geojson_folded_into_cullCell:
        mask = xarray.open_dataset('mask.nc')

        if bounding_box is not None:
            mask = clip_mesh_to_bounding_box(mask, dsMesh, bounding_box)

    else:
        mask = None

    dsMesh = cull(dsMesh, dsInverse=mask, logger=logger)
    write_netcdf(dsMesh, 'culled.nc')

    # Removing horn cells (cells with two or fewer neighbors) can demote a
    # neighboring cell to two or fewer neighbors, creating a new horn, so
    # mark-and-cull is repeated until no horns remain.
    dsMesh = xarray.open_dataset('culled.nc')
    max_horn_passes = 10
    horn_pass = 0
    while True:
        cellsOnCell = dsMesh['cellsOnCell'].values
        nEdgesOnCell = dsMesh['nEdgesOnCell'].values
        maxEdges = cellsOnCell.shape[1]
        col_idx = np.arange(maxEdges)
        valid_edge = col_idx[np.newaxis, :] < nEdgesOnCell[:, np.newaxis]
        nNeighbors = np.sum((cellsOnCell > 0) & valid_edge, axis=1)
        nHorns = int(np.sum(nNeighbors <= 2))
        if nHorns == 0:
            break
        if horn_pass >= max_horn_passes:
            raise RuntimeError(
                f'Horn removal did not converge after '
                f'{max_horn_passes} passes; {nHorns} horn cells remain.')
        horn_pass += 1
        logger.info(f'Marking and culling horns, pass {horn_pass}: '
                    f'{nHorns} horn cells found')
        write_netcdf(dsMesh, 'culled.nc')
        args = ['mark_horns_for_culling', '-f', 'culled.nc']
        check_call(args, logger=logger)
        dsMesh = xarray.open_dataset('culled.nc')
        dsMesh = cull(dsMesh, logger=logger)
        dsMesh = convert(dsMesh, logger=logger)

    logger.info('sorting mesh')
    dsMesh = sort_mesh(dsMesh)
    write_netcdf(dsMesh, 'dehorned_sorted.nc')

    if mesh_generator == 'simmetrix':
        # Fit splines to the dehorned mesh's boundary and generate the
        # final mesh from the resulting geometric model; this mesh already
        # has the correct (dehorned) domain shape, so it replaces dsMesh
        # and needs no further culling.
        dsMesh = fit_boundary_splines(self, dsMesh, section)
        logger.info('sorting boundary-fitted mesh')
        dsMesh = sort_mesh(dsMesh)

    write_netcdf(dsMesh, 'dehorned.nc')

    args = ['create_landice_grid_from_generic_mpas_grid', '-i',
            'dehorned.nc', '-o',
            mesh_name, '-l', levels, '-v', 'glimmer',
            '--beta', '--thermal', '--obs', '--diri']

    check_call(args, logger=logger)

    args = ['interpolate_to_mpasli_grid', '-s',
            gridded_dataset, '-d', mesh_name, '-m', 'b']

    check_call(args, logger=logger)

    if mesh_generator == 'simmetrix':
        # create_landice_grid_from_generic_mpas_grid only copies a fixed
        # allowlist of variables, so re-attach the geometric model
        # classification from dehorned.nc onto the final mesh.
        dsDehorned = xarray.open_dataset('dehorned.nc')
        dsMeshFinal = xarray.open_dataset(mesh_name)
        for var in ('geomModelIdCell', 'geomModelDimCell',
                    'geomModelIdVertex', 'geomModelDimVertex'):
            dsMeshFinal[var] = dsDehorned[var]
        write_netcdf(dsMeshFinal, mesh_name)
        dsDehorned.close()
        dsMeshFinal.close()

    logger.info('Marking domain boundaries dirichlet')
    args = ['mark_domain_boundaries_dirichlet',
            '-f', mesh_name]
    check_call(args, logger=logger)

    args = ['set_lat_lon_fields_in_planar_grid', '-f',
            mesh_name, '-p', projection]
    check_call(args, logger=logger)


def make_region_masks(self, mesh_filename, mask_filename,
                      cores, tags, component='landice', all_tags=True):
    """
    Create masks for ice-sheet subregions based on data
    in ``MPAS-Dev/geometric_fatures``.

    Parameters
    ----------
    mesh_filename : str
        name of NetCDF mesh file for which to create region masks

    mask_filename : str
        name of NetCDF file to contain region masks

    cores : int
        number of processors used to create region masks

    tags : list of str
        Groups of regions for which masks are to be defined
    """

    logger = self.logger
    logger.info('creating region masks')
    gf = GeometricFeatures()
    fcMask = FeatureCollection()

    fc = gf.read(componentName=component, objectType='region',
                 tags=tags, allTags=all_tags)
    fcMask.merge(fc)

    geojson_filename = 'regionMask.geojson'
    fcMask.to_geojson(geojson_filename)

    args = ['compute_mpas_region_masks',
            '-m', mesh_filename,
            '-g', geojson_filename,
            '-o', mask_filename,
            '-t', 'cell', 'edge',
            '--process_count', f'{cores}',
            '--format', mpas_tools.io.default_format,
            '--engine', mpas_tools.io.default_engine]
    check_call(args, logger=logger)


def add_bedmachine_thk_to_ais_gridded_data(self, source_gridded_dataset,
                                           bedmachine_path):
    """
    Copy BedMachine thickness to AIS reference gridded dataset.
    Replace thickness field in the compilation dataset with the one we
    will be using from BedMachine for actual thickness interpolation.
    There are significant inconsistencies between the masking of the two,
    particularly along the Antarctic Peninsula, that lead to funky
    mesh extent and culling if we use the thickness from 8km composite
    dataset to define the cullMask but then actually interpolate thickness
    from BedMachine.
    This function uses bilinear interpolation to interpolate from the 500 m
    resolution of BedMachine to the 8 km resolution of the reference dataset.
    It is not particularly accurate, but is fast and adequate for generating
    the flood filled mask for culling the mesh.  Highly accurate conservative
    remapping is performed later for actually interpolating BedMachine
    thickness to the final MALI mesh.

    Parameters
    ----------
    source_gridded_dataset : str
        name of NetCDF file containing original AIS gridded datasets

    bedmachine_path : str
        path to BedMachine dataset

    Returns
    -------
    gridded_dataset_with_bm_thk : str
        name of NetCDF file with gridded dataset with BedMachine thk added
    """

    logger = self.logger

    tic = time.perf_counter()
    bm_data = Dataset(bedmachine_path, 'r')
    bm_x = bm_data.variables['x'][:]
    bm_y = bm_data.variables['y'][:]
    bm_mask = bm_data.variables['iceMask'][:]
    bm_thk = bm_data.variables['thk'][:]
    # BedMachine v2 includes a mask with: 0=ocean, 1=land, 2=grd ice
    #                                  3=flt ice, 4=vostok
    # NOTE: Later versions of BedMachine may not have the same mask values!
    # We only want to keep thickness where the mask has ice;
    # this is necessary because thickness has been extrapolated.
    bm_thk *= (bm_mask > 1.5)
    # The two datasets are oriented differently, so align them.
    bm_thk = np.flipud(np.rot90(bm_thk))
    gridded_dataset_with_bm_thk = \
        f"{source_gridded_dataset.split('.')[:-1][0]}_BedMachineThk.nc"
    copyfile(source_gridded_dataset, gridded_dataset_with_bm_thk)
    gg = Dataset(gridded_dataset_with_bm_thk, 'r+')
    gg_x = gg.variables['x1'][:]
    gg_y = gg.variables['y1'][:]
    gg_xx, gg_yy = np.meshgrid(gg_x, gg_y)
    gg_thk = interpn((bm_x, bm_y), bm_thk, (gg_xx, gg_yy),
                     bounds_error=False, fill_value=0.0)
    gg.variables['thk'][0, :, :] = gg_thk
    gg.close()
    bm_data.close()
    toc = time.perf_counter()
    logger.info('Finished interpolating BedMachine thickness to reference '
                f'grid in {toc - tic} seconds')
    return gridded_dataset_with_bm_thk


def preprocess_ais_data(self, source_gridded_dataset,
                        floodFillMask):
    """
    Perform adjustments to gridded AIS datasets needed
    for rest of compass workflow to utilize them

    Parameters
    ----------
    source_gridded_dataset : str
        name of NetCDF file containing original AIS gridded datasets

    floodFillMask : numpy.ndarray
        0/1 mask of flood filled ice region

    Returns
    -------
    preprocessed_gridded_dataset : str
        name of NetCDF file with preprocessed version of gridded dataset
    """

    logger = self.logger

    # Apply floodFillMask to thickness field to help with culling
    file_with_flood_fill = \
        f"{source_gridded_dataset.split('.')[:-1][0]}_floodFillMask.nc"
    copyfile(source_gridded_dataset, file_with_flood_fill)
    gg = Dataset(file_with_flood_fill, 'r+')
    gg.variables['thk'][0, :, :] *= floodFillMask
    gg.variables['vx'][0, :, :] *= floodFillMask
    gg.variables['vy'][0, :, :] *= floodFillMask
    gg.close()

    # Now deal with the peculiarities of the AIS dataset.
    preprocessed_gridded_dataset = \
        f"{file_with_flood_fill.split('.')[:-1][0]}_filledFields.nc"
    copyfile(file_with_flood_fill,
             preprocessed_gridded_dataset)
    data = Dataset(preprocessed_gridded_dataset, 'r+')
    data.set_auto_mask(False)
    x1 = data.variables["x1"][:]
    y1 = data.variables["y1"][:]
    cellsWithIce = data.variables["thk"][:].ravel() > 0.
    data.createVariable('iceMask', 'f', ('time', 'y1', 'x1'))
    data.variables['iceMask'][:] = data.variables["thk"][:] > 0.

    # Note: dhdt is only reported over grounded ice, so we will have to
    # either update the dataset to include ice shelves or give them values of
    # 0 with reasonably large uncertainties.
    dHdt = data.variables["dhdt"][:]
    dHdtErr = 0.05 * dHdt  # assign arbitrary uncertainty of 5%
    # Where dHdt data are missing, set large uncertainty
    dHdtErr[dHdt > 1.e30] = 1.

    # Extrapolate fields beyond region with ice to avoid interpolation
    # artifacts of undefined values outside the ice domain
    # Do this by creating a nearest neighbor interpolator of the valid data
    # to recover the actual data within the ice domain and assign nearest
    # neighbor values outside the ice domain
    xGrid, yGrid = np.meshgrid(x1, y1)
    xx = xGrid.ravel()
    yy = yGrid.ravel()
    bigTic = time.perf_counter()
    for field in ['thk', 'bheatflx', 'vx', 'vy',
                  'ex', 'ey', 'thkerr', 'dhdt']:
        tic = time.perf_counter()
        logger.info(f"Beginning building interpolator for {field}")
        if field in ['thk', 'thkerr']:
            mask = cellsWithIce.ravel()
        elif field == 'bheatflx':
            mask = np.logical_and(
                data.variables[field][:].ravel() < 1.0e9,
                data.variables[field][:].ravel() != 0.0)
        elif field in ['vx', 'vy', 'ex', 'ey', 'dhdt']:
            mask = np.logical_and(
                data.variables[field][:].ravel() < 1.0e9,
                cellsWithIce.ravel() > 0)
        else:
            mask = cellsWithIce
        interp = NearestNDInterpolator(
            list(zip(xx[mask], yy[mask])),
            data.variables[field][:].ravel()[mask])
        toc = time.perf_counter()
        logger.info(f"Finished building interpolator in {toc - tic} seconds")

        tic = time.perf_counter()
        logger.info(f"Beginning interpolation for {field}")
        # NOTE: Do not need to evaluate the extrapolator at all grid cells.
        #       Only needed for ice-free grid cells, since is NN extrapolation
        data.variables[field][0, :] = interp(xGrid, yGrid)
        toc = time.perf_counter()
        logger.info(f"Interpolation completed in {toc - tic} seconds")

    bigToc = time.perf_counter()
    logger.info(f"All interpolations completed in {bigToc - bigTic} seconds.")

    # Now perform some additional clean up adjustments to the dataset
    data.createVariable('dHdtErr', 'f', ('time', 'y1', 'x1'))
    data.variables['dHdtErr'][:] = dHdtErr

    data.createVariable('vErr', 'f', ('time', 'y1', 'x1'))
    data.variables['vErr'][:] = np.sqrt(data.variables['ex'][:]**2 +
                                        data.variables['ey'][:]**2)

    data.variables['bheatflx'][:] *= 1.e-3  # correct units
    data.variables['bheatflx'].units = 'W m-2'

    data.variables['subm'][:] *= -1.0  # correct basal melting sign
    data.variables['subm_ss'][:] *= -1.0

    data.renameVariable('dhdt', 'dHdt')
    data.renameVariable('thkerr', 'topgerr')

    data.createVariable('x', 'f', ('x1'))
    data.createVariable('y', 'f', ('y1'))
    data.variables['x'][:] = x1
    data.variables['y'][:] = y1

    data.close()

    return preprocessed_gridded_dataset


def interp_gridded2mali(self, source_file, mali_scrip, parallel_executable,
                        nProcs, dest_file, proj, variables="all"):
    """
    Interpolate gridded dataset (e.g. MEASURES, BedMachine) onto a MALI mesh

    Parameters
    ----------
    source_file : str
        filepath to the source gridded datatset to be interpolated

    mali_scrip : str
        name of scrip file corresponding to destination MALI mesh

    parallel_executable : str
        executable needed to launch a parallel job

    nProcs : int
        number of processors to use for generating remapping weights

    dest_file: str
        MALI input file to which data should be remapped

    proj: str
        projection of the source dataset

    variables: "all" or list of strings
        either the string "all" or a list of strings
    """

    def __guess_scrip_name(filename):

        # try searching for string followed by a version number
        match = re.search(r'(^.*[_-]v\d*[_-])+', filename)

        if match:
            # slice string to end of match minus one to leave of final _ or -
            base_fn = filename[:match.end() - 1]
        else:
            # no matches were found, just use the filename (minus extension)
            base_fn = os.path.splitext(filename)[0]

        return f"{base_fn}.scrip.nc"

    logger = self.logger

    source_scrip = __guess_scrip_name(os.path.basename(source_file))
    weights_filename = "gridded_to_MPAS_weights.nc"

    # make sure variables is a list, encompasses the variables="all" case
    if isinstance(variables, str):
        variables = [variables]
    if not isinstance(variables, list):
        raise TypeError("Arugment 'variables' is of incorrect type, must"
                        " either the string 'all' or a list of strings")

    logger.info('creating scrip file for source dataset')
    # Note: writing scrip file to workdir
    args = ['create_scrip_file_from_planar_rectangular_grid',
            '-i', source_file,
            '-s', source_scrip,
            '-p', proj,
            '-r', '2']
    check_call(args, logger=logger)

    # Generate remapping weights
    logger.info('generating gridded dataset -> MPAS weights')
    args = [parallel_executable, '-n', nProcs, 'ESMF_RegridWeightGen',
            '--source', source_scrip,
            '--destination', mali_scrip,
            '--weight', weights_filename,
            '--method', 'conserve',
            "--netcdf4",
            "--dst_regional", "--src_regional", '--ignore_unmapped']
    check_call(args, logger=logger)

    # Perform actual interpolation using the weights
    logger.info('calling interpolate_to_mpasli_grid')
    args = ['interpolate_to_mpasli_grid',
            '-s', source_file,
            '-d', dest_file,
            '-m', 'e',
            '-w', weights_filename,
            '-v'] + variables

    check_call(args, logger=logger)


def clean_up_after_interp(fname):
    """
    Perform some final clean up steps after interpolation

    Parameters
    ----------
    fname : str
        name of file on which to perform clean up
    """

    # Create a backup in case clean-up goes awry
    backup_name = f"{fname.split('.')[:-1][0]}_backup.nc"
    copyfile(fname, backup_name)

    # Clean up: trim to iceMask and set large velocity
    # uncertainties where appropriate.
    data = Dataset(fname, 'r+')
    data.set_auto_mask(False)
    data.variables['thickness'][:] *= (data.variables['iceMask'][:] > 1.5)

    mask = np.logical_or(
        np.isnan(data.variables['observedSurfaceVelocityUncertainty'][:]),
        data.variables['thickness'][:] < 1.0)
    mask = np.logical_or(
        mask,
        data.variables['observedSurfaceVelocityUncertainty'][:] == 0.0)
    data.variables['observedSurfaceVelocityUncertainty'][0, mask[0, :]] = 1.0
    data.close()
