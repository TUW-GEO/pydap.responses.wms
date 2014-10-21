from __future__ import division

from StringIO import StringIO
import re
import operator
import bisect

# from paste.request import construct_url, parse_dict_querystring
# from paste.httpexceptions import HTTPBadRequest
# from paste.util.converters import asbool
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.colorbar import ColorbarBase
from matplotlib.colors import Normalize
from matplotlib import rcParams
rcParams['xtick.labelsize'] = 'small'
rcParams['ytick.labelsize'] = 'small'
import iso8601
import coards
import json

try:
    from PIL import Image
except:
    PIL = None

from pydap.model import *
# from pydap.responses.lib import BaseResponse
# from pydap.util.template import GenshiRenderer, StringLoader, TemplateNotFound
# from pydap.util.safeeval import expr_eval
# from pydap.lib import walk, encode_atom
from jinja2 import Environment, PackageLoader, ChoiceLoader
from webob import Response
from webob.dec import wsgify
from webob.exc import HTTPSeeOther
from urllib import unquote

import pandas as pd


def to_dygraph_format(self):
    labels = ['date']
    labels.extend(self.columns.values.tolist())
    data_values = np.hsplit(self.values, self.columns.values.size)
    data_index = self.index.values.astype('M8[s]').tolist()
    data_index = [x.strftime("%Y/%m/%d %H:%M:%S") for x in data_index]
    data_index = np.reshape(data_index, (len(data_index), 1))
    data_values.insert(0, data_index)
    data_values = np.column_stack(data_values)
    return labels, data_values.tolist()

pd.DataFrame.to_dygraph_format = to_dygraph_format

from pydap.responses.lib import BaseResponse
from pydap.lib import __version__
from pydap.responses.wms.load_cmap import load as get_cmap


WMS_ARGUMENTS = ['request', 'bbox', 'cmap', 'layers',
                 'width', 'height', 'transparent', 'time']


class WMSResponse(BaseResponse):

    __description__ = "Web Map Service image"
    __version__ = __version__

    __template__ = ""
#
#     renderer = GenshiRenderer(
# options={}, loader=StringLoader({'capabilities.xml': DEFAULT_TEMPLATE}))

    def __init__(self, dataset):
        BaseResponse.__init__(self, dataset)
        self.headers.append(('Content-description', 'dods_wms'))

        # our default environment
        self.loaders = [
            PackageLoader("pydap.responses.wms", "templates"),
        ]

    @wsgify
    def __call__(self, req):
        query = req.GET
        try:
            dap_query = ['%s=%s' % (k, query[k]) for k in query
                         if k.lower() not in WMS_ARGUMENTS]
            dap_query = [pair.rstrip('=') for pair in dap_query]
            dap_query.sort()  # sort for uniqueness
            dap_query = '&'.join(dap_query)
            location = req.path_url + "?" + dap_query
            self.cache = req.environ['beaker.cache'].get_cache(
                'pydap.responses.wms+' + location)
        except KeyError:
            self.cache = None

        # check if the server has specified a render environment; if it has,
        # make a copy and add our loaders to it
        if "pydap.jinja2.environment" in req.environ:
            env = req.environ["pydap.jinja2.environment"].overlay()
            env.loader = ChoiceLoader([
                loader for loader in [env.loader] + self.loaders if loader])
        else:
            env = Environment(loader = ChoiceLoader(self.loaders))

        env.filters["unquote"] = unquote
        self.__template__ = env.get_template("wms.xml")

        # Handle GetMap and GetCapabilities requests
        type_ = query.get('REQUEST', 'GetMap')
        content = ""
        if type_ == 'GetCapabilities':
            content = self._get_capabilities(req)[0]
            self.headers.append(('Content-type', 'text/xml'))
            self.headers.append(('Access-Control-Allow-Origin', '*'))
        elif type_ == 'GetMap':
            content = self._get_map(req)[0]
            self.headers.append(('Content-type', 'image/png'))
        elif type_ == 'GetColorbar':
            content = self._get_colorbar(req)[0]
            self.headers.append(('Content-type', 'image/png'))
        elif type_ == 'GetTimeseries':
            content = self._get_timeseries(req)
            self.headers.append(('Content-type', 'application/json'))
        else:
            content = "Invalid REQUEST: " + str(type_)
            self.headers.append(('Content-type', 'text/html'))
            # @TODO Implement Exception
            # raise HTTPBadRequest('Invalid REQUEST "%s"' % type_)

        return Response(body = content, headers = self.headers)

    def _get_timeseries(self, req):
        query = req.GET

        def serialize(dataset):
            fix_map_attributes(dataset)

            bbox = [float(v) for v in query.get('BBOX', '0,0,0,0').split(',')]
            layers = [layer for layer in query.get('LAYERS', '').split(',')
                      if layer] or [var.id for var in walk(dataset, GridType)]
            layer = layers[0]
            names = [dataset] + layer.split('.')
            grid = reduce(operator.getitem, names)

            lon = np.asarray(get_lon(grid, dataset)[:])
            lat = np.asarray(get_lat(grid, dataset)[:])
            i0, i1 = find_containing_bounds(lon, bbox[0], bbox[2])
            j0, j1 = find_containing_bounds(lat, bbox[1], bbox[3])
            lons = lon[i0:i1:1]
            lats = lat[j0:j1:1]

            # get order of dimensions
            time_i = slice(None)
            lat_i = slice(j0, j1, 1)
            lon_i = slice(i0, i1, 1)
            gridarr_dim = []
            flip = False
            hastime = False
            for dim in grid.dimensions:
                if dim.lower() == 'time':
                    hastime = True
                    gridarr_dim.append(time_i)
                elif (dim.lower() == 'lon' or dim.lower() == 'longitude' or dim.lower() == 'coadsx'):
                    gridarr_dim.append(lon_i)
                    flip = False
                elif (dim.lower() == 'lat' or dim.lower() == 'latitude' or dim.lower() == 'coadsy'):
                    gridarr_dim.append(lat_i)
                else:
                    gridarr_dim.append(Ellipsis)

            if len(gridarr_dim) < 3:
                gridarr_dim.append(Ellipsis)

            data = np.asarray(
                grid.array[gridarr_dim[0], gridarr_dim[1], gridarr_dim[2]])
            if hastime:
                timeelements = data.shape[0]
            else:
                timeelements = 0
            if 'missing_value' in grid.attributes:
                data = np.ma.masked_equal(
                    data, grid.attributes['missing_value'])
            elif '_FillValue' in grid.attributes:
                data = np.ma.masked_equal(data, grid.attributes['_FillValue'])

            if grid.attributes.get('scale_factor'):
                data = data * float(grid.attributes['scale_factor'])
            if grid.attributes.get('add_offset'):
                data += grid.attributes['add_offset']

            # Only allows time in 1st dimension
            ts = data
            while len(ts.shape) > 1:
                ts = np.nanmean(ts, axis = 1)
            d = {layer: ts}
            index = get_time(grid, dataset)
            df = pd.DataFrame(d, index = index)
            if hasattr(dataset, 'close'):
                dataset.close()
            labels, values = df.to_dygraph_format()
            return json.dumps({'labels': [l.encode('utf-8') for l in labels], 'data': values})
        return serialize(self.dataset)

    def _get_colorbar(self, req):
        w, h = 300, 70
        query = req.GET

        dpi = float(req.environ.get('pydap.responses.wms.dpi', 80))
        figsize = w / dpi, h / dpi
        cmap = query.get(
            'cmap', req.environ.get('pydap.responses.wms.cmap', 'jet'))

        def serialize(dataset):
            fix_map_attributes(dataset)
            fig = Figure(figsize = figsize, dpi = dpi)
            fig.figurePatch.set_alpha(0.0)
            ax = fig.add_axes([0.05, 0.50, 0.90, 0.45])
            ax.axesPatch.set_alpha(0.5)

            # Plot requested grids.
            layers = [layer for layer in query.get('LAYERS', '').split(',')
                      if layer] or [var.id for var in walk(dataset, GridType)]
            layer = layers[0]
            names = [dataset] + layer.split('.')
            grid = reduce(operator.getitem, names)

            actual_range, range_guest = self._get_actual_range(grid)
            fontsize = 14
            if actual_range[1] > 999:
                fontsize = 12
            norm = Normalize(vmin = actual_range[0], vmax = actual_range[1])
            cb = ColorbarBase(ax, cmap = get_cmap(cmap), norm = norm,
                              orientation = 'horizontal')
            cb.set_label(self._get_units(grid))
            for tick in cb.ax.get_yticklabels():
                tick.set_fontsize(fontsize)
                tick.set_color('white')
                # tick.set_fontweight('bold')

            # Save to buffer.
            canvas = FigureCanvas(fig)
            output = StringIO()
            canvas.print_png(output)
            if hasattr(dataset, 'close'):
                dataset.close()
            return [output.getvalue()]
        return serialize(self.dataset)

    def _get_units(self, grid):
        try:
            units = self.cache.get_value(str((grid.id, 'units')))
        except (KeyError, AttributeError):
            try:
                units = grid.attributes['units']
            except KeyError:
                try:
                    units = grid.attributes['unit']
                except KeyError:
                    units = ""
            if self.cache:
                self.cache.set_value(str((grid.id, 'units')), units)
        return units

    def _get_actual_range(self, grid):
        try:
            actual_range = self.cache.get_value(str((grid.id, 'actual_range')))
        except (KeyError, AttributeError):
            try:
                actual_range = grid.attributes['actual_range']
            except KeyError:
                try:
                    actual_range = grid.attributes['valid_range']
                except KeyError:
                    return [0, 1], True  # return here ... don't cache
                    # data = fix_data(np.asarray(grid.array[:]), grid.attributes)
                    # actual_range = np.nanmin(data), np.nanmax(data)
            # Try to parse float
            try:
                actual_range = [float(actual_range[0]), float(actual_range[1])]
            except (ValueError):
                return [0, 1], True  # return here ... don't cache

            # Cache actual range if beaker is active
            if self.cache:
                self.cache.set_value(
                    str((grid.id, 'actual_range')), actual_range)

        return actual_range, False

    def _get_valid_range(self, grid):
        try:
            valid_range = self.cache.get_value(str((grid.id, 'valid_range')))
        except (KeyError, AttributeError):
            try:
                valid_range = grid.attributes['valid_range']
            except KeyError:
                data = fix_data(np.asarray(grid.array[:]), grid.attributes)
                valid_range = np.nanmin(data), np.nanmax(data)
            if self.cache:
                self.cache.set_value(
                    str((grid.id, 'valid_range')), valid_range)
        return valid_range

    def _get_map(self, req):
        # Calculate appropriate figure size.
        query = req.GET

        dpi = float(req.environ.get('pydap.responses.wms.dpi', 80))
        w = float(query.get('WIDTH', 256))
        h = float(query.get('HEIGHT', 256))
        time = query.get('TIME')
        figsize = w / dpi, h / dpi
        bbox = [float(v)
                for v in query.get('BBOX', '-180,-90,180,90').split(',')]
        cmap = query.get(
            'cmap', req.environ.get('pydap.responses.wms.cmap', 'jet'))

        def serialize(dataset):
            fix_map_attributes(dataset)
            fig = Figure(figsize = figsize, dpi = dpi)
            ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])

            # Set transparent background; found through
            # http://sparkplot.org/browser/sparkplot.py.
            if asbool(query.get('TRANSPARENT', 'true')):
                fig.figurePatch.set_alpha(0.0)
                ax.axesPatch.set_alpha(0.0)

            # Plot requested grids (or all if none requested).
            layers = [layer for layer in query.get('LAYERS', '').split(',')
                      if layer] or [var.id for var in walk(dataset, GridType)]
            for layer in layers:
                names = [dataset] + layer.split('.')
                grid = reduce(operator.getitem, names)
                if is_valid(grid, dataset):
                    self._plot_grid(
                        dataset, grid, time, bbox, (w, h), ax, cmap)

            # Save to buffer.
            ax.axis([bbox[0], bbox[2], bbox[1], bbox[3]])
            ax.axis('off')
            canvas = FigureCanvas(fig)
            output = StringIO()
            # Optionally convert to paletted png
            paletted = asbool(
                req.environ.get('pydap.responses.wms.paletted', 'false'))
            if paletted:
                # Read image
                buf, size = canvas.print_to_buffer()
                im = Image.frombuffer('RGBA', size, buf, 'raw', 'RGBA', 0, 1)
                # Find number of colors
                colors = im.getcolors(256)
                # Only convert if the number of colors is less than 256
                if colors is not None:
                    ncolors = len(colors)
                    # Get alpha band
                    alpha = im.split()[-1]
                    # Convert to paletted image
                    im = im.convert("RGB")
                    im = im.convert(
                        "P", palette = Image.ADAPTIVE, colors = ncolors)
                    # Set all pixel values below ncolors to 1 and the rest to 0
                    mask = Image.eval(alpha, lambda a: 255 if a <= 128 else 0)
                    # Paste the color of index ncolors and use alpha as a mask
                    im.paste(ncolors, mask)
                    # Truncate palette to actual size to save space
                    im.palette.palette = im.palette.palette[:3 * (ncolors + 1)]
                    im.save(
                        output, 'png', optimize = False, transparency = ncolors)
                else:
                    canvas.print_png(output)
            else:
                canvas.print_png(output)
            if hasattr(dataset, 'close'):
                dataset.close()
            return [output.getvalue()]
        return serialize(self.dataset)

    def _plot_grid(self, dataset, grid, time, bbox, size, ax, cmap = 'jet'):
        # Get actual data range for levels.
        actual_range, range_guest = self._get_actual_range(grid)
        V = np.linspace(actual_range[0], actual_range[1], 10)
        extent = (0, 0, 0, 0)
        # Slice according to time request (WMS-T).
        if time is not None:
            values = np.array(get_time(grid, dataset))
            l = np.zeros(len(values), bool)  # get no data by default

            tokens = time.split(',')
            for token in tokens:
                if '/' in token:  # range
                    start, end = token.strip().split('/')
                    start = iso8601.parse_date(start, default_timezone = None)
                    end = iso8601.parse_date(end, default_timezone = None)
                    l[(values >= start) & (values <= end)] = True
                else:
                    instant = iso8601.parse_date(
                        token.strip().rstrip('Z'), default_timezone = None)
                    l[values == instant] = True
        else:
            l = None

        # Plot the data over all the extension of the bbox.
        # First we "rewind" the data window to the begining of the bbox:
        lon = get_lon(grid, dataset)
        cyclic = hasattr(lon, 'modulo')
        lon = np.asarray(lon[:])
        lat = np.asarray(get_lat(grid, dataset)[:])
        while np.min(lon) > bbox[0] and np.min(lon) > 0:
            lon -= 360.0
        # Now we plot the data window until the end of the bbox:
        w, h = size
        while np.min(lon) < bbox[2]:
            # Retrieve only the data for the request bbox, and at the
            # optimal resolution (avoiding oversampling).
            if len(lon.shape) == 1:
                i0, i1 = find_containing_bounds(lon, bbox[0], bbox[2])
                j0, j1 = find_containing_bounds(lat, bbox[1], bbox[3])
                istep_float = float(
                    (len(lon) * (bbox[2] - bbox[0])) / (w * abs(lon[-1] - lon[0])))
                jstep_float = float(
                    (len(lat) * (bbox[3] - bbox[1])) / (h * abs(lat[-1] - lat[0])))
                izoom = float(1) / istep_float
                jzoom = float(1) / jstep_float
                zoom = int(izoom)
                if jzoom > izoom:
                    zoom = int(jzoom)
                istep = int(
                    max(1, np.floor((len(lon) * (bbox[2] - bbox[0])) / (w * abs(lon[-1] - lon[0])))))
                jstep = int(
                    max(1, np.floor((len(lat) * (bbox[3] - bbox[1])) / (h * abs(lat[-1] - lat[0])))))
                lons = lon[i0:i1:istep]
                lats = lat[j0:j1:jstep]

                # get order of dimensions
                try:
                    time_i = np.where(l)[0][0]
                except:
                    time_i = slice(None)
                lat_i = slice(j0, j1, 1)
                lon_i = slice(i0, i1, 1)
                gridarr_dim = []
                datastep = []
                flip = False
                for dim in grid.dimensions:
                    if dim.lower() == 'time':
                        gridarr_dim.append(time_i)
                        datastep.append(1)
                    elif (dim.lower() == 'lon' or dim.lower() == 'longitude' or dim.lower() == 'coadsx'):
                        gridarr_dim.append(lon_i)
                        flip = False
                        datastep.append(istep)
                    elif (dim.lower() == 'lat' or dim.lower() == 'latitude' or dim.lower() == 'coadsy'):
                        gridarr_dim.append(lat_i)
                        datastep.append(jstep)
                        # flip if latitude is second spatial dimension
                        flip = True
                    else:
                        gridarr_dim.append(Ellipsis)

                if len(gridarr_dim) < 3:
                    gridarr_dim.append(Ellipsis)

                data = np.asarray(
                    grid.array[gridarr_dim[0], gridarr_dim[1], gridarr_dim[2]])

                import scipy.ndimage.interpolation as interp
                if zoom > 4:
                    if len(data.shape) == 2:
                        zoom = (zoom, zoom)
                    elif len(data.shape) == 3:
                        zoom = (1, zoom, zoom)

                    data = interp.zoom(data, zoom, order = 0)
                else:
                    zoom = int(1)

                if len(datastep) < 3:
                    data = data[::datastep[0], ::datastep[1]]
                else:
                    data = data[::datastep[0], ::datastep[1], ::datastep[2]]

                # Fix cyclic data.
                if cyclic:
                    lons = np.ma.concatenate((lons, lon[0:1] + 360.0), 0)

                    if l is not None:
                        if flip:
                            gridarr_dim[1] = slice(0, 1, 1)
                        else:
                            gridarr_dim[2] = slice(0, 1, 1)
                        data = np.ma.concatenate((
                            data, interp.zoom(grid.array[gridarr_dim[0], gridarr_dim[1], gridarr_dim[2]], zoom, order = 0)), -1)
                    else:
                        if flip:
                            gridarr_dim[0] = slice(0, 1, 1)
                        else:
                            gridarr_dim[1] = slice(0, 1, 1)
                        data = np.ma.concatenate((
                            data, interp.zoom(grid.array[..., gridarr_dim[0], gridarr_dim[1]], zoom, order = 0)), -1)

                # Flip if dimension order is changed
                if flip:
                    data = np.fliplr(np.rot90(data, 3))
                X, Y = np.meshgrid(lons, lats)
                extent = (
                    X[0, 0], X[0, X.shape[1] - 1], Y[Y.shape[0] - 1, 0], Y[0, 0])
            elif len(lon.shape) == 2:
                i, j = np.arange(lon.shape[1]), np.arange(lon.shape[0])
                I, J = np.meshgrid(i, j)

                xcond = (lon >= bbox[0]) & (lon <= bbox[2])
                ycond = (lat >= bbox[1]) & (lat <= bbox[3])
                if not xcond.any() or not ycond.any():
                    lon += 360.0
                    continue

                i0, i1 = np.min(I[xcond]), np.max(I[xcond])
                j0, j1 = np.min(J[ycond]), np.max(J[ycond])
                istep = max(1, int(np.floor(
                    (lon.shape[1] * (bbox[2] - bbox[0])) / (w * abs(np.max(lon) - np.amin(lon))))))
                jstep = max(1, int(np.floor(
                    (lon.shape[0] * (bbox[3] - bbox[1])) / (h * abs(np.max(lat) - np.amin(lat))))))

                X = lon[j0:j1:jstep, i0:i1:istep]
                Y = lat[j0:j1:jstep, i0:i1:istep]
                data = grid.array[..., j0:j1:jstep, i0:i1:istep]

                # apply time slices
                if l is not None:
                    data = np.asarray(
                        grid.array[np.where(l)[0][0], j0:j1:1, i0:i1:1])
                else:
                    data = np.asarray(grid.array[..., j0:j1:1, i0:i1:1])

                extent = (
                    X[0, 0], X[X.shape[0] - 1, 0], Y[0, Y.shape[1] - 1], Y[0, 0])

            # Plot data.
            if data.shape:
                # reduce dimensions and mask missing_values
                data = fix_data(data, grid.attributes)

                ax.imshow(data, extent = extent,
                          vmin = actual_range[0], vmax = actual_range[
                              1], cmap = get_cmap(cmap),
                          interpolation = 'none')
            lon += 360.0

    def _get_capabilities(self, req):
        def serialize(dataset):
            fix_map_attributes(dataset)
            grids = [
                grid for grid in walk(dataset, GridType) if is_valid(grid, dataset)]

            # Set global lon/lat ranges.
            try:
                lon_range = self.cache.get_value('lon_range')
            except (KeyError, AttributeError):
                try:
                    lon_range = dataset.attributes['NC_GLOBAL']['lon_range']
                except KeyError:
                    lon_range = [np.inf, -np.inf]
                    for grid in grids:
                        lon = np.asarray(get_lon(grid, dataset)[:])
                        lon_range[0] = min(lon_range[0], np.min(lon))
                        lon_range[1] = max(lon_range[1], np.max(lon))
                if self.cache:
                    self.cache.set_value('lon_range', lon_range)
            try:
                lat_range = self.cache.get_value('lat_range')
            except (KeyError, AttributeError):
                try:
                    lat_range = dataset.attributes['NC_GLOBAL']['lat_range']
                except KeyError:
                    lat_range = [np.inf, -np.inf]
                    for grid in grids:
                        lat = np.asarray(get_lat(grid, dataset)[:])
                        lat_range[0] = min(lat_range[0], np.min(lat))
                        lat_range[1] = max(lat_range[1], np.max(lat))
                if self.cache:
                    self.cache.set_value('lat_range', lat_range)

            # Remove ``REQUEST=GetCapabilites`` from query string.
            location = req.url
            base = location.split('REQUEST=')[0].rstrip('?&')

            # Get layer bboxes
            layer_info_dict = {}
            for grid in grids:
                g_lon = list(get_lon(grid, dataset))
                g_lat = list(get_lat(grid, dataset))
                time = get_time(grid, dataset)
                minx, maxx = np.min(g_lon), np.max(g_lon)
                miny, maxy = np.min(g_lat), np.max(g_lat)

                # Get actual data range for levels.
                actual_range, range_guest = self._get_actual_range(grid)

                layer_inf = {
                    'time': time, 'minx': minx, 'maxx': maxx, 'miny': miny, 'maxy': maxy, 'actual_range': actual_range, 'range_guest': range_guest}
                layer_info_dict[grid._id] = layer_inf

            context = {
                'dataset': dataset,
                'location': base,
                'layers': grids,
                'lon_range': lon_range,
                'lat_range': lat_range,
                'layer_info': layer_info_dict
            }

            output = self.__template__.render(context)
            if hasattr(dataset, 'close'):
                dataset.close()
            return [output.encode('utf-8')]
        return serialize(self.dataset)


def is_valid(grid, dataset):
    return (get_lon(grid, dataset) is not None and
            get_lat(grid, dataset) is not None)


def get_lon(grid, dataset):
    def check_attrs(var):
        if (re.match('degrees?_e', var.attributes.get('units', ''), re.IGNORECASE) or
                var.attributes.get('axis', '').lower() == 'x' or
                var.attributes.get('standard_name', '') == 'longitude'):
            return var

    # check maps first
    for dim in grid.maps.values():
        if check_attrs(dim) is not None:
            return dim

    # check curvilinear grids
    if hasattr(grid, 'coordinates'):
        coords = grid.coordinates.split()
        for coord in coords:
            if coord in dataset and check_attrs(dataset[coord].array) is not None:
                return dataset[coord].array

    return None


def get_lat(grid, dataset):
    def check_attrs(var):
        if (re.match('degrees?_n', var.attributes.get('units', ''), re.IGNORECASE) or
                var.attributes.get('axis', '').lower() == 'y' or
                var.attributes.get('standard_name', '') == 'latitude'):
            return var

    # check maps first
    for dim in grid.maps.values():
        if check_attrs(dim) is not None:
            return dim

    # check curvilinear grids
    if hasattr(grid, 'coordinates'):
        coords = grid.coordinates.split()
        for coord in coords:
            if coord in dataset and check_attrs(dataset[coord].array) is not None:
                return dataset[coord].array

    return None


def get_time(grid, dataset):
    for dim in grid.maps.values():
        if ' since ' in dim.attributes.get('units', ''):
            try:
                return [coards.parse(value, dim.attributes.get('units')) for value in np.asarray(dim.data)]
            except:
                pass

    return None


def fix_data(data, attrs):
    if 'missing_value' in attrs:
        data = np.ma.masked_equal(data, attrs['missing_value'])
    elif '_FillValue' in attrs:
        data = np.ma.masked_equal(data, attrs['_FillValue'])

    if attrs.get('scale_factor'):
        data = data * float(attrs['scale_factor'])
    if attrs.get('add_offset'):
        data += attrs['add_offset']

    while len(data.shape) > 2:
        # data = data[0]
        data = np.ma.mean(data, 0)
    return data


def fix_map_attributes(dataset):
    # @TODO: Get all variables of type GridType from dataset
    for grid in walk(dataset, GridType):
        for map_ in grid.maps.values():
            if not map_.attributes and map_.name in dataset:
                map_.attributes = dataset[map_.name].attributes.copy()


def find_containing_bounds(axis, v0, v1):
    """
    Find i0, i1 such that axis[i0:i1] is the minimal array with v0 and v1.

    For example::

        >>> from numpy import *
        >>> a = arange(10)
        >>> i0, i1 = find_containing_bounds(a, 1.5, 6.5)
        >>> print a[i0:i1]
        [1 2 3 4 5 6 7]
        >>> i0, i1 = find_containing_bounds(a, 1, 6)
        >>> print a[i0:i1]
        [1 2 3 4 5 6]
        >>> i0, i1 = find_containing_bounds(a, 4, 12)
        >>> print a[i0:i1]
        [4 5 6 7 8 9]
        >>> i0, i1 = find_containing_bounds(a, 4.5, 12)
        >>> print a[i0:i1]
        [4 5 6 7 8 9]
        >>> i0, i1 = find_containing_bounds(a, -4, 7)
        >>> print a[i0:i1]
        [0 1 2 3 4 5 6 7]
        >>> i0, i1 = find_containing_bounds(a, -4, 12)
        >>> print a[i0:i1]
        [0 1 2 3 4 5 6 7 8 9]
        >>> i0, i1 = find_containing_bounds(a, 12, 19)
        >>> print a[i0:i1]
        []

    It also works with decreasing axes::

        >>> b = a[::-1]
        >>> i0, i1 = find_containing_bounds(b, 1.5, 6.5)
        >>> print b[i0:i1]
        [7 6 5 4 3 2 1]
        >>> i0, i1 = find_containing_bounds(b, 1, 6)
        >>> print b[i0:i1]
        [6 5 4 3 2 1]
        >>> i0, i1 = find_containing_bounds(b, 4, 12)
        >>> print b[i0:i1]
        [9 8 7 6 5 4]
        >>> i0, i1 = find_containing_bounds(b, 4.5, 12)
        >>> print b[i0:i1]
        [9 8 7 6 5 4]
        >>> i0, i1 = find_containing_bounds(b, -4, 7)
        >>> print b[i0:i1]
        [7 6 5 4 3 2 1 0]
        >>> i0, i1 = find_containing_bounds(b, -4, 12)
        >>> print b[i0:i1]
        [9 8 7 6 5 4 3 2 1 0]
        >>> i0, i1 = find_containing_bounds(b, 12, 19)
        >>> print b[i0:i1]
        []
    """
    ascending = axis[1] > axis[0]
    if not ascending:
        axis = axis[::-1]
    i0 = i1 = len(axis)
    for i, value in enumerate(axis):
        if value > v0 and i0 == len(axis):
            i0 = i - 1
        if not v1 > value and i1 == len(axis):
            i1 = i + 1
    if not ascending:
        i0, i1 = len(axis) - i1, len(axis) - i0
    return max(0, i0), min(len(axis), i1)


def walk(var, type_ = object):
    """
    Yield all variables of a given type from a dataset.
    The iterator returns also the parent variable.
    """
    if isinstance(var, type_):
        yield var
    if isinstance(var, StructureType):
        for child in var._dict:
            for subvar in walk(var._dict[child], type_):
                yield subvar


def asbool(obj):
    if isinstance(obj, (str, unicode)):
        obj = obj.strip().lower()
        if obj in ['true', 'yes', 'on', 'y', 't', '1']:
            return True
        elif obj in ['false', 'no', 'off', 'n', 'f', '0']:
            return False
        else:
            raise ValueError("String is not true/false: %r" % obj)
    return bool(obj)
