from dask import delayed
import dask.array as da
import napari
from vispy.color import Colormap
from omero.gateway import BlitzGateway

conn = BlitzGateway("username", "password", port=4064, host="localhost")
conn.connect()
conn.SERVICE_OPTS.setOmeroGroup('-1')

IMAGE_ID = 4424
image = conn.getObject("Image", IMAGE_ID)
print(image.name)
cache = {}

def get_lazy_stack(img, c=0):
    sz = img.getSizeZ()
    st = img.getSizeT()
    plane_names = ["%s,%s,%s" % (z, c, t) for t in range(st) for z in range(sz)]

    def get_plane(plane_name):
        if plane_name in cache:
            return cache[plane_name]
        z, c, t = [int(n) for n in plane_name.split(",")]
        print('get_plane', z, c, t)
        pixels = img.getPrimaryPixels()
        p = pixels.getPlane(z, c, t)
        cache[plane_name] = p
        return p

    # read the first file to get the shape and dtype
    # ASSUMES THAT ALL FILES SHARE THE SAME SHAPE/TYPE
    sample = get_plane(plane_names[0])

    lazy_imread = delayed(get_plane)  # lazy reader
    lazy_arrays = [lazy_imread(pn) for pn in plane_names]
    dask_arrays = [
        da.from_delayed(delayed_reader, shape=sample.shape, dtype=sample.dtype)
        for delayed_reader in lazy_arrays
    ]
    # Stack into one large dask.array
    if sz == 1 or st == 1:
        return da.stack(dask_arrays, axis=0)

    z_stacks = []
    for t in range(st):
        z_stacks.append(da.stack(dask_arrays[t * sz: (t + 1) * sz], axis=0))
    stack = da.stack(z_stacks, axis=0)
    return stack


with napari.gui_qt():
    # specify contrast_limits and is_pyramid=False with big data
    # to avoid unecessary computations
    viewer = napari.Viewer()

    for c, channel in enumerate(image.getChannels()):
        print('loading channel %s' % c)
        data = get_lazy_stack(image, c=c)
        # use current rendering settings from OMERO
        start = channel.getWindowStart()
        end = channel.getWindowEnd()
        color = channel.getColor().getRGB()
        color = [r/256 for r in color]
        cmap = Colormap([[0, 0, 0], color])
        # Z-scale for 3D viewing
        z_scale = 1
        if image.getSizeZ() > 1:
            size_x = image.getPixelSizeX()
            size_z = image.getPixelSizeZ()
            if size_x is not None and size_z is not None:
                z_scale = [1, size_z / size_x, 1, 1]
        viewer.add_image(data, blending='additive',
                        contrast_limits=[start, end],
                        is_pyramid=False,
                        colormap=('from_omero', cmap),
                        # scale=[1, z_scale, 1, 1],
                        name=channel.getLabel())

print('closing conn...')
conn.close()