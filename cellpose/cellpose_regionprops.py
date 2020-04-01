import pandas as pd
from glob import glob
import mxnet
from cellpose import models
import matplotlib.pyplot as plt
from skimage import io, measure, segmentation


model = models.Cellpose(device=mxnet.cpu(), model_type='cyto')

files_raw = sorted(glob('*/*/*.tif'))
files = list(filter(lambda f: f.startswith('wt') or f.startswith('mut'), files_raw))
images = map(io.imread, files)

channels = [0, 0]

scale = 0.65  # µm per pixel
header = True


for filename, image in zip(files, images):
    print(f'{filename} started')

    # get cell mask
    masks, _, _, _ = model.eval([image], rescale=None, channels=channels)
    mask = segmentation.clear_border(masks[0])

    # make and save dataframe
    props = pd.DataFrame(
        measure.regionprops_table(
            mask, properties=('label', 'area', 'centroid')
        )
    ).set_index('label')
    props['filename'] = filename
    props['type'] = 'wt' if filename.startswith('wt') else 'mut'
    props['area (µm²)'] = props['area'] * (scale**2)
    props.to_csv('out.csv', mode='a', header=header)
    header = False

    # make and save figure
    marked = segmentation.mark_boundaries(
        image, mask, color=(0, 0, 0), mode='thick'
    )
    dpi = 80
    figsize = (image.shape[1] / dpi, image.shape[0] / dpi)
    fig = plt.Figure(figsize=figsize, dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(marked, interpolation='nearest')
    for label, (area, y, x, fn) in props.iterrows():
        ax.text(x, y, str(int(label)),
                verticalalignment='center',
                horizontalalignment='center')
    output_filename = filename[:-4] + '_segmentation.png'
    fig.savefig(output_filename)

    print(f'{filename} done')
