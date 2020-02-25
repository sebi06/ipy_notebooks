from aicsimageio import AICSImage, dask_utils

# Create a local dask cluster and client for the duration of the context manager
with AICSImage("filename.ome.tiff") as img:
    # do your work like normal
    print(img.dask_data.shape)

# Specify arguments for the local cluster initialization
with AICSImage("filename.ome.tiff", dask_kwargs={"nworkers": 4}) as img:
    # do your work like normal
    print(img.dask_data.shape)

# Connect to a dask client for the duration of the context manager
with AICSImage("filename.ome.tiff", dask_kwargs={"address": "tcp://localhost:12345"}) as img:
    # do your work like normal
    print(img.dask_data.shape)

# Or spawn a local cluster and / or connect to a client outside of a context manager
# This uses the same "address" and dask kwargs as above
# If you pass an address in, it will create and shutdown the client and no cluster will be created.
# Similar to AICSImage, these objects will be connected and useable for the lifespan of the context manager.
with dask_utils.cluster_and_client() as (cluster, client):

    img1 = AICSImage("1.tiff")
    img2 = AICSImage("2.tiff")
    img3 = AICSImage("3.tiff")

    # Do your image processing work
