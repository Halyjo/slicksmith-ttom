def compute_res_for_shape(sources, target_shape=(100, 100)):
    """
    Compute a (xres, yres) tuple that will make rasterio.merge
    produce `target_shape` when no alignment snapping is requested.

    Parameters
    ----------
    sources : sequence of opened rasterio datasets
    target_shape : (int, int)
        Desired (width, height) in pixels.

    Returns
    -------
    res : (float, float)
        Pixel sizes to pass to rasterio.merge(..., res=res)
    bounds : (left, bottom, right, top)
        The overall bounds you may also want to pass explicitly.
    """
    # 1. Collect the envelope of every scene
    lefts, rights, bottoms, tops = [], [], [], []
    for src in sources:
        left, bottom, right, top = src.bounds
        lefts.append(left)
        rights.append(right)
        bottoms.append(bottom)
        tops.append(top)

    w, s, e, n = min(lefts), min(bottoms), max(rights), max(tops)
    width, height = target_shape

    # 2. Solve for the pixel size that gives that many pixels
    xres = (e - w) / float(width)
    yres = (n - s) / float(height)

    return (xres, yres), (w, s, e, n)
