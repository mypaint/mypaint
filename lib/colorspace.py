
def float_rgb_lin_to_srgb(src, dst):
    # sRGB gamma correction
    # TODO: not correct, but close enough for a first impression
    assert src.shape[2] == 3
    dst[:] = 255*src**(1/2.2)

def srgb_to_float_rgb_lin(src, dst):
    # same as above
    dst[:] = (src**(2.2))/255.0
