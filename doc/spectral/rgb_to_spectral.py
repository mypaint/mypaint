# -*- coding: utf-8 -*-
"""RGB_to_Spectral

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/16mwE57LUxwMQFwyHuZ2I--fD9YgS0uK8
"""

# !pip install git+git://github.com/colour-science/colour

import colour
import numpy as np
from scipy.optimize import minimize


from colour.colorimetry import (
    STANDARD_OBSERVERS_CMFS,
    SpectralDistribution,
    SpectralShape,
    sd_ones,
    spectral_to_XYZ_integration,
    sd_zeros,
)
from colour.utilities import to_domain_1, from_range_100


# could use straight Meng but we can do Chromatic Adaptation via the illuminant_SPD instead
# this is Meng modified to be more similar to Scott Allen Burns least log slope squared method
# constrain for max refl


def XYZ_to_spectral(
    XYZ,
    cmfs=STANDARD_OBSERVERS_CMFS["CIE 1931 2 Degree Standard Observer"],
    interval=5,
    tolerance=1e-10,
    maximum_iterations=5000,
    illuminant=sd_ones(),
    max_refl=1.0,
):

    XYZ = to_domain_1(XYZ)
    shape = SpectralShape(cmfs.shape.start, cmfs.shape.end, interval)
    cmfs = cmfs.copy().align(shape)
    illuminant = illuminant.copy().align(shape)
    spd = sd_zeros(shape)

    def function_objective(a):
        """
        Objective function.
        """

        return np.sum(np.diff(a) ** 2)

    def function_constraint(a):
        """
        Function defining the constraint for XYZ=XYZ.
        """

        spd[:] = np.exp(a)

        return XYZ - (
            spectral_to_XYZ_integration(spd, cmfs=cmfs, illuminant=illuminant)
        )

    def function_constraint2(a):
        """
        Function defining constraint on emission/reflectance
        """
        if max_refl <= 0.0:
            return 0.0
        return max_refl - np.exp(np.max(a)) * 100.0

    wavelengths = spd.wavelengths
    bins = wavelengths.size
    constraints = (
        {"type": "eq", "fun": function_constraint},
        {"type": "ineq", "fun": function_constraint2},
    )

    result = minimize(
        function_objective,
        spd.values,
        method="SLSQP",
        constraints=constraints,
        options={
            "ftol": tolerance,
            "maxiter": maximum_iterations,
            "disp": True,
        },
    )

    if not result.success:
        raise RuntimeError(
            'Optimization failed for {0} after {1} iterations: "{2}".'.format(
                XYZ, result.nit, result.message
            )
        )

    return SpectralDistribution(
        from_range_100(np.exp(result.x) * 100),
        wavelengths,
        name="Meng (2015) - {0}".format(XYZ),
    )


# Define our illuminant and color space matrices

D65_xy = colour.ILLUMINANTS["cie_2_1931"]["D65"]
D65 = colour.xy_to_XYZ(D65_xy)

CMFS = colour.CMFS["cie_2_1931"]
D65_SPD = colour.ILLUMINANTS_SDS["D65"] / 100.0
# illuminant_SPD = colour.XYZ_to_spectral_Meng2015(D65, cmfs=CMFS, interval=shape.interval)
illuminant_SPD = D65_SPD

np.set_printoptions(formatter={"float": "{:0.15f}".format}, threshold=np.nan)

colorspace = colour.models.BT709_COLOURSPACE

colorspace.use_derived_transformation_matrices(True)


RGB_to_XYZ_m = colorspace.RGB_to_XYZ_matrix
XYZ_to_RGB_m = colorspace.XYZ_to_RGB_matrix

# for performance use a larger interval.  Harder to solve, must raise tol

interval = 40
shape = SpectralShape(380.0, 730.0, interval)

# spd via Meng-ish Burns-ish recovery
target_XYZ = colour.sRGB_to_XYZ([1, 0, 0])
spd = XYZ_to_spectral(
    target_XYZ,
    cmfs=CMFS.align(shape),
    illuminant=illuminant_SPD,
    interval=interval,
    tolerance=1e-8,
    max_refl=1.00,
)
print("red SPD is", spd.values)

target_XYZ = colour.sRGB_to_XYZ([0, 1, 0])
spd = XYZ_to_spectral(
    target_XYZ,
    cmfs=CMFS.align(shape),
    illuminant=illuminant_SPD,
    interval=interval,
    tolerance=1e-8,
    max_refl=1.00,
)
print("green SPD is", spd.values)


target_XYZ = colour.sRGB_to_XYZ([0, 0, 1])
spd = XYZ_to_spectral(
    target_XYZ,
    cmfs=CMFS.align(shape),
    illuminant=illuminant_SPD,
    interval=interval,
    tolerance=1e-8,
    max_refl=1.00,
)
print("blue SPD is", spd.values)

CMFS_ = CMFS.align(spd.shape).values.transpose()  # align and transpose the CMFS
illuminant_SPD_ = illuminant_SPD.align(spd.shape).values  # align illuminant vector

print(
    "Spectral to RGB Matrix is",
    np.matmul(
        XYZ_to_RGB_m,
        np.matmul(CMFS_, np.diag(illuminant_SPD_))  # weight for whitepoint
        / np.matmul(CMFS_[1], illuminant_SPD_),
    ),
)
