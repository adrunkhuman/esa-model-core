import numpy
from setuptools import Extension, setup

setup(
    ext_modules=[
        Extension(
            "esa_model._motivation_c",
            ["esa_model/_motivation_c.c"],
            include_dirs=[numpy.get_include()],
            optional=True,
        )
    ]
)
