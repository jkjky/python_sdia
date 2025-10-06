from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy

extensions = [
    Extension(
        "C_accelerations.knn_cython",
        ["KNN_cy.pyx"],
        include_dirs=[numpy.get_include()],
    )
]

setup(
    name="KNearest_cy",
    ext_modules=cythonize(
        extensions,
        annotate=True,
        language_level="3",
    ),
)
