from setuptools import setup
from catkin_pkg.python_setup import generate_distutils_setup
d = generate_distutils_setup(
    packages=['det_img_seg'],
    package_dir={'': 'python'}
)
setup(**d)
