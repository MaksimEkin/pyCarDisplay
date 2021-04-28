import os
from setuptools import setup, find_packages
import subprocess
import pkg_resources
__version__ = "0.0.3"

# HELPER CLASS AND FUNCTIONS
class BColors:
    """Colors for printing"""
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

# enforce pip version
pkg_resources.require(['pip >= 20.0.0'])

# add readme
with open('README.md', 'r') as f:
    LONG_DESCRIPTION = f.read()

# add dependencies
with open('requirements.txt', 'r') as f:
    INSTALL_REQUIRES = f.read().strip().split('\n')

setup(
    name='pyCarDisplay',
    version=__version__,

    author='Ryan Barron, Maksim E. Eren, Charles Varga, and Wei Wang',
    author_email='meren1@umbc.edu',
    description='A car display.',
    long_description=LONG_DESCRIPTION,
    long_description_content_type='text/markdown',
    platforms = ["Linux", "Mac"],
    include_package_data=True,
    url='https://github.com/MaksimEkin/pyCarDisplay',
    packages=find_packages(),
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Programming Language :: Python :: 3.8',
        'Topic :: Software Development :: Libraries'
    ],
    python_requires='>=3.8.5',
    install_requires=INSTALL_REQUIRES,
    license='License :: Apache2 License',
    zip_safe=False
)


# Done
print(f'{BColors.OKGREEN}\tFinished installing.{BColors.ENDC}')
