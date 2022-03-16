# coding: utf-8

from setuptools import setup, find_packages
import pathlib

here = pathlib.Path(__file__).parent.resolve()

#long_description = (here / 'README.md').read_text(encoding='utf-8')

required_packages = list(filter(
    lambda x:not (x == '' or x.startswith('#')),
    [_.strip() for _ in (here / 'requirements.txt').read_text().split('\n')]
))

setup(
    name='bertbui',  # Required
    version='1.0.0',  # Required
    description='bertbui dev',
    #long_description=long_description,
    long_description_content_type='text/markdown',
    #url='https://github.com/pypa/sampleproject',  # Optional
    author='T. Iki',
    #author_email='author@example.com',  # Optional
    classifiers=[  # Optional
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        'Development Status :: 3 - Alpha',
    ],
    #keywords='sample, setuptools, development',  # Optional
    package_dir={'': 'src'},  # Optional
    packages=find_packages(where='src'),  # Required
    
    python_requires='>=3.6, <4',
    
    install_requires=required_packages,  # Optional

    #package_data={  # Optional
    #    'sample': ['package_data.dat'],
    #},
    #
    # data_files=[('my_data', ['data/data_file'])],  # Optional
    #
    #entry_points={  # Optional
    #    'console_scripts': [
    #        'sample=sample:main',
    #    ],
    #},
    #project_urls={  # Optional
    #    'Bug Reports': 'https://github.com/pypa/sampleproject/issues',
    #    'Funding': 'https://donate.pypi.org',
    #    'Say Thanks!': 'http://saythanks.io/to/example',
    #    'Source': 'https://github.com/pypa/sampleproject/',
    #},
)

