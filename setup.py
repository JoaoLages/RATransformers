from setuptools import setup, find_packages
import pathlib

here = pathlib.Path(__file__).parent.resolve()

# Get the long description from the README file
long_description = (here / 'README.md').read_text(encoding='utf-8')

# Arguments marked as "Required" below must be included for upload to PyPI.
# Fields marked as "Optional" may be commented out.
setup(
    name='ratransformers',
    version='0.0.0',
    # description='A sample Python project',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/JoaoLages/RATransformers',
    author='Joao Lages',
    author_email='joaop.glages@gmail.com',

    classifiers=[
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Topic :: Utilities',

        # Pick your license as you wish
        # 'License :: OSI Approved :: MIT License',

        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
    ],
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    python_requires='>=3.6, <4'
)