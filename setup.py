from distutils.core import setup

setup(
    name='Electra Processing',
    version='1.0',
    description='An application to process Electra .ele files, GUI or CLI',
    author='Luca Peruzzo',
    url='https://github.com/Peruz/ElectraProcessing.git',
    packages=['electra_processing'],
    package_dir={'electra_processing': './electra_processing'},
    package_data={'electra_processing': ['tests/wen.ele', 'grad_rec.ele', 'sch.ele']},
    install_requires=[
        'PySide6>=6.3.1',
        'pandas>=1.4.3',
        'numpy',
        'numba',
        'matplotlib',
        'datetime',
        'argparse',

    ]
)
