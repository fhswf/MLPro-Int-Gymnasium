from setuptools import setup


setup(name='mlpro-int-gymnasium',
version='0.1.0',
description='MLPro: Integration Gymnasium',
author='MLPro Team',
author_mail='mlpro@listen.fh-swf.de',
license='Apache Software License (http://www.apache.org/licenses/LICENSE-2.0)',
packages=['mlpro_int_gymnasium'],

# Package dependencies for full installation
extras_require={
    "full": [
        "mlpro[full]>=1.3.1",
        "gymnasium[all]>=0.29",
        "pandas>=2.1.3"
    ],
},

zip_safe=False)