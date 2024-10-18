from setuptools import setup

setup(name='injspection',
      version='0.0.2',
      url='https://github.com/askap-craco/injspection',
      author='Joscha Jahns-Schindler',
      author_email='jjahnsschindler@swin.edu.au',
      description='Injection inspection',
      packages=['injspection'],
      scripts=['bin/injspect.py', 'bin/injspect_all_injs_ever.py'],
      install_requires=[],
)