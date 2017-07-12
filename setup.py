from setuptools import setup
from setuptools import find_packages

install_requires = [
    'Keras',
]

setup(
      name='keras-aquarium',
      version='0.1.0',
      description='A small collection of models for matrix factorization, topic modeling, text classification with keras',
      author='Aaron Yin',
      author_email='919155161@qq.com',
      url='https://github.com/AlonAzrael/',
      license='MIT',
      install_requires=install_requires,
      packages=find_packages(),
    #   dependency_links=['']
)
