from setuptools import setup, find_packages

setup(
    name='ffhflow',
    version='0.1.0',
    description='FFHFlow',
    author='Qian Feng',
    author_email='qian.feng@tum.de',
    # url='https://blog.godatadriven.com/setup-py',
    packages=find_packages(include=['ffhflow', 'ffhflow.heads', 'ffhflow.backbones', 'ffhflow.configs', 'ffhflow.datasets',
                                    'ffhflow.heads', 'ffhflow.utils'])
)