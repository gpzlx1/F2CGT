from setuptools import setup, find_packages
import os

setup(name="pagraph", packages=find_packages(), include_package_data=True, 
    data_files=[('pagraph', ['../build/libpg.so'])])