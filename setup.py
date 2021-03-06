from setuptools import find_packages, setup
import autopandas

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
     name='autopandas',
     version=autopandas.__version__,
     author="Adrien Pavao",
     author_email="adrien.pavao@gmail.com",
     description="Process, visualize and use data easily.",
     long_description=long_description,
     long_description_content_type="text/markdown",
     url="https://github.com/didayolo/autopandas",
     packages=find_packages(),
     include_package_data=True,
     classifiers=[
         "Programming Language :: Python :: 3",
         "Operating System :: Unix",
     ],
 )
