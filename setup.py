import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()
setuptools.setup(
     name='autopandas',
     version='0.1',
     scripts=['script'] ,
     author="Adrien Pavao",
     author_email="adrien.pavao@gmail.com",
     description="Process, visualize and use data easily.",
     long_description=long_description,
     long_description_content_type="text/markdown",
     url="https://github.com/didayolo/autopandas",
     packages=setuptools.find_packages(),
     classifiers=[
         "Programming Language :: Python :: 3",
         "Operating System :: Unix",
     ],
 )
