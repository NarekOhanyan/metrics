#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import setuptools


# In[ ]:


with open("README.md", "r") as fh:
    long_description = fh.read()


# In[ ]:


setuptools.setup(
    name="metrics",
    version="0.0.1",
    author="Narek Ohanyan",
    author_email="ohanyannarek@example.com",
    description="Econometrics library for Python",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/NarekOhanyan/metrics",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)

