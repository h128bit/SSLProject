from setuptools import setup, find_packages


setup(
    name="sslproject",                  
    version="1.0.4",                          
    author="h128bit",
    description="Flex Implementation of self supervised learning methods ",
    url="https://github.com/h128bit/SSLProject",

    packages=find_packages(),                 
   
    install_requires=[
    "albumentations>=1.3.0",
    "matplotlib>=3.5.0",
    "numpy>=2",
    "pandas>=1.4.0",
    "Pillow>=9.0.0",        
    "timm>=0.9.0",
    "torch>=2.0.0",
    "torchvision>=0.15.0",
    "tqdm>=4.60.0",
    "info-nce-pytorch"
    ],

    python_requires=">=3.10",

    classifiers=[
        "Programming Language :: Python :: 3",
    ],
)