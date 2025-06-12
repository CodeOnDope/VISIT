from setuptools import setup, find_packages 
 
setup( 
    name="visit_museum_tracker", 
    version="0.1.0", 
    packages=find_packages(), 
    install_requires=[ 
        "opencv-python>=4.5.0", 
        "mediapipe>=0.8.9", 
        "PyQt5>=5.15.0", 
        "numpy>=1.19.0", 
        "pillow>=8.0.0", 
        "sqlalchemy>=1.4.0", 
        "python-dotenv>=0.19.0", 
    ], 
) 
