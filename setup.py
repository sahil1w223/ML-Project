from setuptools import setup, find_packages


def selected_libraries(path):
    libs = []
    with open(path, 'r') as f:
        libs = [line.strip() for line in f if line.strip()]
    
    if '-e .' in libs:
        libs.remove('-e .')
    
    return libs


setup(
    name="ml-model",
    version="0.0.1",
    author="sahil jal",
    author_email="sahiljal@gmail.com",
    packages=find_packages(),
    install_requires=selected_libraries("requirements.txt")
)