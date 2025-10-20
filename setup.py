from setuptools import setup, find_packages
from setuptools.command.install import install
import subprocess
import sys

class PostInstallCommand(install):
    def run(self):
        install.run(self)
        try:
            subprocess.check_call([
                sys.executable,
                '-m', 'pyspice_post_installation',
                '--install-ngspice-dll'
            ])
            print("✅ ngspice DLL successfully installed.")
        except Exception as e:
            print("⚠️ Failed to install ngspice DLL automatically.")
            print("Please run manually: pyspice-post-installation --install-ngspice-dll")
            print("Error:", e)

setup(
    name='llc_simulator',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'PySpice',
        'matplotlib',
        'numpy',
        'tabulate',
        'pandas',
        
    ],
    description='Simulador de conversores LLC usando PySpice',
    author='GSEC',
    python_requires='>=3.7',
    cmdclass={
        'install': PostInstallCommand,
    }
)



