from pathlib import Path

from setuptools import setup, find_packages

package_path = Path(__file__).parent

version_path = Path(__file__).parent.joinpath(f"./VERSION")
version = version_path.read_text().strip()
install_requires = (
    Path(__file__).parent.joinpath("requirements.txt").read_text().splitlines()
)
data_files = [
    str(version_path.relative_to(package_path)),
]

setup(
    name="PushNet",
    version=version,
    description='Install dependencies for PushNet',
    author='Akash Manna',
    author_email='manna.akash@emagegroup.com',
    install_requires=install_requires,
    # package_dir={"": "src"},
    packages=find_packages("./"),
    include_package_data=True,
    data_files=data_files,
    package_data={'PushNet': ['data/target_pose.png']}
)