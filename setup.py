from glob import glob

from setuptools import find_packages, setup


package_name = "marsdog_vision_interaction"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["tests"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/config", glob("config/*.yaml")),
        ("share/" + package_name + "/launch", glob("launch/*.py")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="MarsDog Vision Team",
    maintainer_email="noreply@marsdog.dev",
    description="MarsDog visual interaction ROS2 node",
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            "marsdog-vision-interaction = marsdog_vision_interaction.main:main",
            "marsdog-camera-driver = marsdog_vision_interaction.nodes.camera_driver_node:main",
            "marsdog-vision-viewer = marsdog_vision_interaction.nodes.vision_debug_viewer_node:main",
        ],
    },
)
