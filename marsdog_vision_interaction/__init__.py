"""MarsDog visual interaction ROS2 package."""

from pkgutil import extend_path


# Source execution from the repository must still see the ``srv`` subpackage
# generated into a sourced colcon install prefix.  Without an extended package
# path, the source directory shadows the generated ROSIDL Python modules.
__path__ = extend_path(__path__, __name__)

__version__ = "0.1.1"
