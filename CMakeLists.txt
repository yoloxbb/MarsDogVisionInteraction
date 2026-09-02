cmake_minimum_required(VERSION 3.8)
project(marsdog_vision_interaction)

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(rclpy REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)

rosidl_generate_interfaces(
  ${PROJECT_NAME}
  "srv/VisionTask.srv"
)

install(
  DIRECTORY ${PROJECT_NAME}/
  DESTINATION ${PYTHON_INSTALL_DIR}/${PROJECT_NAME}
  PATTERN "srv" EXCLUDE
  PATTERN "__pycache__" EXCLUDE
  PATTERN "*.pyc" EXCLUDE
)

install(DIRECTORY config launch DESTINATION share/${PROJECT_NAME})

install(PROGRAMS
  scripts/vision_interaction
  scripts/camera_driver
  scripts/vision_debug_viewer
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
