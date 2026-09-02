from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def generate_launch_description() -> LaunchDescription:
    share = FindPackageShare("marsdog_vision_interaction")
    config_path = LaunchConfiguration("config_path")
    pose_model_variant = LaunchConfiguration("pose_model_variant")
    landmarker_running_mode = LaunchConfiguration("landmarker_running_mode")
    face_api_host = LaunchConfiguration("face_api_host")
    face_api_port = LaunchConfiguration("face_api_port")
    return LaunchDescription([
        DeclareLaunchArgument(
            "config_path",
            default_value=PathJoinSubstitution([share, "config", "vision.yaml"]),
        ),
        DeclareLaunchArgument("pose_model_variant", default_value=""),
        DeclareLaunchArgument("landmarker_running_mode", default_value=""),
        DeclareLaunchArgument("face_api_host", default_value=""),
        DeclareLaunchArgument("face_api_port", default_value="0"),
        Node(
            package="marsdog_vision_interaction",
            executable="vision_interaction",
            name="vision_interaction",
            output="screen",
            parameters=[{
                "config_path": config_path,
                "pose_model_variant": pose_model_variant,
                "landmarker_running_mode": landmarker_running_mode,
                "face_api_host": ParameterValue(
                    face_api_host, value_type=str
                ),
                "face_api_port": face_api_port,
            }],
        ),
    ])
