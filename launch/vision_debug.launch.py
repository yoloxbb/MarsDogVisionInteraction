from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def generate_launch_description() -> LaunchDescription:
    share = FindPackageShare("marsdog_vision_interaction")
    start_vision_node = LaunchConfiguration("start_vision_node")
    config_path = LaunchConfiguration("config_path")
    pose_model_variant = LaunchConfiguration("pose_model_variant")
    landmarker_running_mode = LaunchConfiguration("landmarker_running_mode")
    face_api_host = LaunchConfiguration("face_api_host")
    face_api_port = LaunchConfiguration("face_api_port")
    camera_topic = LaunchConfiguration("camera_topic")
    visual_topic = LaunchConfiguration("visual_topic")
    gesture_debug_topic = LaunchConfiguration("gesture_debug_topic")
    vision_task_service = LaunchConfiguration("vision_task_service")
    object_topic = LaunchConfiguration("object_topic")
    web_host = LaunchConfiguration("web_host")
    web_port = LaunchConfiguration("web_port")
    show_window = LaunchConfiguration("show_window")
    publish_debug_image = LaunchConfiguration("publish_debug_image")
    max_render_fps = LaunchConfiguration("max_render_fps")
    render_scale = LaunchConfiguration("render_scale")
    jpeg_quality = LaunchConfiguration("jpeg_quality")
    event_history_limit = LaunchConfiguration("event_history_limit")

    return LaunchDescription([
        DeclareLaunchArgument("start_vision_node", default_value="true"),
        DeclareLaunchArgument(
            "config_path",
            default_value=PathJoinSubstitution([share, "config", "vision.yaml"]),
        ),
        DeclareLaunchArgument("pose_model_variant", default_value=""),
        DeclareLaunchArgument("landmarker_running_mode", default_value=""),
        DeclareLaunchArgument("face_api_host", default_value=""),
        DeclareLaunchArgument("face_api_port", default_value="0"),
        DeclareLaunchArgument(
            "camera_topic",
            default_value="/camera/camera/color/image_raw",
        ),
        DeclareLaunchArgument(
            "visual_topic",
            default_value="/perception/visual_event",
        ),
        DeclareLaunchArgument(
            "gesture_debug_topic",
            default_value="/perception/vision/gesture_debug",
        ),
        DeclareLaunchArgument(
            "vision_task_service",
            default_value="/perception/vision/task",
        ),
        DeclareLaunchArgument("web_host", default_value="127.0.0.1"),
        DeclareLaunchArgument("web_port", default_value="8765"),
        DeclareLaunchArgument("show_window", default_value="false"),
        DeclareLaunchArgument("publish_debug_image", default_value="false"),
        DeclareLaunchArgument("max_render_fps", default_value="8.0"),
        DeclareLaunchArgument("render_scale", default_value="0.75"),
        DeclareLaunchArgument("jpeg_quality", default_value="75"),
        DeclareLaunchArgument(
            "object_topic",
            default_value="/perception/vision/object_detections",
        ),
        DeclareLaunchArgument("event_history_limit", default_value="200"),
        Node(
            package="marsdog_vision_interaction",
            executable="vision_interaction",
            name="vision_interaction",
            output="screen",
            condition=IfCondition(start_vision_node),
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
        Node(
            package="marsdog_vision_interaction",
            executable="vision_debug_viewer",
            name="vision_debug_viewer",
            output="screen",
            parameters=[{
                "camera_topic": camera_topic,
                "visual_topic": visual_topic,
                "gesture_debug_topic": gesture_debug_topic,
                "vision_task_service": vision_task_service,
                "object_topic": object_topic,
                "object_only": False,
                "web_host": web_host,
                "web_port": web_port,
                "show_window": show_window,
                "publish_debug_image": publish_debug_image,
                "max_render_fps": max_render_fps,
                "render_scale": render_scale,
                "jpeg_quality": jpeg_quality,
                "object_poll_hz": 0.0,
                "event_history_limit": event_history_limit,
                "stereo_enabled": False,
                "web_enabled": True,
            }],
        ),
    ])
