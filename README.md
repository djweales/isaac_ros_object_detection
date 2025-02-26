# Isaac ROS Object Detection

NVIDIA-accelerated, deep learned model support for object detection including DetectNet.

to run

cd ${ISAAC_ROS_WS}/src/isaac_ros_common && \
./scripts/run_dev.sh

sudo apt-get update
 
rosdep update && rosdep install --from-paths ${ISAAC_ROS_WS}/src/isaac_ros_object_detection/isaac_ros_yolo11_seg --ignore-src -y

cd ${ISAAC_ROS_WS} && \
   colcon build --symlink-install --packages-up-to isaac_ros_yolo11_seg --base-paths ${ISAAC_ROS_WS}/src/isaac_ros_object_detection/isaac_ros_yolo11_seg

source install/setup.bash

ros2 launch isaac_ros_yolo11_seg yolo11_seg_tensor_rt.launch.py model_file_path:=./isaac_ros_assets/models/yolo11/nursery8.onnx engine_file_path:=./isaac_ros_assets/models/yolo11/nursery8.plan network_image_width:=1280 network_image_height:=1280 confidence_threshold:=0.50 num_classes:=1 input_image_width:=1280 input_image_height:=1280

ros2 launch isaac_ros_yolo11_seg yolo11_seg_tensor_rt.launch.py model_file_path:=./isaac_ros_assets/models/yolo11/nursery2.onnx engine_file_path:=./isaac_ros_assets/models/yolo11/nursery2.plan network_image_width:=640 network_image_height:=640 confidence_threshold:=0.50 num_classes:=1 input_image_width:=640 input_image_height:=640

ros2 launch isaac_ros_yolo11_seg yolo11_seg_tensor_rt.launch.py model_file_path:=./isaac_ros_assets/models/yolo11/nursery11_1080.onnx engine_file_path:=./isaac_ros_assets/models/yolo11/nursery11_1080.plan network_image_width:=1088 network_image_height:=1088 confidence_threshold:=0.50 num_classes:=1 input_image_width:=1088 input_image_height:=1088
