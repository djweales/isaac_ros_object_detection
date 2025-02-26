#!/usr/bin/env python3

# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

# This script listens for images and object detections on the image,
# then renders the output boxes on top of the image and publishes
# the result as an image message

import cv2
import numpy as np
import cv_bridge
import message_filters
import rclpy
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup
from rclpy.node import Node
from sensor_msgs.msg import Image, CompressedImage
from vision_msgs.msg import Detection2DArray

names = {
        0: 'person',
}


class Yolo11SegVisualizer(Node):
    QUEUE_SIZE = 10
    color = (0, 255, 0)
    bbox_thickness = 2

    def __init__(self):
        super().__init__('yolo11_seg_visualizer')
        # Reads the parameters from the launch file for camera name and camera type
        self.camera_name = self.declare_parameter("camera_name", "").value
        self.camera_type = self.declare_parameter("camera_type", "").value
        self.frame_n = 0

        self._bridge = cv_bridge.CvBridge()
        self._processed_image_pub = self.create_publisher(
            CompressedImage, f'/{self.camera_name}/yolo11_seg_processed_image',  self.QUEUE_SIZE)
        self._depth_image_pub = self.create_publisher(
            CompressedImage, f'/{self.camera_name}/depth_visualizer',  self.QUEUE_SIZE)

        self._detections_subscription = message_filters.Subscriber(
            self,
            Detection2DArray,
            'detections_output')
        self._detections_mask_subscription = message_filters.Subscriber(
            self,
            Image,
            'detections_mask')
        self._image_subscription = message_filters.Subscriber(
            self,
            Image,
            f'{self.camera_name}/image')
        self._depth_subscription = message_filters.Subscriber(
            self,
            Image,
            f'{self.camera_name}/depth')


        if self.camera_type == "camera":
            self.time_synchronizer = message_filters.TimeSynchronizer(
                [self._image_subscription, self._depth_subscription],
                self.QUEUE_SIZE)
            self.time_synchronizer.registerCallback(self.camera_callback)
        elif self.camera_type == "detection":
            self.time_synchronizer = message_filters.TimeSynchronizer(
                [self._detections_subscription, self._image_subscription, self._detections_mask_subscription, self._depth_subscription],
                self.QUEUE_SIZE)

            self.time_synchronizer.registerCallback(self.detections_callback)

    def camera_callback(self, img_msg, depth_msg):
        cv2_img = self._bridge.imgmsg_to_cv2(img_msg, desired_encoding="bgr8")
        cv2_depth = self._bridge.imgmsg_to_cv2(depth_msg, desired_encoding="passthrough")

        self.frame_n += 1

        if self.frame_n % 10 == 0:
            cv2_img = cv2.resize(cv2_img, (0,0), fx=0.25, fy=0.25)
            processed_img = self._bridge.cv2_to_compressed_imgmsg(
                cv2_img)
            self._processed_image_pub.publish(processed_img)
            
            cv2_depth = cv2.resize(cv2_depth, (0,0), fx=0.25, fy=0.25)
            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(cv2_depth, alpha=0.1), cv2.COLORMAP_HSV)
            depth_colormap_msg = self._bridge.cv2_to_compressed_imgmsg(
                depth_colormap) 
            self._depth_image_pub.publish(depth_colormap_msg)

    def detections_callback(self, detections_msg, img_msg, mask_msg, depth_msg):
        def model_to_image_pos(model_point, image_shape, model_shape):
            model_px, model_py = model_point
            image_x, image_y = image_shape
            model_x, model_y = model_shape
                        
            rel_image_y = image_y / image_x
            rel_model_y = model_y / model_x
            rel_model_point_y = model_py / model_y
            rel_model_point_x = model_px / model_x

            image_py = round(image_y * ((rel_model_point_y - ((rel_model_y - rel_image_y) / 2)) / rel_image_y))
            image_px = round(image_x * rel_model_point_x)
 
            return (image_px, image_py)

        
        txt_color = (255, 0, 255)
        cv2_img = self._bridge.imgmsg_to_cv2(img_msg, desired_encoding="bgr8")
        cv2_mask = self._bridge.imgmsg_to_cv2(mask_msg)
        cv2_depth = self._bridge.imgmsg_to_cv2(depth_msg, desired_encoding="passthrough")
        image_shape = [cv2_img.shape[1], cv2_img.shape[0]]
        model_shape = [cv2_mask.shape[1], cv2_mask.shape[0]]

        upscaled_mask = cv2.resize(cv2_mask, (cv2_img.shape[1], cv2_img.shape[1]), interpolation=cv2.INTER_LINEAR)
        height_start = round((cv2_img.shape[1] - cv2_img.shape[0]) / 2)
        height_end = cv2_img.shape[1] - height_start
        upscaled_mask = upscaled_mask[height_start:height_end, 0:cv2_img.shape[1]]
        red_color = (0, 0, 255)
        opacity = 0.6
        red_mask = np.zeros_like(cv2_img)
        red_mask[upscaled_mask > 0] = red_color
        cv2_img = cv2.addWeighted(cv2_img, 1.0, red_mask, opacity, 0.0)

        for detection in detections_msg.detections:
            center_x = detection.bbox.center.position.x
            center_y = detection.bbox.center.position.y
            width = detection.bbox.size_x
            height = detection.bbox.size_y

            label = names[int(detection.results[0].hypothesis.class_id)]
            conf_score = detection.results[0].hypothesis.score
            label = f'{label} {conf_score:.2f}'

            min_pt = (round(center_x - (width / 2.0)),
                      round(center_y - (height / 2.0)))
            max_pt = (round(center_x + (width / 2.0)),
                      round(center_y + (height / 2.0)))
            
            min_pt = model_to_image_pos(min_pt, image_shape, (cv2_img.shape[1], cv2_img.shape[1]))
            max_pt = model_to_image_pos(max_pt, image_shape, (cv2_img.shape[1], cv2_img.shape[1]))
            
            lw = max(round((img_msg.height + img_msg.width) / 2 * 0.003), 2)  # line width
            tf = max(lw - 1, 1)  # font thickness
            # text width, height
            w, h = cv2.getTextSize(label, 0, fontScale=lw / 3, thickness=tf)[0]
            outside = min_pt[1] - h >= 3

            cv2.rectangle(cv2_img, min_pt, max_pt,
                          self.color, self.bbox_thickness)
            cv2.putText(cv2_img, label, (min_pt[0], min_pt[1]-2 if outside else min_pt[1]+h+2),
                        0, lw / 3, txt_color, thickness=tf, lineType=cv2.LINE_AA)
            
            lowest_point_x = round(detection.results[0].pose.pose.position.x * cv2_mask.shape[1])
            lowest_point_y = round(detection.results[0].pose.pose.position.y * cv2_mask.shape[0])
            lowest_point = model_to_image_pos((lowest_point_x, lowest_point_y), image_shape, model_shape)
            cv2.circle(cv2_img, lowest_point, radius=10, color=(255, 0, 0), thickness=-1)

        self.frame_n += 1

        if self.frame_n % 10 == 0:
            cv2_img = cv2.resize(cv2_img, (0,0), fx=0.25, fy=0.25)
            processed_img = self._bridge.cv2_to_compressed_imgmsg(
                cv2_img)
            # processed_img = self._bridge.cv2_to_imgmsg(
            #     cv2_img, encoding=img_msg.encoding)
            self._processed_image_pub.publish(processed_img)
            
            cv2_depth = cv2.resize(cv2_depth, (0,0), fx=0.25, fy=0.25)
            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(cv2_depth, alpha=0.1), cv2.COLORMAP_HSV)
            depth_colormap_msg = self._bridge.cv2_to_compressed_imgmsg(
                depth_colormap) 
            self._depth_image_pub.publish(depth_colormap_msg)

def main():
    rclpy.init()
    node = Yolo11SegVisualizer()
    executor = MultiThreadedExecutor()
    executor.add_node(node)

    try:
        node.get_logger().info('Beginning client, shut down with CTRL-C')
        executor.spin()
    except KeyboardInterrupt:
        node.get_logger().info('Keyboard interrupt, shutting down.\n')
    node.destroy_node()
    rclpy.shutdown()

    # rclpy.init()
    # rclpy.spin(Yolo11SegVisualizer())
    # rclpy.shutdown()


if __name__ == '__main__':
    main()