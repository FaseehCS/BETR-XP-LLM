from typing import Optional
import grpc
from google.protobuf.internal import containers as _containers
import pipeline_pb2, pipeline_pb2_grpc
import numpy as np
import cv2 as cv

class MLDetector:
    def __init__(self, endpoint, api_key="test"):
        self.endpoint = endpoint
        self.api_key = api_key

    def check_connection(self):
        with grpc.insecure_channel(self.endpoint) as channel:
            stub = pipeline_pb2_grpc.ImageModelPipelineStub(channel)
            response = stub.Ping(pipeline_pb2.PingRequest(seq=1))
            if response.seq != 1:
                raise Exception("Cannot connect to the detection server")

    def detect(self, rgb_image, prompt="object")->Optional[np.ndarray]:
        """
        Do detection of rgb_image using a prompt

        Parameters:
            rgb_image: An OpenCV image in RGB color space.
            prompt: What to search for (may be None, then default value is used)

        Returns:
            Masks of identified objects or None if no object could be found
        """

        success, rgb_rawdata = cv.imencode(".jpg", rgb_image, [cv.IMWRITE_JPEG_QUALITY, 100])
        if not success:
            raise Exception("RGB image could not be encoded")

        self.predictions = self.detect_raw(prompt, pipeline_pb2.Image(image_format="jpg", image_data=bytes(rgb_rawdata)))

        if len(self.predictions.masks) == 0:
            return None
        
        masks = []
        boxes = []
        scores = []
        for i in range(len(self.predictions.masks)):
                mask = self.predictions.masks[i]
                box = self.predictions.regions[i]
                masks.append(np.unpackbits(np.frombuffer(mask.packedbits, dtype=np.uint8), count=mask.w*mask.h).reshape(mask.h, mask.w))
                boxes.append([box.x, box.y, box.w, box.h])
                scores.append(mask.score)

        return masks, boxes, scores, self.predictions.label

    def detect_pose(self, rgb_image, depth_image, intrinsics, prompt="object", box_threshold=0.3):
        """
        Do detection of rgb_image using a prompt

        Parameters:
            rgb_image: An OpenCV image in RGB color space.
            depth_image: numpy array of the depth image in meters
            intrinsics: Camera intrinsics in form 3x3 matrix
            prompt: What to search for (may be None, then default value is used)
            box_threshold: Detection threshold for bounding box confidence

        Returns:
            poses: list of 6D poses of each detected object in form [x, y, z, roll, pitch, yaw] in the camera frame
            masks: list of masks of each detected object
            boxes: list of bounding boxes of each detected object in form [x, y, w, h]
            scores: list of detection confidence scores of each detected object
            labels: list of string labels extracted from prompt corresponding to each detected object
        """

        if prompt is None or prompt == "":
            prompt = "object"

        success, rgb_rawdata = cv.imencode(".jpg", rgb_image, [cv.IMWRITE_JPEG_QUALITY, 100])
        if not success:
            raise Exception("RGB image could not be encoded")

        # We convert the depth image to mm and then to 16 bit
        depth_image = depth_image * 1000
        depth_image = depth_image.astype(np.uint16)
        success, depth_rawdata = cv.imencode(".png", depth_image, [cv.IMWRITE_PNG_COMPRESSION, 9])
        if not success:
            raise Exception("Depth image could not be encoded")

        self.predictions = self.detect_raw_pose(prompt, pipeline_pb2.Image(image_format="jpg", image_data=bytes(rgb_rawdata)), pipeline_pb2.Image(image_format="png", image_data=bytes(depth_rawdata)), intrinsics, box_threshold)

        if len(self.predictions.masks) == 0:
            return None
        
        poses = []
        masks = []
        boxes = []
        scores = []
        for i in range(len(self.predictions.masks)):
                pose = self.predictions.pose[i]
                mask = self.predictions.masks[i]
                box = self.predictions.regions[i]
                poses.append({"position": pose.position, "orientation": pose.orientation})
                masks.append(np.unpackbits(np.frombuffer(mask.packedbits, dtype=np.uint8), count=mask.w*mask.h).reshape(mask.h, mask.w))
                boxes.append([box.x, box.y, box.w, box.h])
                scores.append(mask.score)
                                        
        return poses, masks, boxes, scores, self.predictions.label

    def detect_raw(self, prompt: str, img: pipeline_pb2.Image)->pipeline_pb2.ObjectDetectionReply:
        """
        Send image and prompt to image pipeline, internal method

        Parameters:
            prompt: what to search for
            img: encoded image for the service

        Returns:
            Service reply
        """
        with grpc.insecure_channel(self.endpoint) as channel:
            stub = pipeline_pb2_grpc.ImageModelPipelineStub(channel)
            return stub.PromptObjectDetection(pipeline_pb2.PromptObjectDetectionRequest(api_key=self.api_key, prompt=prompt, image=img))
        
    def detect_raw_pose(self, prompt: str, rgb: pipeline_pb2.Image, depth: pipeline_pb2.Image, intrinsics: _containers.RepeatedScalarFieldContainer[float], box_threshold: float)->pipeline_pb2.PoseDetectionReply:
        """
        Send image and prompt to image pipeline, internal method

        Parameters:
            prompt: what to search for
            img: encoded image for the service

        Returns:
            Service reply
        """
        with grpc.insecure_channel(self.endpoint) as channel:
            stub = pipeline_pb2_grpc.ImageModelPipelineStub(channel)
            return stub.PoseDetection(pipeline_pb2.PoseDetectionRequest(api_key=self.api_key, prompt=prompt, rgb=rgb, depth=depth, intrinsics=intrinsics, box_threshold=box_threshold))
    
    def xyxy(self):
        """ Return bounding boxes for detected objects in x1, y1, x2, y2 format """
        return [[box.x, box.y, box.x + box.w, box.y + box.h] for box in self.predictions.regions]