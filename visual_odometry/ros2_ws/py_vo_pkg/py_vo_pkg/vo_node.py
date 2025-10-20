#!/usr/bin/env python3
import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Quaternion 

from cv_bridge import CvBridge
import cv2
import numpy as np
import rerun as rr
from scipy.spatial.transform import Rotation as R_scipy

def ros_quat_to_rerun_rot(q: Quaternion):
    return rr.Quaternion(xyzw=[q.x, q.y, q.z, q.w])

def odom_to_pose_matrix(odom_msg: Odometry):
    """Converts a ROS Odometry message (position and orientation) into a 4x4 homogeneous matrix."""
    
    # Get position & orientation
    pos = odom_msg.pose.pose.position
    t = np.array([pos.x, pos.y, pos.z])
    quat = odom_msg.pose.pose.orientation
    r = R_scipy.from_quat([quat.x, quat.y, quat.z, quat.w])
    R_mat = r.as_matrix()
    
    # Construct 4x4 matrix
    T = np.eye(4)
    T[:3, :3] = R_mat
    T[:3, 3] = t
    
    return T


class VisualOdometryNode(Node):
    def __init__(self):
        super().__init__('visual_odometry_node')
        
        # Create subscribers
        self.image_sub = self.create_subscription(
            Image,
            "/airsim_node/drone_1/front_center_custom/Scene",
            self.image_callback,
            10)
        
        self.odom_sub = self.create_subscription(
            Odometry,
            "/airsim_node/drone_1/odom_local_ned",
            self.odom_callback,
            1) 

        rr.init("airsim_vo_logging", spawn=True)
        self.bridge = CvBridge()
        
        # Initialize VO variables
        self.prev_image = None
        self.prev_kp = None
        self.prev_des = None
        self.gt_trajectory_points = []
        self.est_trajectory_points = []
        self.initial_pose_set = False

        # Initialize OpenCV components
        self.orb = cv2.ORB_create()
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        self.pose = np.eye(4)

        self.ned_to_viz_transform = np.array([
            [ 0,  1,  0,  0],  # X_viz = Y_ned (East)
            [ 1,  0,  0,  0],  # Y_viz = X_ned (North)
            [ 0,  0, -1,  0],  # Z_viz = -Z_ned (Up)
            [ 0,  0,  0,  1]
        ])


        # Use the node's logger
        self.get_logger().info("Visual Odometry node initialized. Waiting for initial Ground Truth pose...")

    def image_callback(self, msg):
        
        if not self.initial_pose_set:
            self.get_logger().info("Waiting for initial pose from Odometry...")
            return

        # Convert ROS image to OpenCV
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rr.log("airsim/camera/image", rr.Image(frame_rgb))

        # Detect features
        kp, des = self.orb.detectAndCompute(gray, None)

        if self.prev_image is not None and self.prev_des is not None:
            try:
                matches = self.bf.knnMatch(self.prev_des, des, k=2)
                
                # Apply Lowe's ratio test to filter poor matches
                good_matches = []
                ratio_thresh = 0.75
                for m, n in matches:
                    if m.distance < ratio_thresh * n.distance:
                        good_matches.append(m)

                # Extract keypoints for essential matrix calculation
                points1 = []
                points2 = []
                for match in good_matches:
                    pt1 = keypoints1[match.queryIdx].pt
                    pt2 = keypoints2[match.trainIdx].pt
                    points1.append(pt1)
                    points2.append(pt2)
                points1 = np.float32(points1).reshape(-1, 1, 2)
                points2 = np.float32(points2).reshape(-1, 1, 2)
        
                focal_length = 268.5 # f = W / (2*tan(fov/2))
                principal_point = (320.0, 240.0) 

                E, mask = cv.findEssentialMat(points1, points2, K, cv.RANSAC, threshold=threshold)
                
                if E is not None:
                    _, R, t, mask = cv2.recoverPose(E, points1, points2, 
                                                    focal=focal_length, 
                                                    pp=principal_point)

                    T_rel = np.eye(4)
                    T_rel[:3, :3] = R.T  
                    T_rel[:3, 3] = (-R.T @ t)[:, 0] 

                    # Update global pose
                    # NOTE: This assumes scale = 1, common limitation of monocular VO
                    self.pose = self.pose @ T_rel
                    trans = self.pose[:3, 3]
                    
                    # Transform pose for Rerun visualization 
                    viz_pose = self.ned_to_viz_transform @ self.pose 
                    viz_trans = viz_pose[:3, 3]


                    self.get_logger().info(f"Estimated position: x={viz_trans[0]:.2f}, y={viz_trans[1]:.2f}, z={viz_trans[2]:.2f}")
                    estimate_pos = viz_trans.tolist()
                    self.est_trajectory_points.append(estimate_pos)

                    # Log to Rerun
                    rr.log(
                        "airsim/trajectory/estimate/trajectory_trace", 
                        rr.LineStrips3D(
                            [np.array(self.est_trajectory_points)], 
                            colors=[255, 0, 0] # Red for Estimate
                        )
                    )

                    rr.log(
                        "airsim/trajectory/estimate/points",
                        rr.Points3D(estimate_pos, class_ids=1, colors=[255, 0, 0])
                    )


            except cv2.error as e:
                self.get_logger().warn(f"OpenCV Error: {e}")
            except Exception as e:
                self.get_logger().error(f"Error in VO calculation: {e}")


        self.prev_image = gray
        self.prev_kp = kp
        self.prev_des = des

    def odom_callback(self, msg):
        
        # 1. Set the initial VO pose using the first ground truth  message
        if not self.initial_pose_set:
            initial_gt_pose = odom_to_pose_matrix(msg)
            self.pose = initial_gt_pose
            self.initial_pose_set = True
            self.get_logger().info("Initial estimated pose set from Ground Truth.")
        
        
        # 2. Log Ground Truth for comparison
        pos = msg.pose.pose.position
        # Convert NED position to visualization frame
        current_pos_viz = np.array([pos.y, pos.x, -pos.z]).tolist() 

        self.gt_trajectory_points.append(current_pos_viz)
        rr.log(
            "airsim/trajectory/ground_truth/trajectory_trace", 
            rr.LineStrips3D(
                [np.array(self.gt_trajectory_points)], 
                colors=[0, 255, 0] # Green for Ground Truth
            )
        )

        rr.log(
            "airsim/trajectory/ground_truth/points",
            rr.Points3D(current_pos_viz, class_ids=1, colors=[0, 255, 0])
        )

        self.get_logger().info(f"Ground Truth (Viz Frame): x={current_pos_viz[0]:.2f}, y={current_pos_viz[1]:.2f}, z={current_pos_viz[2]:.2f}")


def main(args=None):
    rclpy.init(args=args)
    
    try:
        vo_node = VisualOdometryNode()
        rclpy.spin(vo_node)
        
    except KeyboardInterrupt:
        pass
    finally:
        vo_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()