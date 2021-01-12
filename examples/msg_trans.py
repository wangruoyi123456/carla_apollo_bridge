#!/usr/bin/env python

from cyber_py import cyber, cyber_time
'''apollo msg:
/apollo/canbus/chassis                                  modules.canbus.proto.chassis_pb2
/apollo/perception/obstacles                      modules.perception.proto.perception_obstacle_pb2
/apollo/planning                                                modules.planning.proto.planning_pb2
/apollo/sensor/lidar128/compensator/PointCloud2
/apollo/control
/apollo/localization/pose
'''

'''read msg'''
from modules.control.proto.control_cmd_pb2 import ControlCommand
from modules.canbus.proto.chassis_pb2 import Chassis
from modules.planning.proto.planning_pb2 import ADCTrajectory
from modules.routing.proto.routing_pb2 import RoutingResponse
from modules.drivers.proto.sensor_image_pb2 import CompressedImage
from modules.drivers.proto.pointcloud_pb2 import PointXYZIT, PointCloud
from modules.perception.proto.perception_obstacle_pb2 import PerceptionObstacle, PerceptionObstacles
from modules.localization.proto.localization_pb2 import LocalizationEstimate
from modules.transform.proto.transform_pb2 import TransformStamped, TransformStampeds

'''neolix msg
/neolix/canbus/chassis                                  from global_adc_status_pb2 
/neolix/perception/obstacles                      perception_obstacle_pb2
/neolix/control
'''

'''write msg'''
from global_adc_status_pb2 import Chassis as NeolixChassis
from global_adc_status_pb2 import ErrorCode as NeolixErrorCode
from global_adc_status_pb2 import GlobalState as NeolixGlobalState
from global_adc_status_pb2 import State as NeolixState
from global_adc_status_pb2 import ResultCode as NeolixResultCode
from global_adc_status_pb2 import StopReason as NeolixStopReason
from global_adc_status_pb2 import SysFaultLevel as NeolixSysFaultLevel
from control_cmd_pb2 import ControlCommand as  NeoControlCommand
from perception_obstacle_pb2 import  PerceptionObstacles as NeolixPerceptionObstacles
from perception_obstacle_pb2 import  PerceptionObstacle as NeolixPerceptionObstacle
from perception_obstacle_pb2 import  Point as NeolixPoint

from sensor_image_pb2 import CompressedImage  as NeolixCompressedImage
from sensor_pointcloud_pb2 import PointCloud as NeolixPointCloud
from sensor_pointcloud_pb2 import PointXYZIT as NeolixPointXYZIT
from carla_msg_pb2 import CarlaMsg
from localization_pose_pb2 import LocalizationEstimate as NeolixLocalizationEstimate

import math
import numpy as np
import types
import time

from pyproj import Proj, transform

def caculate_lon_lat(pose_x, pose_y):
    # proj = Proj(proj='tmerc', lon_0=-122.01625392642, lat_0=37.4153680065658, preserve_units=False)
    proj = Proj(proj='tmerc', lon_0=116.49294334757724, lat_0=39.97830155553286, preserve_units=False)
    lon_and_lat = proj(pose_x, pose_y, inverse = True)
    return lon_and_lat

def get_pose_utm(lon, lat):
    wgs84 = Proj(init='EPSG:4326')
    # utm = Proj(init="EPSG:32610")
    utm = Proj(init="EPSG:32650")
    pose = transform(wgs84, utm, lon, lat, radians=False)
    return pose
def euler_from_matrix(matrix, axes='sxyz'):
    """Return Euler angles from rotation matrix for specified axis sequence.

    axes : One of 24 axis sequences as string or encoded tuple

    Note that many Euler angle triplets can describe one matrix.

    # >>> R0 = euler_matrix(1, 2, 3, 'syxz')
    # >>> al, be, ga = euler_from_matrix(R0, 'syxz')
    # >>> R1 = euler_matrix(al, be, ga, 'syxz')
    # >>> numpy.allclose(R0, R1)
    # True
    # >>> angles = (4.0*math.pi) * (numpy.random.random(3) - 0.5)
    # >>> for axes in _AXES2TUPLE.keys():
    # ...    R0 = euler_matrix(axes=axes, *angles)
    # ...    R1 = euler_matrix(axes=axes, *euler_from_matrix(R0, axes))
    # ...    if not numpy.allclose(R0, R1): print axes, "failed"

    """
    try:
        firstaxis, parity, repetition, frame = _AXES2TUPLE[axes.lower()]
    except (AttributeError, KeyError):
        _ = _TUPLE2AXES[axes]
        firstaxis, parity, repetition, frame = axes

    i = firstaxis
    j = _NEXT_AXIS[i+parity]
    k = _NEXT_AXIS[i-parity+1]

    M = numpy.array(matrix, dtype=numpy.float64, copy=False)[:3, :3]
    if repetition:
        sy = math.sqrt(M[i, j]*M[i, j] + M[i, k]*M[i, k])
        if sy > _EPS:
            ax = math.atan2( M[i, j],  M[i, k])
            ay = math.atan2( sy,       M[i, i])
            az = math.atan2( M[j, i], -M[k, i])
        else:
            ax = math.atan2(-M[j, k],  M[j, j])
            ay = math.atan2( sy,       M[i, i])
            az = 0.0
    else:
        cy = math.sqrt(M[i, i]*M[i, i] + M[j, i]*M[j, i])
        if cy > _EPS:
            ax = math.atan2( M[k, j],  M[k, k])
            ay = math.atan2(-M[k, i],  cy)
            az = math.atan2( M[j, i],  M[i, i])
        else:
            ax = math.atan2(-M[j, k],  M[j, j])
            ay = math.atan2(-M[k, i],  cy)
            az = 0.0

    if parity:
        ax, ay, az = -ax, -ay, -az
    if frame:
        ax, az = az, ax
    return ax, ay, az
def euler_from_quaternion(quaternion, axes='sxyz'):
    """Return Euler angles from quaternion for specified axis sequence.

    >>> angles = euler_from_quaternion([0.06146124, 0, 0, 0.99810947])
    >>> numpy.allclose(angles, [0.123, 0, 0])
    True

    """
    return euler_from_matrix(quaternion_matrix(quaternion), axes)

# def quaternion_from_euler(ai, aj, ak, axes='sxyz'):
def quaternion_from_euler(ai, aj, ak, axes='sxyz'):
    """Return quaternion from Euler angles and axis sequence.
    ai, aj, ak : Euler's roll, pitch and yaw angles
    axes : One of 24 axis sequences as string or encoded tuple
    # >>> q = quaternion_from_euler(1, 2, 3, 'ryxz')
    # >>> numpy.allclose(q, [0.310622, -0.718287, 0.444435, 0.435953])
    True
    """
    try:
        firstaxis, parity, repetition, frame = _AXES2TUPLE[axes.lower()]
    except (AttributeError, KeyError):
        _ = _TUPLE2AXES[axes]
        firstaxis, parity, repetition, frame = axes

    i = firstaxis
    j = _NEXT_AXIS[i+parity]
    k = _NEXT_AXIS[i-parity+1]

    if frame:
        ai, ak = ak, ai
    if parity:
        aj = -aj

    ai /= 2.0
    aj /= 2.0
    ak /= 2.0
    ci = math.cos(ai)
    si = math.sin(ai)
    cj = math.cos(aj)
    sj = math.sin(aj)
    ck = math.cos(ak)
    sk = math.sin(ak)
    cc = ci*ck
    cs = ci*sk
    sc = si*ck
    ss = si*sk

    quaternion = np.empty((4, ), dtype=np.float64)
    if repetition:
        quaternion[i] = cj*(cs + sc)
        quaternion[j] = sj*(cc + ss)
        quaternion[k] = sj*(cs - sc)
        quaternion[3] = cj*(cc - ss)
    else:
        quaternion[i] = cj*sc - sj*cs
        quaternion[j] = cj*ss + sj*cc
        quaternion[k] = cj*cs - sj*sc
        quaternion[3] = cj*cc + sj*ss
    if parity:
        quaternion[j] *= -1

    return quaternion

# axis sequences for Euler angles
_NEXT_AXIS = [1, 2, 0, 1]

# map axes strings to/from tuples of inner axis, parity, repetition, frame
_AXES2TUPLE = {
    'sxyz': (0, 0, 0, 0), 'sxyx': (0, 0, 1, 0), 'sxzy': (0, 1, 0, 0),
    'sxzx': (0, 1, 1, 0), 'syzx': (1, 0, 0, 0), 'syzy': (1, 0, 1, 0),
    'syxz': (1, 1, 0, 0), 'syxy': (1, 1, 1, 0), 'szxy': (2, 0, 0, 0),
    'szxz': (2, 0, 1, 0), 'szyx': (2, 1, 0, 0), 'szyz': (2, 1, 1, 0),
    'rzyx': (0, 0, 0, 1), 'rxyx': (0, 0, 1, 1), 'ryzx': (0, 1, 0, 1),
    'rxzx': (0, 1, 1, 1), 'rxzy': (1, 0, 0, 1), 'ryzy': (1, 0, 1, 1),
    'rzxy': (1, 1, 0, 1), 'ryxy': (1, 1, 1, 1), 'ryxz': (2, 0, 0, 1),
    'rzxz': (2, 0, 1, 1), 'rxyz': (2, 1, 0, 1), 'rzyz': (2, 1, 1, 1)}

_TUPLE2AXES = dict((v, k) for k, v in _AXES2TUPLE.items())


class MsgTransform():
    def __init__(self):
        cyber.init()
        self.msg_trans_node = cyber.Node("msg_transform")
        self.sequence_num = 0
        self.sequence_num_pose = 0

        # # self.neolix_control_cmd = self.msg_trans_node.create_writer("/planning/control", neocc, 1)
        # apollo_control_cmd = self.msg_trans_node.create_reader("/apollo/control", ControlCommand, self.controllcmd_callback)
        # self.msg_trans_node.create_reader('/apollo/canbus/chassis', Chassis, self.chassis_callback)
        # self.msg_trans_node.create_reader('/apollo/perception/obstacles', PerceptionObstacles, self.perceptionobstacle_callback)
        # self.msg_trans_node.create_reader('/apollo/planning', ADCTrajectory, self.planning_callback)

        self.chassis_writer = self.msg_trans_node.create_writer('/planning/proxy/DuDriveChassis', NeolixChassis, 1)
        # self.chassis_writer = self.msg_trans_node.create_writer('/DuDriveChassis', NeolixChassis, 1)
        self.obstacle_writer = self.msg_trans_node.create_writer('/perception/obstacles', NeolixPerceptionObstacles, 1)
        # self.lidar_pointcloud_writer = self.msg_trans_node.create_writer('/sensor/velodyne16/all/tfcompensator/PointCloud2', NeolixPointCloud, 1)
        self.carla_msg_write = self.msg_trans_node.create_writer('/carla/msg', CarlaMsg, 1)
        self.carla_pose_write = self.msg_trans_node.create_writer('/localization/100hz/localization_pose', NeolixLocalizationEstimate, 1)
        # self.static_tf_write = self.msg_trans_node.create_writer('/static_tf', )
        # self.carla_global_state = self.msg_trans_node.create_writer('/pnc/global_state', NeolixGlobalState, 1)


        while not cyber.is_shutdown():

            time.sleep(0.01)
            self.msg_trans_node.create_reader('/apollo/canbus/chassis', Chassis, self.chassis_callback)
            self.msg_trans_node.create_reader('/apollo/perception/obstacles', PerceptionObstacles, self.obstacle_callback)
            # self.msg_trans_node.create_reader('/apollo/sensor/lidar128/compensator/PointCloud2', PointCloud, self.lidar_callback)
            # self.msg_trans_node.create_reader('/neolix/control', NeoControlCommand, self.cmd_callback)
            self.msg_trans_node.create_reader('/pnc/control', NeoControlCommand, self.cmd_callback)
            self.msg_trans_node.create_reader('/apollo/localization/pose', LocalizationEstimate, self.pose_callback)
            # self.msg_trans_node.create_reader('/tf',)
            # global_state = NeolixGlobalState()
            # global_state.header.timestamp_sec = cyber_time.Time.now().to_sec()
            # global_state.state = NeolixState.CRUISE
            # global_state.context.finish_context.last_state = NeolixState.INIT
            # global_state.context.finish_context.result_code = NeolixResultCode.RESULT_SUCCESS
            # global_state.stop_reason = NeolixStopReason.NOT_STOP
            # self.carla_global_state.write(global_state)

            self.msg_trans_node.spin()
       
    def chassis_callback(self, chassis_msg):
        neolix_chassis = NeolixChassis()
        neolix_chassis.engine_started = chassis_msg.engine_started
        neolix_chassis.speed_mps = chassis_msg.speed_mps
        neolix_chassis.engine_rpm = 380.00  #carla: 5800
        neolix_chassis.odometer_m = 0.0
        neolix_chassis.throttle_percentage = chassis_msg.throttle_percentage
        neolix_chassis.brake_percentage = chassis_msg.brake_percentage
        neolix_chassis.steering_percentage = chassis_msg.steering_percentage
        neolix_chassis.parking_brake = chassis_msg.parking_brake
        neolix_chassis.header.timestamp_sec = chassis_msg.header.timestamp_sec
        neolix_chassis.header.module_name = "chassis"
        neolix_chassis.header.sequence_num = self.sequence_num
        neolix_chassis.driving_mode = chassis_msg.driving_mode
        neolix_chassis.gear_location = chassis_msg.gear_location
        neolix_chassis.error_code = NeolixErrorCode.NO_ERROR
        neolix_chassis.safety_msg.vehicle_whole_fault_level = 4
        neolix_chassis.safety_msg.drive_system_fault_level = 0
        neolix_chassis.safety_msg.high_voltage_battery_fault_level=0
        neolix_chassis.safety_msg.steering_system_fault_level=0
        neolix_chassis.safety_msg.brake_system_fault_level = 0
        neolix_chassis.safety_msg.aeb_trigger_state = 0
        neolix_chassis.extend_info.kinglong.dump_energy=78.00
        neolix_chassis.extend_info.kinglong.charge_state=1

        neolix_chassis.sysfaultlevel = NeolixSysFaultLevel.NO_FAULT
        self.chassis_writer.write(neolix_chassis)
        self.sequence_num +=1
    def obstacle_callback(self, obstacle_msg):
        # print("obstacle_msg:", obstacle_msg)
        neolix_obstacle = NeolixPerceptionObstacles()
        neolix_obstacle.header.timestamp_sec = obstacle_msg.header.timestamp_sec
        neolix_obstacle.header.frame_id = obstacle_msg.header.frame_id
        length = len(obstacle_msg.perception_obstacle)
        if length:
            for i in range(length):
                # print("type(ob): ", type(ob))
                temp_apollo_obstacle = PerceptionObstacle()
                temp_apollo_obstacle = obstacle_msg.perception_obstacle[i]
                obstacle = NeolixPerceptionObstacle()
                obstacle.id = temp_apollo_obstacle.id

                get_obstacle_lon_lat = caculate_lon_lat(temp_apollo_obstacle.position.x, temp_apollo_obstacle.position.y)
                utm_pose = get_pose_utm(get_obstacle_lon_lat[0], get_obstacle_lon_lat[1])
                # print('utm_pose:', utm_pose)
                obstacle.position.x = utm_pose[0]
                obstacle.position.y = utm_pose[1]
                obstacle.position.z = temp_apollo_obstacle.position.z

                # q = quaternion_from_euler(
                #     -math.radians(transform.rotation.roll),
                #     math.radians(transform.rotation.pitch),
                #     (-math.radians(transform.rotation.yaw) + math.radians(-90))
                # )
                # roll, pitch, yaw = euler_from_quaternion(q)

                obstacle.theta = temp_apollo_obstacle.theta
                # obstacle.theta = yaw
                obstacle.velocity.x = temp_apollo_obstacle.velocity.x
                obstacle.velocity.y = temp_apollo_obstacle.velocity.y
                obstacle.velocity.z = temp_apollo_obstacle.velocity.z
                obstacle.length = temp_apollo_obstacle.length
                obstacle.width = temp_apollo_obstacle.width
                obstacle.height = temp_apollo_obstacle.height
                obstacle.type = temp_apollo_obstacle.type
                # obstacle.type = NeolixPerceptionObstacle.Type.UNKNOWN
                obstacle.timestamp = cyber_time.Time.now().to_sec()

                delta_x = temp_apollo_obstacle.length*math.sin(temp_apollo_obstacle.theta)/2.0
                delta_y = temp_apollo_obstacle.length*math.cos(temp_apollo_obstacle.theta)/2.0
                delta_z = temp_apollo_obstacle.position.z/ 2.0
                delta_x_1 = temp_apollo_obstacle.width* math.cos(temp_apollo_obstacle.theta)/2.0
                delta_y_1 = temp_apollo_obstacle.width * math.sin(temp_apollo_obstacle.theta) / 2.0
                # delta_x = temp_apollo_obstacle.length * math.cos(yaw) / 2.0
                # delta_y = temp_apollo_obstacle.length * math.sin(yaw) / 2.0
                # delta_z = temp_apollo_obstacle.position.z / 2.0
                # delta_x_1 = temp_apollo_obstacle.width * math.sin(yaw) / 2.0
                # delta_y_1 = temp_apollo_obstacle.width * math.sin(yaw) / 2.0


                '''A--|--B
                    |    |
                    |    |
                   C-----D
                   A->C->D->B'''

                for i in range(2):
                    point_1_i = NeolixPoint()
                    point_1_i.x = utm_pose[0] + delta_x - delta_x_1
                    point_1_i.y = utm_pose[1] + delta_y + delta_y_1
                    point_1_i.z = temp_apollo_obstacle.position.z + math.pow(-1, i) * delta_z
                    obstacle.polygon_point.append(point_1_i)

                    point_3_i = NeolixPoint()
                    point_3_i.x = utm_pose[0] - delta_x - delta_x_1
                    point_3_i.y = utm_pose[1] - delta_y + delta_y_1
                    point_3_i.z = temp_apollo_obstacle.position.z + math.pow(-1, i) * delta_z
                    obstacle.polygon_point.append(point_3_i)

                    point_4_i = NeolixPoint()
                    point_4_i.x = utm_pose[0] - delta_x + delta_x_1
                    point_4_i.y = utm_pose[1] - delta_y - delta_y_1
                    point_4_i.z = temp_apollo_obstacle.position.z + math.pow(-1, i) * delta_z
                    obstacle.polygon_point.append(point_4_i)

                    point_2_i = NeolixPoint()
                    point_2_i.x = utm_pose[0] + delta_x + delta_x_1
                    point_2_i.y = utm_pose[1] + delta_y - delta_y_1
                    point_2_i.z = temp_apollo_obstacle.position.z + math.pow(-1, i) * delta_z
                    obstacle.polygon_point.append(point_2_i)


                obstacle.anchor_point.x = utm_pose[0]
                obstacle.anchor_point.y = utm_pose[1]
                obstacle.anchor_point.z = temp_apollo_obstacle.position.z+0.05
                obstacle.bbox2d.xmin = 0.0
                obstacle.bbox2d.ymin = 0.0
                obstacle.bbox2d.xmax = 0.0
                obstacle.bbox2d.ymax = 0.0


                neolix_obstacle.perception_obstacle.append(obstacle)
        self.obstacle_writer.write(neolix_obstacle)
    # def lidar_callback(self, lidar_msg):
    #     neolix_pointcloud = NeolixPointCloud()
    #     neolix_pointcloud.header.timestamp_sec = lidar_msg.header.timestamp_sec
    #     neolix_pointcloud.header.frame_id = lidar_msg.header.frame_id
    #     neolix_pointcloud.frame_id = lidar_msg.frame_id
    #     neolix_pointcloud.is_dense = lidar_msg.is_dense
    #     for point in lidar_msg.point:
    #         neolix_point = NeolixPointXYZIT()
    #         neolix_point.x = point.x
    #         neolix_point.y = point.y
    #         neolix_point.z = point.z
    #         neolix_point.intensity = point.intensity
    #         neolix_point.stamp = point.timestamp
    #         neolix_pointcloud.point.append(neolix_point)
    #     neolix_pointcloud.measurement_time = lidar_msg.measurement_time
    #     neolix_pointcloud.width = lidar_msg.width
    #     neolix_pointcloud.height = lidar_msg.height
    #     self.lidar_pointcloud_writer.write(neolix_pointcloud)

    def cmd_callback(self, cmd_msg):
        carla_cmd = CarlaMsg()
        carla_cmd.throttle = cmd_msg.throttle
        carla_cmd.brake = cmd_msg.brake
        carla_cmd.steering_rate = cmd_msg.steering_rate
        carla_cmd.steering_target = -cmd_msg.steering_target
        carla_cmd.driving_mode = cmd_msg.driving_mode
        carla_cmd.gear_location = cmd_msg.gear_location
        self.carla_msg_write.write(carla_cmd)

    def pose_callback(self, pose_msg):
        pose = NeolixLocalizationEstimate()
        pose.header.timestamp_sec = pose_msg.header.timestamp_sec
        pose.header.module_name = 'perception_localization_pose'
        pose.header.sequence_num = self.sequence_num_pose
        pose.pose.position.x = pose_msg.pose.position.x
        pose.pose.position.y = pose_msg.pose.position.y
        pose.pose.position.z = pose_msg.pose.position.z
        pose.pose.orientation.qx = pose_msg.pose.orientation.qx
        pose.pose.orientation.qy = pose_msg.pose.orientation.qy
        pose.pose.orientation.qz = pose_msg.pose.orientation.qz
        pose.pose.orientation.qw = pose_msg.pose.orientation.qw
        pose.pose.linear_velocity.x = pose_msg.pose.linear_velocity.x
        pose.pose.linear_velocity.y = pose_msg.pose.linear_velocity.y
        pose.pose.linear_velocity.z = pose_msg.pose.linear_velocity.z
        pose.pose.linear_acceleration.x = pose_msg.pose.linear_acceleration_vrf.x
        pose.pose.linear_acceleration.y = pose_msg.pose.linear_acceleration_vrf.y
        pose.pose.linear_acceleration.z = pose_msg.pose.linear_acceleration_vrf.z
        pose.pose.angular_velocity.x = pose_msg.pose.angular_velocity_vrf.x
        pose.pose.angular_velocity.y = pose_msg.pose.angular_velocity_vrf.y
        pose.pose.angular_velocity.z = pose_msg.pose.angular_velocity_vrf.z
        self.carla_pose_write.write(pose)
        self.sequence_num_pose +=1

if __name__ == '__main__':
    MsgTransform()






