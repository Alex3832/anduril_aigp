# Made by Jack Yeulenski 4/26/2026
# mavlink interface for communication with simulation

import sys

from pymavlink import mavutil
import time

class mavlink_interface:
    def __init__(self, connection_string):
        # connection string has the format: [protocol:]address[:port]

        # will default to a UDP address if no protocol is specified
        # udpin: Listen for a UDP connection on the specified address and port.

        # address: IP address, serial port name, or file name
        # port: IP port (only if address is an IP address)

        # LIKELY CHOICE --> MAVLink API listening for SITL connection via UDP:	udpin:localhost:14540 (or udp:localhost:14540, 127.0.0.1:14540,etc.)
        # MAVLink API initiating a connection to SITL via UDP:	udpout:localhost:14540 (or udpout:127.0.0.1:14540)
        

        self.connection = mavutil.mavlink_connection(connection_string)
        self.connection.wait_heartbeat()
        print("Heartbeat from system (system %u component %u)" % (self.connection.target_system, self.connection.target_component))

    ## Commands

    def command_position(self, params):
        #comands a position in local coordinates x, y, z

        type_mask = (
            mavutil.mavlink.POSITION_TARGET_TYPEMASK_X_IGNORE |
            mavutil.mavlink.POSITION_TARGET_TYPEMASK_Y_IGNORE |
            mavutil.mavlink.POSITION_TARGET_TYPEMASK_Z_IGNORE |
            mavutil.mavlink.POSITION_TARGET_TYPEMASK_VX_IGNORE |
            mavutil.mavlink.POSITION_TARGET_TYPEMASK_VY_IGNORE |
            mavutil.mavlink.POSITION_TARGET_TYPEMASK_VZ_IGNORE |
            mavutil.mavlink.POSITION_TARGET_TYPEMASK_AX_IGNORE |
            mavutil.mavlink.POSITION_TARGET_TYPEMASK_AY_IGNORE |
            mavutil.mavlink.POSITION_TARGET_TYPEMASK_AZ_IGNORE |
            mavutil.mavlink.POSITION_TARGET_TYPEMASK_YAW_IGNORE |
            mavutil.mavlink.POSITION_TARGET_TYPEMASK_YAW_RATE_IGNORE
        )
        
        self.connection.mav.set_position_target_local_ned_send(
            int(time.time() * 1000), # time_boot_ms
            self.connection.target_system, # target_system
            self.connection.target_component, # target_component
            mavutil.mavlink.MAV_FRAME_LOCAL_NED, # coordinate frame
            type_mask, # type_mask (only position)
            params[0], params[1], params[2], # x, y, z positions (used)
            0, 0, 0, # x, y, z velocity in m/s (not used)
            0, 0, 0, # x, y, z acceleration (not used)
            0, 0 #  Yaw in radians (not used), yaw rate in rad/s (not used)
        )

    def command_velocity_yaw(self, params):
        #comands a velocity vector in local coordinates in x, y, z direction. 
        #commands a yaw.

        type_mask = (
            mavutil.mavlink.POSITION_TARGET_TYPEMASK_X_IGNORE |
            mavutil.mavlink.POSITION_TARGET_TYPEMASK_Y_IGNORE |
            mavutil.mavlink.POSITION_TARGET_TYPEMASK_Z_IGNORE |
            mavutil.mavlink.POSITION_TARGET_TYPEMASK_AX_IGNORE |
            mavutil.mavlink.POSITION_TARGET_TYPEMASK_AY_IGNORE |
            mavutil.mavlink.POSITION_TARGET_TYPEMASK_AZ_IGNORE |
            mavutil.mavlink.POSITION_TARGET_TYPEMASK_YAW_RATE_IGNORE
        )
        
        self.connection.mav.set_position_target_local_ned_send(
            int(time.time() * 1000), # time_boot_ms
            self.connection.target_system, # target_system
            self.connection.target_component, # target_component
            mavutil.mavlink.MAV_FRAME_LOCAL_NED, # coordinate frame
            type_mask, # type_mask (only velocity and yaw rate enabled)
            0, 0, 0, # x, y, z positions (not used)
            params[0], params[1], params[2], # x, y, z velocity in m/s
            0, 0, 0, # x, y, z acceleration (not used)
            params[3], 0 #  Yaw in radians (used), yaw rate in rad/s (not used)
        )

    ## receive methods
    # The returned object is the subclass of MAVLink_message for the specific message. 
    # You can access the message fields as class attributes.

    def receive_IMU(self):
        msg = self.connection.recv_match(type='HIGHRES_IMU',blocking=True)
        if not msg:
            return
        if msg.get_type() == "BAD_DATA":
            if mavutil.all_printable(msg.data):
                sys.stdout.write(msg.data)
                sys.stdout.flush()
        else:
            return msg

    def receive_attitude(self):
        msg = self.connection.recv_match(type='ATTITUDE',blocking=True)
        if not msg:
            return
        if msg.get_type() == "BAD_DATA":
            if mavutil.all_printable(msg.data):
                sys.stdout.write(msg.data)
                sys.stdout.flush()
        else:
            return msg
