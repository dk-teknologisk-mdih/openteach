"""
This script is for replaying demonstrations from .pkl files and from the RLDS files
for verification and debugging purposes.

In order to run this script, you need to have the following running on the NUC

cd deoxys_control/deoxys && ./auto_scripts/auto_arm.sh config/charmander.yml
cd deoxys_control/deoxys && ./auto_scripts/auto_gripper.sh config/charmander.yml

"""

# deoxys_control
from deoxys.franka_interface import FrankaInterface
import deoxys.proto.franka_interface.franka_controller_pb2 as franka_controller_pb2
from deoxys.utils.config_utils import get_default_controller_config
from deoxys.utils.log_utils import get_deoxys_example_logger
# from examples.osc_control import move_to_target_pose, deltas_move
from deoxys.experimental.motion_utils import reset_joints_to
from deoxys.utils.transform_utils import quat2axisangle, mat2euler, mat2quat, quat_distance, quat2mat, euler2mat, axisangle2quat, quat_multiply

# General
import numpy as np
import pickle as pkl
import h5py
import tensorflow_datasets as tfds
import math
import argparse
import cv2
from time import sleep
import sys
import importlib
from scipy.spatial.transform import Rotation as R
import os
import time
from openteach.utils.timer import FrequencyTimer
from easydict import EasyDict
from matplotlib import pyplot as plt

# Directory containing charmander.yml and other configs (next to this script).
CONFIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "configs")

parser = argparse.ArgumentParser(description="Replay demonstrations from .pkl or RLDS files.")
parser.add_argument(
    "--source", choices=["pkl", "rlds"], default="pkl",
    help="Replay from a .pkl file or from an RLDS dataset.",
)
parser.add_argument(
    "--pkl", type=str, default=None,
    help="Path to the .pkl demonstration file (used when --source pkl).",
)
parser.add_argument(
    "--rlds-dataset", type=str, default="openteach_franka",
    help="Name of the RLDS dataset to load (used when --source rlds).",
)
parser.add_argument(
    "--rlds-builder-dir", type=str, default=None,
    help="Optional path to the RLDS dataset builder to add to sys.path (used when --source rlds).",
)
parser.add_argument(
    "--rlds-data-dir", type=str,
    default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "vla_data", "rlds"),
    help="TFDS data directory to load the RLDS dataset from (default: vla_data/rlds).",
)
parser.add_argument(
    "--config-dir", type=str, default=CONFIG_DIR,
    help="Directory containing charmander.yml.",
)
parser.add_argument(
    "--control-freq", type=float, default=15.0,
    help="Control frequency (Hz) for replay. Higher is faster; match the "
         "data collection rate (~15 Hz) for real-time playback.",
)

DEFAULT_CONTROLLER = EasyDict({
    'controller_type': 'OSC_POSE',
    'is_delta': False,
    'traj_interpolator_cfg': {
        'traj_interpolator_type': 'LINEAR_POSE',
        'time_fraction': 0.3
    },
    'Kp': {
        'translation': [250.0, 250.0, 250.0],
        'rotation': [250.0, 250.0, 250.0]
    },
    'action_scale': {
        'translation': 1.0,
        'rotation': 1.0
    },
    'residual_mass_vec': [0.0, 0.0, 0.0, 0.0, 0.1, 0.5, 0.5],
    'state_estimator_cfg': {
        'is_estimation': False,
        'state_estimator_type': 'EXPONENTIAL_SMOOTHING',
        'alpha_q': 0.9,
        'alpha_dq': 0.9,
        'alpha_eef': 1.0,
        'alpha_eef_vel': 1.0
    }
})

def replay_from_rlds(args):
    robot_interface = FrankaInterface(
        os.path.join(args.config_dir, 'charmander.yml'), use_visualizer=False,
        control_freq=args.control_freq,
        state_freq=200
    )
    reset_joint_positions = [
            0.09162008114028396,
            -0.19826458111314524,
            -0.01990020486871322,
            -2.4732269941140346,
            -0.01307073642274261,
            2.30396583422025,
            0.8480939705504309,
        ]
    reset_joints_to(robot_interface, reset_joint_positions)

    # Load demonstration data
    if args.rlds_builder_dir:
        sys.path.append(args.rlds_builder_dir)
    ds = tfds.load(args.rlds_dataset, split='train', data_dir=args.rlds_data_dir)

    # timer = FrequencyTimer(15)
    for episode in ds.take(1):
        for st in episode['steps']:
            # breakpoint()
            # timer.start_loop()
            action = st['action'].numpy()  # deltas for (x, y, z, r, p, y, gripper)
            state = st['observation']['state'].numpy()  # [xyz(3), euler(3), pad(1), gripper(1)]
            cv2.imshow("image", st['observation']['image'].numpy()[:, :, ::-1])  # convert to BGR for cv2
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            # DEFAULT_CONTROLLER is is_delta=False, so it expects an ABSOLUTE
            # target pose. The RLDS action is a delta, so reconstruct the
            # absolute pose from the current EEF state exactly like the pkl path:
            #   abs_pos  = cur_pos + delta_pos
            #   abs_quat = delta_quat * cur_quat
            delta_pos = action[:3]
            delta_quat = mat2quat(euler2mat(action[3:6]))
            cur_pos = state[:3]
            cur_quat = mat2quat(euler2mat(state[3:6]))

            target_pos = cur_pos + delta_pos
            target_axisangle = quat2axisangle(quat_multiply(delta_quat, cur_quat))
            arm_action = np.concatenate([target_pos, target_axisangle])
            full_action = np.concatenate([arm_action, [action[6]]])  # include the gripper value too
            print(" ".join(f"{v:+8.4f}" for v in full_action))
            robot_interface.control(
                    controller_type='OSC_POSE',
                    action=arm_action,
                    controller_cfg=DEFAULT_CONTROLLER,
                )
            # RLDS gripper: 1.0 == open, 0.0 == close. gripper_control expects
            # {-1 (open), +1 (close)}, so map back.
            robot_interface.gripper_control(-1 if action[6] >= 0.5 else 1)
            # timer.end_loop()



def replay_from_pkl(args):
    if args.pkl is None:
        raise ValueError("--pkl must be provided when --source pkl")
    filename = os.path.expanduser(args.pkl)
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Demonstration file {filename} does not exist.")
    with open(filename, 'rb') as dbfile:
        db = pkl.load(dbfile)
    # breakpoint()
    pos = db['eef_pos'].squeeze() + db['arm_action'][:, :3]
    quat_rot_actions = [axisangle2quat(x) for x in db['arm_action'][:, 3:]]
    rot = np.array([
        quat2axisangle(quat_multiply(i, j)) for i,j in \
            zip(quat_rot_actions, db['eef_quat'])
            ])
    arm_action = np.hstack((pos, rot))

    # breakpoint()

    # images = []
    # for i in db['rgb_frames'][:, 2]: # grab from the right camera
    #     images.append(cv2.cvtColor(i, cv2.COLOR_BGR2RGB))
    #     cv2.imshow("Camera", i)  # convert back to BGR for cv2
    #     if cv2.waitKey(30) & 0xFF == ord('q'):
    #         break
    # cv2.destroyAllWindows()
    # image_strip = np.concatenate(images[::4], axis=1)
    # plt.figure()
    # plt.imshow(image_strip)
    # plt.show()
    # breakpoint()
    robot_interface = FrankaInterface(
        os.path.join(args.config_dir, 'charmander.yml'), use_visualizer=False,
        control_freq=args.control_freq,  # setting control frequency here so we don't have to handle it with a timer
        state_freq=200
    )
    # timer = FrequencyTimer(15)

    # move robot to start position
    reset_joints_to(robot_interface, db['joint_pos'][0])
    for i in range(0, len(arm_action)):
        # timer.start_loop()

        deltas = arm_action[i]
        full_action = np.concatenate([deltas, [db["gripper_action"][i]]])  # include the gripper value too
        print(" ".join(f"{v:+8.4f}" for v in full_action))
        robot_interface.control(
                controller_type=DEFAULT_CONTROLLER["controller_type"],
                action=deltas,
                controller_cfg=DEFAULT_CONTROLLER,
            )

        robot_interface.gripper_control(db["gripper_action"][i])
        # timer.end_loop()


if __name__ == "__main__":
    args = parser.parse_args()
    if args.source == "rlds":
        replay_from_rlds(args)
    else:
        replay_from_pkl(args)
