# Original code authored by Xue Bin Peng (MimicKit)
# Modified by [LocalAIVRLab] in 2026 for integration with Virt-A-Mate (VaM).
# Licensed under the Apache License, Version 2.0.

import gymnasium.spaces as spaces
import os
import socket
import struct
import numpy as np
import torch

import engines.engine as engine
import envs.deepmimic_env as deepmimic_env
import util.circular_buffer as circular_buffer
import util.torch_util as torch_util
#import envs.task_location_env as task_location_env
from scipy.spatial.transform import Rotation as R

#from .deepmimic_env import compute_done

class AMPEnv(deepmimic_env.DeepMimicEnv):
    def __init__(self, env_config, engine_config, num_envs, device, visualize, record_video=False):
        #self._rand_reset = env_config.get("rand_reset", "random")
        self._num_disc_obs_steps = env_config["num_disc_obs_steps"]
        self._disc_dof_vel_obs = env_config.get("disc_dof_vel_obs", True)
        self._tar_speed = float(env_config.get("tar_speed", 1.0))
        self._tar_change_time_min = float(env_config.get("tar_change_time_min", 5.0))
        self._tar_change_time_max = float(env_config.get("tar_change_time_max", 10.0))
        self._tar_dist_min = float(env_config.get("tar_dist_min", 0.0))
        self._tar_dist_max = float(env_config.get("tar_dist_max", 5.0))
        self._dist_threshold = float(env_config.get("dist_threshold", 0.5))
        self._pos_err_scale = float(env_config.get("pos_err_scale", 0.5))
        self._vel_err_scale = float(env_config.get("vel_err_scale", 4.0))
        self._pos_reward_w = float(env_config.get("pos_reward_w", 0.5))
        self._vel_reward_w = float(env_config.get("vel_reward_w", 0.4))
        self._face_reward_w = float(env_config.get("face_reward_w", 0.1))
        assert self._tar_dist_min <= self._tar_dist_max
        assert self._tar_change_time_min <= self._tar_change_time_max
        self._marker_ids = []
       
        self._vrhand_enabled = bool(env_config.get("vrhand_udp_enabled", visualize))
        self._vrhand_ip = env_config.get("vrhand_udp_ip", "0.0.0.0")
        self._vrhand_port = int(env_config.get("vrhand_udp_port", 9999))
        self._vrhand_packet = struct.Struct("<23f")
        # VRhand offset is independent of the existing VaM retarget correction.
        self._vrhand_position_offset = np.asarray(
            env_config.get("vrhand_position_offset", [0.0, 0.15, 0.0]), dtype=np.float64
        )                 
        self._prev_vrhand_pos = {}
        self._vrhand_debug = bool(env_config.get("vrhand_debug", True))
        self._vrhand_packet_count = 0
        self._vrhand_update_count = 0
        self._vrhand_bad_packet_count = 0
        self._vrhand_socket = None
        self._vrhand_box_ids = []
        self._vrhand_latest = None
        self._vrhand_asset = os.path.join("data", "assets", "objects", "vrhand_collision_box.xml")
        self._vrhmd_asset = os.path.join("data", "assets", "objects", "vrhmd_collision_box.xml")
        if self._vrhand_enabled:
            self._write_vrhand_asset()
            self._write_vrhmd_asset()
            
            self._vrhand_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self._vrhand_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self._vrhand_socket.bind((self._vrhand_ip, self._vrhand_port))
            self._vrhand_socket.setblocking(False)
            if self._vrhand_debug:
                print(f"[VRhand] UDP listening on {self._vrhand_ip}:{self._vrhand_port}; packet={self._vrhand_packet.size} bytes; offset={self._vrhand_position_offset.tolist()}", flush=True)                 
        super().__init__(env_config=env_config, engine_config=engine_config, num_envs=num_envs, device=device,
                         visualize=visualize, record_video=record_video)                 
        self._vam_enabled = env_config.get("vam_udp_enabled", visualize)
        self._vam_ip = env_config.get("vam_udp_ip", "192.168.2.100")
        self._vam_port = int(env_config.get("vam_udp_port", 9998))
        self._vam_packet = struct.Struct("<133f")
        self._vam_position_offset = np.asarray(env_config.get("vam_position_offset", [0.0, 0.06, 0.0]), dtype=np.float32)
        self._vam_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM) if self._vam_enabled else None

        return

    def _build_envs(self, config, num_envs):
        if self._visualize:
            self._marker_ids = []
        super()._build_envs(config, num_envs)
        return

    def _build_env(self, env_id, config):
        super()._build_env(env_id, config)
        if self._visualize:
            marker_id = self._build_marker(env_id)
            if env_id == 0:
                self._marker_ids.append(marker_id)
            else:
                assert self._marker_ids[0] == marker_id
        if self._vrhand_enabled:
            print("start")
            left_id = self._engine.create_obj(
                env_id=env_id, obj_type=engine.ObjType.rigid,
                asset_file=self._vrhand_asset, name="vrhand_left_collision_box",
                is_visual=False, fix_root=False, color=[0.2, 0.65, 1.0] # False に変更
            )
            right_id = self._engine.create_obj(
                env_id=env_id, obj_type=engine.ObjType.rigid,
                asset_file=self._vrhand_asset, name="vrhand_right_collision_box",
                is_visual=False, fix_root=False, color=[1.0, 0.4, 0.2] # False に変更
            )
            hmd_id = self._engine.create_obj(
                env_id=env_id, obj_type=engine.ObjType.rigid,
                asset_file=self._vrhmd_asset, name="vrhmd_collision_box",
                is_visual=False, fix_root=False, color=[0.4, 1.0, 0.2] # False に変更
            )
            if env_id == 0:
                self._vrhand_box_ids = [left_id, right_id, hmd_id]
                if self._vrhand_debug:
                    print(f"[VRhand] collision boxes created: left_id={left_id}, right_id={right_id}, hmd_id={hmd_id} env_id={env_id}, asset={self._vrhand_asset}", flush=True)
            else:
                assert self._vrhand_box_ids == [left_id, right_id, hmd_id]
        return

    # def _build_marker(self, env_id):
        # return self._engine.create_obj(env_id=env_id,
                                       # obj_type=engine.ObjType.rigid,
                                       # asset_file="data/assets/objects/location_marker.xml",
                                       # name="marker",
                                       # is_visual=True,
                                       # fix_root=True,
                                       # color=[0.8, 0.0, 0.0])

    # def _get_marker_id(self):
        # return self._marker_ids[0]

    # def _update_marker(self, env_ids):
        # marker_id = self._get_marker_id()
        # rot = torch.tensor([0, 0, 0, 1], device=self._device, dtype=torch.float32)
        # self._engine.set_root_pos(env_ids, marker_id, self._tar_pos[env_ids])
        # self._engine.set_root_rot(env_ids, marker_id, rot)
        # self._engine.set_root_vel(env_ids, marker_id, 0.0)
        # self._engine.set_root_ang_vel(env_ids, marker_id, 0.0)
        # return

    def _build_sim_tensors(self, config):
        super()._build_sim_tensors(config)
        
        if self._vrhand_enabled and len(self._vrhand_box_ids) == 3:
            num_envs = self.get_num_envs()
            env_ids = torch.arange(num_envs, device=self._device, dtype=torch.long)
            
            # 左手の退避位置: 空中左側 (x = -2.0, y = 0.0, z = 10.0)
            left_parked_pos = torch.zeros((num_envs, 3), device=self._device, dtype=torch.float32)
            left_parked_pos[:, 0] = -2.0
            left_parked_pos[:, 2] = 10.0
            
            # 右手の退避位置: 空中右側 (x = 2.0, y = 0.0, z = 10.0)
            right_parked_pos = torch.zeros((num_envs, 3), device=self._device, dtype=torch.float32)
            right_parked_pos[:, 0] = 2.0
            right_parked_pos[:, 2] = 10.0
            
            hmd_parked_pos = torch.zeros((num_envs, 3), device=self._device, dtype=torch.float32)
            hmd_parked_pos[:, 1] = 2.0
            hmd_parked_pos[:, 2] = 10.0
            
            # それぞれ別の座標に配置
            left_id, right_id, hmd_id = self._vrhand_box_ids[0], self._vrhand_box_ids[1], self._vrhand_box_ids[2]
            
            self._engine.set_root_pos(env_ids, left_id, left_parked_pos)
            self._engine.set_root_vel(env_ids, left_id, 0.0)
            self._engine.set_root_ang_vel(env_ids, left_id, 0.0)
            
            self._engine.set_root_pos(env_ids, right_id, right_parked_pos)
            self._engine.set_root_vel(env_ids, right_id, 0.0)
            self._engine.set_root_ang_vel(env_ids, right_id, 0.0)
            
            self._engine.set_root_pos(env_ids, hmd_id, hmd_parked_pos)
            self._engine.set_root_vel(env_ids, hmd_id, 0.0)
            self._engine.set_root_ang_vel(env_ids, hmd_id, 0.0)
            
            
            
            if self._vrhand_debug:
                print(f"[VRhand] simulation tensors ready; left parked at (-2, 0, 10), right parked at (2, 0, 10) for {num_envs} env(s)", flush=True)
        
        num_envs = self.get_num_envs()
        self._tar_pos = torch.zeros([num_envs, 3], device=self._device, dtype=torch.float)
        self._tar_change_times = torch.zeros([num_envs], device=self._device, dtype=torch.float)
        self._prev_root_pos = torch.zeros([num_envs, 3], device=self._device, dtype=torch.float)
        return

    def _pre_physics_step(self, actions):
        super()._pre_physics_step(actions)
        char_id = self._get_char_id()
        self._prev_root_pos[:] = self._engine.get_root_pos(char_id)
        return

    def _compute_obs(self, env_ids=None):
        obs = super()._compute_obs(env_ids)
        char_id = self._get_char_id()
        root_pos = self._engine.get_root_pos(char_id)
        root_rot = self._engine.get_root_rot(char_id)
        tar_pos = self._tar_pos
        if env_ids is not None:
            root_pos = root_pos[env_ids]
            root_rot = root_rot[env_ids]
            tar_pos = tar_pos[env_ids]
        location_obs = task_location_env.compute_location_observations(
            root_pos, root_rot, tar_pos
        )
        return torch.cat([obs, location_obs], dim=-1)

    def get_disc_obs_space(self):
        disc_obs = self.fetch_disc_obs_demo(1)
        disc_obs_shape = list(disc_obs.shape[1:])
        disc_obs_dtype = torch_util.torch_dtype_to_numpy(disc_obs.dtype)
        disc_obs_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=disc_obs_shape,
            dtype=disc_obs_dtype,
        )
        return disc_obs_space

    def get_num_disc_obs_steps(self):
        return self._num_disc_obs_steps

    def fetch_disc_obs_demo(self, num_samples):
        motion_ids, motion_times0 = self._sample_motion_times(num_samples)
        disc_obs = self._compute_disc_obs_demo(motion_ids, motion_times0)
        return disc_obs

    def _compute_disc_obs_demo(self, motion_ids, motion_times0):
        root_pos, root_rot, root_vel, root_ang_vel, joint_rot, dof_vel, body_pos = self._fetch_disc_demo_data(motion_ids, motion_times0)
        
        if (self._track_global_root()):
            ref_root_pos = torch.zeros_like(root_pos[..., -1, :])
            ref_root_rot = torch.zeros_like(root_rot[..., -1, :])
            ref_root_rot[..., -1] = 1
        else:
            ref_root_pos = root_pos[..., -1, :]
            ref_root_rot = root_rot[..., -1, :]
            
        if (self._has_key_bodies()):
            key_pos = body_pos[..., self._key_body_ids, :]
        else:
            key_pos = torch.zeros([0], device=self._device)
        
        disc_obs = compute_disc_obs(ref_root_pos=ref_root_pos,
                                  ref_root_rot=ref_root_rot,
                                  root_pos=root_pos,
                                  root_rot=root_rot, 
                                  root_vel=root_vel,
                                  root_ang_vel=root_ang_vel,
                                  joint_rot=joint_rot,
                                  dof_vel=dof_vel,
                                  key_pos=key_pos,
                                  global_obs=self._global_obs,
                                  root_height_obs=self._root_height_obs,
                                  dof_vel_obs=self._disc_dof_vel_obs)
        return disc_obs

    def _fetch_disc_demo_data(self, motion_ids, motion_times0):
        num_samples = motion_ids.shape[0]

        motion_ids = torch.tile(motion_ids.unsqueeze(-1), [1, self._num_disc_obs_steps])
        motion_times = motion_times0.unsqueeze(-1)
        time_steps = -self._engine.get_timestep() * torch.arange(0, self._num_disc_obs_steps, device=self._device)
        time_steps = torch.flip(time_steps, dims=[0])
        motion_times = motion_times + time_steps

        motion_ids = motion_ids.view(-1)
        motion_times = motion_times.view(-1)
        root_pos, root_rot, root_vel, root_ang_vel, joint_rot, dof_vel = self._motion_lib.calc_motion_frame(motion_ids, motion_times)
        
        body_pos, _ = self._kin_char_model.forward_kinematics(root_pos, root_rot, joint_rot)

        root_pos = torch.reshape(root_pos, shape=[num_samples, self._num_disc_obs_steps, root_pos.shape[-1]])
        root_rot = torch.reshape(root_rot, shape=[num_samples, self._num_disc_obs_steps, root_rot.shape[-1]])
        root_vel = torch.reshape(root_vel, shape=[num_samples, self._num_disc_obs_steps, root_vel.shape[-1]])
        root_ang_vel = torch.reshape(root_ang_vel, shape=[num_samples, self._num_disc_obs_steps, root_ang_vel.shape[-1]])
        joint_rot = torch.reshape(joint_rot, shape=[num_samples, self._num_disc_obs_steps, joint_rot.shape[-2], joint_rot.shape[-1]])
        dof_vel = torch.reshape(dof_vel, shape=[num_samples, self._num_disc_obs_steps, dof_vel.shape[-1]])
        body_pos = torch.reshape(body_pos, shape=[num_samples, self._num_disc_obs_steps, body_pos.shape[-2], body_pos.shape[-1]])
        
        return root_pos, root_rot, root_vel, root_ang_vel, joint_rot, dof_vel, body_pos

    def _build_data_buffers(self):
        super()._build_data_buffers()
        self._build_disc_obs_buffers()
        self._info["disc_obs"] = self._disc_obs_buf
        return

    def _build_disc_obs_buffers(self):
        num_envs = self.get_num_envs()
        n = self._num_disc_obs_steps
        
        char_id = self._get_char_id()
        root_pos = self._engine.get_root_pos(char_id)
        root_rot = self._engine.get_root_rot(char_id)
        root_vel = self._engine.get_root_vel(char_id)
        root_ang_vel = self._engine.get_root_ang_vel(char_id)
        dof_pos = self._engine.get_dof_pos(char_id)
        dof_vel = self._engine.get_dof_vel(char_id)
        body_pos = self._engine.get_body_pos(char_id)
        
        joint_rot = self._kin_char_model.dof_to_rot(dof_pos)

        self._disc_hist_root_pos = circular_buffer.CircularBuffer(batch_size=num_envs,
                                                                 buffer_len=n, 
                                                                 shape=root_pos.shape[1:],
                                                                 dtype=root_pos.dtype,
                                                                 device=self._device)

        self._disc_hist_root_rot = circular_buffer.CircularBuffer(batch_size=num_envs,
                                                                 buffer_len=n,
                                                                 shape=root_rot.shape[1:],
                                                                 dtype=root_rot.dtype,
                                                                 device=self._device)

        self._disc_hist_root_vel = circular_buffer.CircularBuffer(batch_size=num_envs,
                                                                 buffer_len=n,
                                                                 shape=root_vel.shape[1:],
                                                                 dtype=root_vel.dtype,
                                                                 device=self._device)
        
        self._disc_hist_root_ang_vel = circular_buffer.CircularBuffer(batch_size=num_envs,
                                                                     buffer_len=n,
                                                                     shape=root_ang_vel.shape[1:],
                                                                     dtype=root_ang_vel.dtype,
                                                                     device=self._device)
        
        self._disc_hist_joint_rot = circular_buffer.CircularBuffer(batch_size=num_envs,
                                                                  buffer_len=n,
                                                                  shape=joint_rot.shape[1:],
                                                                  dtype=joint_rot.dtype,
                                                                  device=self._device)

        self._disc_hist_dof_vel = circular_buffer.CircularBuffer(batch_size=num_envs,
                                                                buffer_len=n,
                                                                shape=dof_vel.shape[1:],
                                                                dtype=dof_vel.dtype,
                                                                device=self._device)
        
        self._disc_hist_body_pos = circular_buffer.CircularBuffer(batch_size=num_envs,
                                                                 buffer_len=n,
                                                                 shape=body_pos.shape[1:],
                                                                 dtype=body_pos.dtype,
                                                                 device=self._device)

        disc_obs_space = self.get_disc_obs_space()
        disc_obs_dtype = torch_util.numpy_dtype_to_torch(disc_obs_space.dtype)
        self._disc_obs_buf = torch.zeros([num_envs] + list(disc_obs_space.shape), device=self._device, dtype=disc_obs_dtype)
        return

    def _update_misc(self):
        super()._update_misc()
        self._update_disc_hist()
        self._update_task()
        self._send_vam_udp()
        self._update_vrhand_boxes()
        return

    def _update_task(self):
        # reset_task_mask = self._time_buf >= self._tar_change_times
        # reset_env_ids = reset_task_mask.nonzero(as_tuple=False).flatten()
        # if reset_env_ids.numel() > 0:
            # self._reset_location_task(reset_env_ids)

        return

    def _reset_location_task(self, env_ids):
        self._reset_location_target(env_ids)
        rand_dt = torch.rand(env_ids.shape[0], device=self._device, dtype=torch.float)
        rand_dt = (self._tar_change_time_max - self._tar_change_time_min) * rand_dt + self._tar_change_time_min
        self._tar_change_times[env_ids] = self._time_buf[env_ids] + rand_dt
        if self._visualize:
            self._update_marker(env_ids)
        return

    def _reset_location_target(self, env_ids):
        char_id = self._get_char_id()
        root_pos = self._engine.get_root_pos(char_id)
        char_root_pos = root_pos[env_ids, 0:2]
        rand_dist = (self._tar_dist_max - self._tar_dist_min) * torch.rand_like(char_root_pos[..., 0]) + self._tar_dist_min
        rand_theta = 2 * np.pi * torch.rand_like(char_root_pos[..., 0])
        rand_pos = char_root_pos.clone()
        rand_pos[..., 0] += rand_dist * torch.cos(rand_theta)
        rand_pos[..., 1] += rand_dist * torch.sin(rand_theta)
        self._tar_pos[env_ids, 0:2] = rand_pos
        self._tar_pos[env_ids, 2] = 0.0
        return
        
    def _write_vrhand_asset(self):
        os.makedirs(os.path.dirname(self._vrhand_asset), exist_ok=True)
        xml = """<mujoco model="vrhand_collision_box">
  <worldbody>
    <body name="vrhand_collision_box" pos="10 10 10">
      <freejoint name="root"/>                        
      <geom name="vrhand_box_geom" type="sphere" size="0.015"
            rgba="0.2 0.65 1.0 0.65"
            mass="70.0"
            solref="0.15 1.6" solimp="0.5 0.99 0.01 0.5 2"/>
    </body>
  </worldbody>
</mujoco>
"""
        with open(self._vrhand_asset, "w", encoding="utf-8") as f:
            f.write(xml)
            
    def _write_vrhmd_asset(self):
        os.makedirs(os.path.dirname(self._vrhmd_asset), exist_ok=True)
        xml = """<mujoco model="vrhmd_collision_box">
  <worldbody>
    <body name="vrhmd_collision_box" pos="10 10 10">
      <freejoint name="root"/>                        
      <geom name="vrhmd_box_geom" type="capsule" size="0.06 0.2" pos="0 0 0.2"
            rgba="0.2 0.65 1.0 0.65"
            mass="10.0"
            solref="0.15 1.6" solimp="0.5 0.99 0.01 0.5 2"/>
      
    </body>
  </worldbody>
</mujoco>
"""
        with open(self._vrhmd_asset, "w", encoding="utf-8") as f:
            f.write(xml)
            
    def _vam_pos_to_engine(self, pos_vam):
        p = np.asarray(pos_vam, dtype=np.float64) - self._vrhand_position_offset
        return np.asarray([p[2], -p[0], p[1]], dtype=np.float32)

    def _vam_quat_to_engine(self, quat_vam_xyzw):
        qx, qy, qz, qw = np.asarray(quat_vam_xyzw, dtype=np.float64)
        q = np.asarray([-qx, -qz, -qy, qw], dtype=np.float64)
        norm = np.linalg.norm(q)
        if not np.isfinite(q).all() or norm < 1e-8:
            return np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
        return (q / norm).astype(np.float32)
        
    def _update_vrhand_boxes(self):
        if not self._vrhand_enabled or self._vrhand_socket is None:
            return
        # 💡 self._vrhand_box_ids を [左手ID, 右手ID, HMD_ID] の3つの要素にしてください
        if len(self._vrhand_box_ids) != 3:
            return

        if not hasattr(self, '_prev_vrhand_pos'):
            self._prev_vrhand_pos = {}
            self._vrhand_triggers = {"left": 0.0, "right": 0.0}
            self._vrhand_grabbed_targets = {"left": None, "right": None}

        # 💡 パケットサイズを 92 バイト（float 23個分：左8 + 右8 + HMD7）に変更
        # ※ self._vrhand_packet（struct）のフォーマットも '23f' に書き換えてください
        EXPECTED_SIZE = 92 

        latest_packet = None
        while True:
            try:
                packet, addr = self._vrhand_socket.recvfrom(1024)
            except BlockingIOError:
                break
            if len(packet) == EXPECTED_SIZE:
                latest_packet = packet
                self._vrhand_packet_count += 1
            else:
                # 💡 想定外のサイズが来ている場合の警告デバッグ
                print(f"[VRHand Debug] Warning: Received invalid packet size: {len(packet)} bytes (Expected: {EXPECTED_SIZE})")

        if latest_packet is not None:
            # 23個のfloatを展開
            self._vrhand_latest = self._vrhand_packet.unpack(latest_packet)
            
            # トリガー値を取得 (左: 7, 右: 15)
            self._vrhand_triggers["left"] = float(self._vrhand_latest[7])
            self._vrhand_triggers["right"] = float(self._vrhand_latest[15])

        if self._vrhand_latest is None:
            return

        # # 💡 --- HMD同期＆データ検証用デバッグ出力 (約1秒=30回に1回出力) ---
        # if self._vrhand_update_count % 30 == 0:
            # print(f"\n================ [VRHand Update Debug #{self._vrhand_update_count}] ================")
            # print(f"Total Floats Unpacked: {len(self._vrhand_latest)}")
            # print(f"Triggers -> Left: {self._vrhand_triggers['left']:.2f} | Right: {self._vrhand_triggers['right']:.2f}")
            
            # names = ["Left Hand", "Right Hand", "HMD"]
            # offsets = [0, 8, 16]
            # for name, offset in zip(names, offsets):
                # raw_pos = self._vrhand_latest[offset:offset+3]
                # raw_quat = self._vrhand_latest[offset+3:offset+7]
                # print(f"[{name}] Raw UDP Pos: {raw_pos} | Quat: {raw_quat}")
            # print("========================================================================\n")
        # # ------------------------------------------------------------------

        env_ids = torch.tensor([0], device=self._device, dtype=torch.long)
        dt = 1.0 / 30.0
        gravity = 9.81

        # 各データのオフセットと、対応するオブジェクトIDのペア
        # 左手(0番目から7個+トリガー1), 右手(8番目から7個+トリガー1), HMD(16番目から7個)
        targets = (
            (0, self._vrhand_box_ids[0]), # 左手
            (8, self._vrhand_box_ids[1]), # 右手
            (16, self._vrhand_box_ids[2]) # HMD
        )

        for offset, obj_id in targets:
            values = self._vrhand_latest
            pos_vam = np.asarray(values[offset:offset + 3], dtype=np.float64)
            #quat_vam = np.asarray(values[offset + 3:offset + 7], dtype=np.float64)
            quat_vam = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
            
            
            if not np.isfinite(pos_vam).all() or not np.isfinite(quat_vam).all() or np.linalg.norm(quat_vam) < 1e-8:
                # 💡 データが不正でスキップされた場合のデバッグログ
                if self._vrhand_update_count % 30 == 0:
                    target_name = "LeftHand" if offset == 0 else "RightHand" if offset == 8 else "HMD"
                    print(f"[VRHand Debug] {target_name} skipped due to invalid data (Finite: {np.isfinite(pos_vam).all()}, Norm: {np.linalg.norm(quat_vam)})")
                continue
            
            pos_np = self._vam_pos_to_engine(pos_vam)
            quat_np = self._vam_quat_to_engine(quat_vam)
            
            # 手・HMD共通でZ軸（奥行き）のスケール補正を適用
            pos_np[2] *= 0.85  # [0]=X, [1]=Y(高さ), [2]=Z
            
            # # 💡 変換後の最終座標の確認用デバッグ
            # if self._vrhand_update_count % 30 == 0:
                # target_name = "LeftHand" if offset == 0 else "RightHand" if offset == 8 else "HMD"
                # print(f"[VRHand Debug] {target_name} -> Final Engine Pos: {pos_np}")

            pos = torch.as_tensor(pos_np, device=self._device, dtype=torch.float32).reshape(1, 3)
            quat = torch.as_tensor(quat_np, device=self._device, dtype=torch.float32)

            # 速度計算
            if obj_id in self._prev_vrhand_pos:
                prev_pos = self._prev_vrhand_pos[obj_id]
                vel = (pos - prev_pos) / dt
                speed = torch.norm(vel)
                if speed < 0.03:
                    vel = torch.zeros_like(vel)
                elif speed > 3.0:
                    vel = (vel / speed) * 3.0
            else:
                vel = torch.zeros((1, 3), device=self._device, dtype=torch.float32)

            self._prev_vrhand_pos[obj_id] = pos

            # 重力沈み込み防止補正
            compensated_pos = pos.clone()
            compensated_pos[:, 2] += 0.5 * gravity * (dt ** 2)

            self._engine.set_root_pos(env_ids, obj_id, compensated_pos)
            self._engine.set_root_rot(env_ids, obj_id, quat)
            self._engine.set_root_vel(env_ids, obj_id, vel)
            self._engine.set_root_ang_vel(env_ids, obj_id, torch.zeros((1, 3), device=self._device))

        # ドラッグ＆ドロップ物理の適用
        self._apply_vrhand_drag(env_ids)
        
        self._vrhand_update_count += 1
        return



            
    # def _update_vrhand_boxes(self):
        # if not self._vrhand_enabled or self._vrhand_socket is None:
            # return
        # if len(self._vrhand_box_ids) != 2:
            # return

        # if not hasattr(self, '_prev_vrhand_pos'):
            # self._prev_vrhand_pos = {}
            # # トリガー状態と現在掴んでいる対象を初期化
            # self._vrhand_triggers = {"left": 0.0, "right": 0.0}
            # self._vrhand_grabbed_targets = {"left": None, "right": None}

        # # 💡 パケットサイズを 64 バイト（float 16個分）に変更
        # # structのフォーマットが '14f' だった場合は '16f' に書き換えておいてください
        # EXPECTED_SIZE = 64 

        # latest_packet = None
        # while True:
            # try:
                # packet, addr = self._vrhand_socket.recvfrom(1024)
            # except BlockingIOError:
                # break
            # if len(packet) == EXPECTED_SIZE:
                # latest_packet = packet
                # self._vrhand_packet_count += 1

        # if latest_packet is not None:
            # # 16個のfloatを展開
            # self._vrhand_latest = self._vrhand_packet.unpack(latest_packet)
            
            # # 🟢 トリガー値を取得 (data[7]が左、data[15]が右)
            # self._vrhand_triggers["left"] = float(self._vrhand_latest[7])
            # self._vrhand_triggers["right"] = float(self._vrhand_latest[15])

        # if self._vrhand_latest is None:
            # return

        # env_ids = torch.tensor([0], device=self._device, dtype=torch.long)
        # dt = 1.0 / 30.0
        # gravity = 9.81

        # # ループ時のオフセットを 7 から 8（1手あたり8要素：pos(3) + quat(4) + trigger(1)）に変更
        # for hand_name, offset, obj_id in (("left", 0, self._vrhand_box_ids[0]), ("right", 8, self._vrhand_box_ids[1])):
            # values = self._vrhand_latest
            # pos_vam = np.asarray(values[offset:offset + 3], dtype=np.float64)
            # quat_vam = np.asarray(values[offset + 3:offset + 7], dtype=np.float64)
            
            # if not np.isfinite(pos_vam).all() or not np.isfinite(quat_vam).all() or np.linalg.norm(quat_vam) < 1e-8:
                # continue
            
            # pos_np = self._vam_pos_to_engine(pos_vam)
            # quat_np = self._vam_quat_to_engine(quat_vam)
            
            # pos_np[2] *= 0.85  # [0]=X, [1]=Y(高さ), [2]=Z
            
            # pos = torch.as_tensor(pos_np, device=self._device, dtype=torch.float32).reshape(1, 3)
            # quat = torch.as_tensor(quat_np, device=self._device, dtype=torch.float32)

            # # 速度計算（Dynamicオブジェクト用）
            # if obj_id in self._prev_vrhand_pos:
                # prev_pos = self._prev_vrhand_pos[obj_id]
                # vel = (pos - prev_pos) / dt
                # speed = torch.norm(vel)
                # if speed < 0.03:
                    # vel = torch.zeros_like(vel)
                # elif speed > 3.0:
                    # vel = (vel / speed) * 3.0
            # else:
                # vel = torch.zeros((1, 3), device=self._device, dtype=torch.float32)

            # self._prev_vrhand_pos[obj_id] = pos

            # # 重力沈み込み防止補正を入れて同期
            # compensated_pos = pos.clone()
            # compensated_pos[:, 2] += 0.5 * gravity * (dt ** 2)

            # self._engine.set_root_pos(env_ids, obj_id, compensated_pos)
            # self._engine.set_root_rot(env_ids, obj_id, quat)
            # self._engine.set_root_vel(env_ids, obj_id, vel)
            # self._engine.set_root_ang_vel(env_ids, obj_id, torch.zeros((1, 3), device=self._device))

        # # 🟢 ドラッグ＆ドロップ（引っ張る力）の物理適用関数を呼び出す
        # self._apply_vrhand_drag(env_ids)
        
        # self._vrhand_update_count += 1
        # return
        

        
    def _apply_vrhand_drag(self, env_ids):
        """
        VRハンドのトリガー入力に基づき、最も近いエージェントの部位に
        Warp外力バッファ経由で引っ張る力を直接与えます。
        """
        char_id = self._get_char_id()
        env_id_int = int(env_ids.item())
        
        # エージェントのボーンの位置と速度を取得
        body_pos_tensor = self._engine.get_body_pos(char_id)
        body_vel_tensor = self._engine.get_body_vel(char_id)
        
        if body_pos_tensor.ndim == 3:
            body_pos = body_pos_tensor[0]
            body_vel = body_vel_tensor[0]
        else:
            body_pos = body_pos_tensor
            body_vel = body_vel_tensor

        # 毎フレーム、外力フラグを一度 False にして、掴んでいる間だけ True にする
        self._has_body_forces = False

        # エンジンの内部外力バッファ（Warp配列、またはNumPy/Torch配列）を取得
        if hasattr(self._sim_state, '_wp_body_force'):
            if hasattr(self._sim_state._wp_body_force, "zero_"):
                self._sim_state._wp_body_force.zero_()
            elif isinstance(self._sim_state._wp_body_force, np.ndarray):
                self._sim_state._wp_body_force.fill(0.0)

        # 💡 遠距離引き寄せフラグ用の辞書が初期化されていない場合は初期化
        if not hasattr(self, "_vrhand_is_fallback"):
            self._vrhand_is_fallback = {"left": False, "right": False}

        for hand_name, hand_id in (("left", self._vrhand_box_ids[0]), ("right", self._vrhand_box_ids[1])):
            trigger_val = self._vrhand_triggers[hand_name]
            hand_pos = self._engine.get_root_pos(hand_id)
            if hand_pos.ndim > 1: hand_pos = hand_pos[0]

            # 🛑 状態A: トリガー解除
            if trigger_val < 0.3:
                self._vrhand_grabbed_targets[hand_name] = None
                self._vrhand_is_fallback[hand_name] = False  # フラグをリセット
                continue

            # 🎯 状態B: ロックオン探索
            if self._vrhand_grabbed_targets[hand_name] is None and trigger_val >= 0.5:
                closest_idx = -1
                min_dist = 999.0
                for b_idx in range(body_pos.shape[0]):
                    dist = torch.norm(body_pos[b_idx] - hand_pos).item()
                    if dist < min_dist:
                        min_dist = dist
                        closest_idx = b_idx
                
                # 1. 通常の近接ロックオン探索（0.4m未満）
                if closest_idx != -1 and min_dist < 0.4:
                    self._vrhand_grabbed_targets[hand_name] = closest_idx
                    self._vrhand_is_fallback[hand_name] = False  # 通常ロックオン
                    if self._vrhand_debug:
                        print(f"[VRhand] {hand_name} hand LOCKED onto bone index: {closest_idx}", flush=True)
                
                # 2. ロックオン対象が見つからない場合のフォールバック（遠距離引き寄せ）
                else:
                    fallback_idx = 23 if hand_name == "right" else 30
                    self._vrhand_grabbed_targets[hand_name] = fallback_idx
                    self._vrhand_is_fallback[hand_name] = True  # 💡 遠距離フラグをTrueに
                    if self._vrhand_debug:
                        print(f"[VRhand] {hand_name} hand FALLBACK LOCKED onto agent hand bone index: {fallback_idx}", flush=True)

            # 🏋️ 状態C: ターゲットをホールド中 -> 外力（Force）を計算してエージェントに与える
            target_bone_idx = self._vrhand_grabbed_targets[hand_name]
            if target_bone_idx is not None:
                bone_pos = body_pos[target_bone_idx]
                bone_vel = body_vel[target_bone_idx]
                
                # trigger_valの倍率範囲を 0.0 ～ 1.0 に変換
                clamped_trigger = max(0.3, min(1.0, trigger_val))
                scaled_trigger = (clamped_trigger - 0.3) / (1.0 - 0.3)
                
                kp_force = 250.0
                kd_force = 15.0
                
                # 元の仕様の計算式をそのまま維持
                pull_force = (kp_force * (hand_pos - bone_pos) - kd_force * bone_vel) * scaled_trigger
                
                # 💡 遠距離引き寄せの場合のみ、さらに力を弱める（例として0.3倍に設定）
                if self._vrhand_is_fallback[hand_name]:
                    pull_force = pull_force * 0.4
                
                # 安全クリッピング
                force_speed = torch.norm(pull_force)
                max_grab_force = 500.0
                if force_speed > max_grab_force:
                    pull_force = (pull_force / force_speed) * max_grab_force

                # NewtonEngine.set_body_forces を呼び出す
                if hasattr(self._engine, "set_body_forces"):
                    forces_tensor = pull_force.reshape(1, 3)
                    self._engine.set_body_forces(
                        env_ids, 
                        char_id, 
                        target_bone_idx, 
                        forces_tensor
                    )

        return


        
    # MIMIC_TO_VAM_MAP = {
        # "pelvis": "abdomen2Control",
        # "waist_yaw_link": "abdomen2Control",
        # "waist_roll_link": "abdomen2Control",
        # "torso_link": "chestControl",
        # "head_link": "headControl",
        
        # # 左腕系
        # "left_shoulder_pitch_link": "lShoulderControl",
        # "left_shoulder_roll_link": "lShoulderControl",
        # "left_shoulder_yaw_link": "lArmControl",
        # "left_elbow_link": "lElbowControl",
        # "left_wrist_roll_link": "lHandControl",
        # "left_wrist_pitch_link": "lHandControl",
        # "left_wrist_yaw_link": "lHandControl",
        
        # # 右腕系
        # "right_shoulder_pitch_link": "rShoulderControl",
        # "right_shoulder_roll_link": "rShoulderControl",
        # "right_shoulder_yaw_link": "rArmControl",
        # "right_elbow_link": "rElbowControl",
        # "right_wrist_roll_link": "rHandControl",
        # "right_wrist_pitch_link": "rHandControl",
        # "right_wrist_yaw_link": "rHandControl",
        
        # # 左脚系
        # "left_hip_pitch_link": "lThighControl",
        # "left_hip_roll_link": "lThighControl",
        # "left_hip_yaw_link": "lThighControl",
        # "left_knee_link": "lKneeControl",
        # "left_ankle_pitch_link": "lFootControl",
        # "left_ankle_roll_link": "lFootControl",
        
        # # 右脚系
        # "right_hip_pitch_link": "rThighControl",
        # "right_hip_roll_link": "rThighControl",
        # "right_hip_yaw_link": "rThighControl",
        # "right_knee_link": "rKneeControl",
        # "right_ankle_pitch_link": "rFootControl",
        # "right_ankle_roll_link": "rFootControl"
    # }
    
    # def _vam_find_body_index(self, key, body_count):
        # """
        # MimiC_Kitのkey名からVaMのインデックスを完全にハードコードで特定する
        # """
        # if key not in self.MIMIC_TO_VAM_MAP:
            # return None
        
        # # 例: "left_wrist_yaw_link" -> "lhandcontrol"
        # vam_target_name = self.MIMIC_TO_VAM_MAP[key].lower()
        
        # # 🟢 固定定義した self._vam_body_names からインデックス(0〜18)を取得
        # if vam_target_name in self._vam_body_names:
            # idx = self._vam_body_names[vam_target_name]
            # return idx
            
        # return None
        
    def _vam_to_numpy(self, value):  # 🟢 self を追加
        if torch.is_tensor(value):
            return value.detach().to("cpu").numpy()
        return np.asarray(value)


    def _vam_pos(self, pos):
        # MuJoCo [x, y, z] -> VaM [-y, z, x] （既存の成功ロジックを維持）
        p = np.asarray(pos, dtype=np.float64)
        rotated_x = -p[1]
        rotated_y = p[2]
        rotated_z = p[0]
        return np.asarray([rotated_x, rotated_y, rotated_z], dtype=np.float32) + self._vam_position_offset

    
        
    def _vam_quat_xyzw(self, quat, key=None, pelvis_quat=None):
        q = np.asarray(quat, dtype=np.float64).reshape(-1)
        if q.size != 4 or not np.all(np.isfinite(q)) or np.linalg.norm(q) < 1e-8:
            return R.from_quat([0.0, 0.0, 0.0, 1.0])
        q = q / np.linalg.norm(q)
        
        # DeepMimic [w, x, y, z] -> scipy [x, y, z, w]
        current_world_rot = R.from_quat([q[1], q[2], q[3], q[0]])

        if pelvis_quat is not None and key != "pelvis":
            qp = np.asarray(pelvis_quat, dtype=np.float64).reshape(-1)
            qp = qp / np.linalg.norm(qp)
            pelvis_world_rot = R.from_quat([qp[1], qp[2], qp[3], qp[0]])
        else:
            pelvis_world_rot = R.from_quat([0.0, 0.0, 0.0, 1.0])

        local_rot = pelvis_world_rot.inv() * current_world_rot

        # --- 2. ローカル空間のオイラー角で軸の入れ替えを行う ---
        euler = local_rot.as_euler('xyz', degrees=True)
        e_x, e_y, e_z = float(euler[0]), float(euler[1]), float(euler[2])
        
        if key in ("rfoot", "lfoot"):
            # # 入れ替え後の [e_y, -e_z, -e_x] の第3引数である「-e_x」がプラスマイナス反転して「e_x」になります
            unity_local_rot = R.from_euler('zxy', [e_y, e_z, e_x], degrees=True)
        elif key in ("relbow", "rhand"):
            unity_local_rot = R.from_euler('zxy', [e_z+90, e_y, e_x], degrees=True)
        elif key in ("lelbow","lhand"):
            unity_local_rot = R.from_euler('zxy', [e_z+90, -e_y, -e_x], degrees=True)
        else:
            # # それ以外の部位は通常通り
            unity_local_rot = R.from_euler('zxy', [e_y, e_z, -e_x], degrees=True)

        # --- 3. 左手系変換に伴うクォータニオンの反転 ---
        qx, qy, qz, qw = unity_local_rot.as_quat()
        unity_local_rot = R.from_quat([-qx, -qy, -qz, qw])

        # --- 4. Tポーズにおける「右半身」のローカル軸反転 ---
        if key in ("rhand", "relbow", "rshin", "rarm", "rthigh"):
            mqx, mqy, mqz, mqw = unity_local_rot.as_quat()
            unity_local_rot = R.from_quat([-mqx, mqy, -mqz, mqw])

        # 🟢 🟢 🟢 【下を向いたZ軸矢印を正面に起こす補正】 🟢 🟢 🟢
        # パケットが正しく送れる状態であれば、ここの角度（90度 または -90度）で
        # 完全にコントロールポイントの矢印の向きを制御できます。
        # まずは -90度（または 90度）を試して、矢印が水平に起き上がるか確認してください。
        if key in ("rfoot", "lfoot"):
            unity_local_rot = unity_local_rot * R.from_euler('x', -70, degrees=True)
        # elif key in ("relbow", "lelbow"):
            # unity_local_rot = unity_local_rot * R.from_euler('x', -90, degrees=True)
        else:
            unity_local_rot = unity_local_rot * R.from_euler('x', -90, degrees=True)
        return unity_local_rot
        
    def _send_vam_udp(self):
        if self._vam_socket is None:
            return            
        try:
            char_id = self._get_char_id()
            
            # 1. MimiC(MuJoCo)エンジンから生データの取得
            body_pos = self._vam_to_numpy(self._engine.get_body_pos(char_id))
            if body_pos.ndim == 3:
                body_pos = body_pos[0]  # 安全に次元を絞る (1, 31, 3) -> (31, 3)
            
            body_count = body_pos.shape[0]  # 31

            root_pos = self._vam_to_numpy(self._engine.get_root_pos(char_id))
            root_rot = self._vam_to_numpy(self._engine.get_root_rot(char_id))
            if root_pos.ndim > 1: root_pos = root_pos[0]
            if root_rot.ndim > 1: root_rot = root_rot[0]

            body_rot = None
            for method_name in ("get_body_rot", "get_body_rotation", "get_body_quat"):
                method = getattr(self._engine, method_name, None)
                if callable(method):
                    try:
                        body_rot = self._vam_to_numpy(method(char_id))
                        if body_rot.ndim == 3:
                            body_rot = body_rot[0]
                        break
                    except Exception:
                        body_rot = None

            # 2. XMLの定義順序に基づく、MimiC(31ボーン) -> VaMへのダイレクトインデックス対応表
            MIMIC_INDEX_MAP = {
                "pelvis": 0,
                "torso": 15,
                "head": 16,
                "rhand": 30,
                "lhand": 23,
                "rfoot": 12,
                "lfoot": 6,
                "relbow": 27,
                "lelbow": 20,
                "rshin": 10,
                "lshin": 4,
                "rarm": 26,
                "larm": 19,
                "rthigh": 9,
                "lthigh": 3,
            }

            # 骨盤（pelvis）の「変換前の生クォータニオン」を抽出
            pelvis_idx = MIMIC_INDEX_MAP["pelvis"]
            if body_rot is not None and pelvis_idx < len(body_rot):
                pelvis_q_base = body_rot[pelvis_idx]
            else:
                pelvis_q_base = root_rot

            # 骨盤のベース位置を取得
            if pelvis_idx < body_count:
                p_base = body_pos[pelvis_idx]
            else:
                p_base = root_pos
            pelvis_converted = self._vam_pos(p_base)

            # 🟢 骨盤自身の変換済み「Unityワールド回転」を真っ先に確定させる
            # (pelvis自身は基準なので pelvis_quat=None で呼び出す)
            pelvis_rot_converted = self._vam_quat_xyzw(pelvis_q_base, "pelvis", pelvis_quat=None)
            
            # 足伸ばし計算で使用する骨盤回転とその逆回転（Inverse）
            R_pelvis = pelvis_rot_converted
            R_pelvis_inv = pelvis_rot_converted.inv()

            # 3. 各ボーンの位置の基本変換、および「ローカル軸反転・マッピング」の実行とワールド復元
            raw_spatial_data = {}
            for label in MIMIC_INDEX_MAP.keys():
                idx = MIMIC_INDEX_MAP[label]
                
                # 位置の取得と Unity 空間への基本変換
                if idx < body_count:
                    p_vec = body_pos[idx]
                else:
                    p_vec = root_pos
                pos_conv = self._vam_pos(p_vec)

                # 回転の取得
                if body_rot is not None and idx < len(body_rot):
                    q_vec = body_rot[idx]
                else:
                    q_vec = root_rot
                
                # 🟢 【今回の核心】
                # すべてのボーンに「生の骨盤クォータニオン」を渡し、一度ローカルに戻して処理をさせる
                local_unity_rot = self._vam_quat_xyzw(q_vec, label, pelvis_quat=pelvis_q_base)
                
                # 🟢 最後にUnity側の骨盤ワールド回転と掛け合わせることで、破綻のない世界座標を生成
                if label == "pelvis":
                    rot_conv = local_unity_rot
                else:
                    rot_conv = pelvis_rot_converted * local_unity_rot
                
                raw_spatial_data[label] = {"pos": pos_conv.copy(), "rot": rot_conv}

            # 🟢 【STEP 1: 骨盤回転を考慮したローカル空間での足伸ばし】
            extended_positions = {}
            leg_scale = 1.32  # 足伸ばし倍率

            # 引き伸ばす前の「最も低い足（床に接している部分）」のY座標（高さ）を記録しておく
            original_lowest_y = 999.0
            for label in ("rfoot", "lfoot"):
                if label in raw_spatial_data:
                    original_lowest_y = min(original_lowest_y, float(raw_spatial_data[label]["pos"][1]))

            for label, data in raw_spatial_data.items():
                pos_conv = data["pos"]
                
                if label in ("rshin", "lshin", "rfoot", "lfoot"):
                    # 骨盤から対象ボーンへの世界座標（Unity空間）ベクトル
                    world_vec = pos_conv - pelvis_converted
                    
                    # 💡 骨盤の逆回転を適用し、ロボット自身のローカル座標ベクトルに変換
                    local_vec = R_pelvis_inv.apply(world_vec)
                    
                    # 🛠️ ロボットのローカルな「股関節から足先への方向」を引き伸ばす
                    local_vec[2] *= leg_scale

                    
                    # 💡 引き伸ばしたローカルベクトルを、現在の骨盤の回転を使って世界座標系に逆変換
                    world_vec_extended = R_pelvis.apply(local_vec)
                    
                    # 骨盤の現在位置をベースに世界座標を確定
                    extended_positions[label] = pelvis_converted + world_vec_extended
                else:
                    # 足以外のボーン（骨盤自身、上半身、頭、腕など）は、この段階では元の位置をキープ
                    extended_positions[label] = pos_conv.copy()

            # 🟢 【STEP 2: 地面基準（床に倒れた時も浮かないようにする）の高さ補正値を算出】
            extended_lowest_y = 999.0
            for label in ("rfoot", "lfoot"):
                if label in extended_positions:
                    extended_lowest_y = min(extended_lowest_y, float(extended_positions[label][1]))

            # 💡 足が伸びたことによって床を突き抜けた（あるいは倒れて位置がずれた）距離を計算
            lift_offset_y = original_lowest_y - extended_lowest_y

            # 🟢 【STEP 3: 全身のボーンに床基準のオフセットを一括適用して最終確定】
            local_offset = np.array([0.0, -0.05, 0.0])
            world_offset = R_pelvis.apply(local_offset)
            final_positions = {}
            for label in raw_spatial_data.keys():
                final_positions[label] = extended_positions[label].copy()
                if label in ("relbow", "lelbow", "rhand", "lhand","head","pelvis"):
                    final_positions[label][1] += (lift_offset_y * 0.5)
                else:
                    final_positions[label][1] += lift_offset_y
                final_positions[label] += world_offset

            # 4. 全15部位の最終変換済みデータを抽出
            pelvis_p, pelvis_q = final_positions["pelvis"], raw_spatial_data["pelvis"]["rot"]
            torso_p,  torso_q  = final_positions["torso"],  raw_spatial_data["torso"]["rot"]
            head_p,   head_q   = final_positions["head"],   raw_spatial_data["head"]["rot"]
            
            # 💡 角度を設定（例：+2度。下を向くか上を向くかはVaMのボーン仕様によります）
            # 数値をマイナス（-2.0など）にすれば逆方向に回転します
            
            
            rhand_p,  rhand_q  = final_positions["rhand"],  raw_spatial_data["rhand"]["rot"]
            lhand_p,  lhand_q  = final_positions["lhand"],  raw_spatial_data["lhand"]["rot"]
            
            rfoot_p,  rfoot_q  = final_positions["rfoot"],  raw_spatial_data["rfoot"]["rot"]
            lfoot_p,  lfoot_q  = final_positions["lfoot"],  raw_spatial_data["lfoot"]["rot"]
            
            relbow_p, relbow_q = final_positions["relbow"], raw_spatial_data["relbow"]["rot"]
            lelbow_p, lelbow_q = final_positions["lelbow"], raw_spatial_data["lelbow"]["rot"]
            
            rshin_p,  rshin_q  = final_positions["rshin"],  raw_spatial_data["rshin"]["rot"]
            lshin_p,  lshin_q  = final_positions["lshin"],  raw_spatial_data["lshin"]["rot"]
            
            rarm_p,   rarm_q   = final_positions["rarm"],   raw_spatial_data["rarm"]["rot"]
            larm_p,   larm_q   = final_positions["larm"],   raw_spatial_data["larm"]["rot"]
            
            rthigh_p, rthigh_q = final_positions["rthigh"], raw_spatial_data["rthigh"]["rot"]
            lthigh_p, lthigh_q = final_positions["lthigh"], raw_spatial_data["lthigh"]["rot"]
            
            offset_pelvis_angle_deg = -90.0 
            pelvis_local_offset = R.from_euler('x', offset_pelvis_angle_deg, degrees=True)
            pelvis_q = pelvis_q * pelvis_local_offset
            
            offset_head_angle_deg = 10.0 
            
            # 首ローカルのX軸回転クォータニオンを作成
            head_local_offset = R.from_euler('x', offset_head_angle_deg, degrees=True)
            
            # 💡 現在の首の世界回転(head_q)に対して、ローカル基準で回転を掛け合わせる（右から掛ける）
            head_q = head_q * head_local_offset

            # 5. VaMの133fパケット仕様（全19個のコントロールポイント）に完全に一致させる
            points_config = [
                ("pelvis", pelvis_p, pelvis_q),
                ("torso", torso_p, torso_q),
                ("head", head_p, head_q),
                ("rhand", rhand_p, rhand_q),
                ("lhand", lhand_p, lhand_q),
                ("rfoot", rfoot_p, rfoot_q),
                ("lfoot", lfoot_p, lfoot_q),
                ("relbow", relbow_p, relbow_q),
                ("lelbow", lelbow_p, lelbow_q),
                ("rshin", rshin_p, rshin_q),
                ("lshin", lshin_p, lshin_q),
                ("rarm", rarm_p, rarm_q),
                ("larm", larm_p, larm_q),
                ("rthigh", rthigh_p, rthigh_q),
                ("lthigh", lthigh_p, lthigh_q),
                ("rarm_dup", rarm_p, rarm_q),  
                ("larm_dup", larm_p, larm_q),  
                ("rtoe", rfoot_p, rfoot_q),    
                ("ltoe", lfoot_p, lfoot_q)     
            ]

            # 6. パケットのシリアライズと送信
            packet = []
            for idx, (label, pos_vec, rot_obj) in enumerate(points_config):
                qx, qy, qz, qw = rot_obj.as_quat()
                packet.extend([float(pos_vec[0]), float(pos_vec[1]), float(pos_vec[2]), 
                               float(qx), float(qy), float(qz), float(qw)])
            
            packet = np.asarray(packet, dtype="<f4")
            
            if packet.size == 133:
                self._vam_socket.sendto(self._vam_packet.pack(*packet.tolist()), (self._vam_ip, self._vam_port))
            else:
                print(f"[ERROR] Packet size is incorrect: {packet.size} (Expected: 133)")
                
        except Exception as e:
            print(f"UDP Send Error: {e}")

   
    
    def _update_disc_hist(self):
        char_id = self._get_char_id()
        root_pos = self._engine.get_root_pos(char_id)
        root_rot = self._engine.get_root_rot(char_id)
        root_vel = self._engine.get_root_vel(char_id)
        root_ang_vel = self._engine.get_root_ang_vel(char_id)
        dof_pos = self._engine.get_dof_pos(char_id)
        dof_vel = self._engine.get_dof_vel(char_id)
        body_pos = self._engine.get_body_pos(char_id)

        self._disc_hist_root_pos.push(root_pos)
        self._disc_hist_root_rot.push(root_rot)
        self._disc_hist_root_vel.push(root_vel)
        self._disc_hist_root_ang_vel.push(root_ang_vel)
        self._disc_hist_dof_vel.push(dof_vel)
        self._disc_hist_body_pos.push(body_pos)
        
        joint_rot = self._kin_char_model.dof_to_rot(dof_pos)
        self._disc_hist_joint_rot.push(joint_rot)
        return
    
    def _update_ref_motion(self):
        if (self._enable_ref_char()):
            super()._update_ref_motion()
        return
    
    def _update_observations(self, env_ids=None):
        super()._update_observations(env_ids)

        if (env_ids is None or len(env_ids) > 0):
            self._update_disc_obs(env_ids)
        return

    def _update_disc_obs(self, env_ids=None):
        root_pos = self._disc_hist_root_pos.get_all()
        root_rot = self._disc_hist_root_rot.get_all()
        root_vel = self._disc_hist_root_vel.get_all()
        root_ang_vel = self._disc_hist_root_ang_vel.get_all()
        joint_rot = self._disc_hist_joint_rot.get_all()
        dof_vel = self._disc_hist_dof_vel.get_all()
        body_pos = self._disc_hist_body_pos.get_all()
        
        if (env_ids is not None):
            root_pos = root_pos[env_ids]
            root_rot = root_rot[env_ids]
            root_vel = root_vel[env_ids]
            root_ang_vel = root_ang_vel[env_ids]
            joint_rot = joint_rot[env_ids]
            dof_vel = dof_vel[env_ids]
            body_pos = body_pos[env_ids]
        
        if (self._track_global_root()):
            ref_root_pos = torch.zeros_like(root_pos[..., -1, :])
            ref_root_rot = torch.zeros_like(root_rot[..., -1, :])
            ref_root_rot[..., -1] = 1
        else:
            ref_root_pos = root_pos[..., -1, :]
            ref_root_rot = root_rot[..., -1, :]
            
        if (self._has_key_bodies()):
            key_pos = body_pos[..., self._key_body_ids, :]
        else:
            key_pos = torch.zeros([0], device=self._device)

        disc_obs = compute_disc_obs(ref_root_pos=ref_root_pos,
                                  ref_root_rot=ref_root_rot,
                                  root_pos=root_pos,
                                  root_rot=root_rot, 
                                  root_vel=root_vel,
                                  root_ang_vel=root_ang_vel,
                                  joint_rot=joint_rot,
                                  dof_vel=dof_vel,
                                  key_pos=key_pos,
                                  global_obs=self._global_obs,
                                  root_height_obs=self._root_height_obs,
                                  dof_vel_obs=self._disc_dof_vel_obs)

        if (env_ids is None):
            self._disc_obs_buf[:] = disc_obs
        else:
            self._disc_obs_buf[env_ids] = disc_obs

        return



    def _update_reward(self):
        return
        

        
    def _reset_envs(self, env_ids):
        super()._reset_envs(env_ids)
        if len(env_ids) > 0:
            self._reset_disc_hist(env_ids)
            self._reset_location_task(env_ids)
        return
        


    def _reset_disc_hist(self, env_ids):
        motion_ids = self._motion_ids[env_ids]
        motion_times0 = self._get_motion_times(env_ids)
        root_pos, root_rot, root_vel, root_ang_vel, joint_rot, dof_vel, body_pos = self._fetch_disc_demo_data(motion_ids, motion_times0)
        
        self._disc_hist_root_pos.fill(env_ids, root_pos)
        self._disc_hist_root_rot.fill(env_ids, root_rot)
        self._disc_hist_root_vel.fill(env_ids, root_vel)
        self._disc_hist_root_ang_vel.fill(env_ids, root_ang_vel)
        self._disc_hist_joint_rot.fill(env_ids, joint_rot)
        self._disc_hist_dof_vel.fill(env_ids, dof_vel)
        self._disc_hist_body_pos.fill(env_ids, body_pos)
        return
        
    # def _sample_motion_times(self, n):
        # motion_ids = self._motion_lib.sample_motions(n)
        # motion_times = torch.zeros(n, dtype=torch.float, device=self._device)
        # return motion_ids, motion_times
        
    def _reset_ref_motion(self, env_ids):
        n = len(env_ids)
        motion_ids, motion_times = self._sample_motion_times(n)
        self._motion_ids[env_ids] = motion_ids
        self._motion_time_offsets[env_ids] = motion_times

        root_pos, root_rot, root_vel, root_ang_vel, joint_rot, dof_vel = self._motion_lib.calc_motion_frame(motion_ids, motion_times)

        # 1. まずは既存のロジックで全対象環境にデフォルト（通常リセット）の値をセット
        self._ref_root_pos[env_ids] = root_pos
        self._ref_root_rot[env_ids] = root_rot
        self._ref_root_vel[env_ids] = root_vel
        self._ref_root_ang_vel[env_ids] = root_ang_vel
        self._ref_joint_rot[env_ids] = joint_rot
        self._ref_dof_vel[env_ids] = dof_vel
        
        # 2. 最初は全環境を通常のFKで計算（これで全環境のベースラインが保証される）
        ref_body_pos, ref_body_rot = self._kin_char_model.forward_kinematics(self._ref_root_pos, self._ref_root_rot,
                                                                             self._ref_joint_rot)
        self._ref_body_pos[:] = ref_body_pos
        self._ref_body_rot[:] = ref_body_rot

        dof_pos = self._motion_lib.joint_rot_to_dof(joint_rot)
        self._ref_dof_pos[env_ids] = dof_pos

        # --- 💥 ここから：条件付き引き継ぎの割り込み処理 💥 ---
        char_id = self._get_char_id()
        
        # エンジンから全環境の生位置を取得 [num_envs, 3]
        actual_root_pos_np = self._vam_to_numpy(self._engine.get_root_pos(char_id))
        actual_root_pos_torch = torch.from_numpy(actual_root_pos_np).to(device=self._device, dtype=torch.float)
        
        # 💡 【選別ロジック】引き継ぎを適用したい「有効な環境（inherit_ids）」を絞り込む
        inherit_ids = []
        for idx in env_ids:
            idx_item = idx.item() if hasattr(idx, 'item') else idx
            
            # ① 初回判定: progress_buf が 0 の場合は初回なのでスキップ
            is_initial = False
            if hasattr(self, 'progress_buf'):
                if self.progress_buf[idx_item] <= 0:
                    is_initial = True
                    
            # ② 転倒判定: キャラクターの現在の高さ（Z座標）が 0.5m 未満なら転倒とみなしてスキップ
            # (※タスク内に self._is_fall_down() や self.die_buf があればそちらに差し替えてもOKです)
            current_z = actual_root_pos_torch[idx_item, 2].item()
            is_fallen = (current_z < 0.5) 
            
            # 初回でも転倒でもない場合のみ、姿勢引き継ぎの対象にする
            if not is_initial and not is_fallen:
                inherit_ids.append(idx_item)
                

        if len(inherit_ids) > 0:
            inherit_ids = torch.tensor(inherit_ids, dtype=torch.long, device=self._device)
            
            # 該当環境（inherit_ids）の位置・回転を取得して上書き
            actual_root_rot_np = self._vam_to_numpy(self._engine.get_root_rot(char_id))
            actual_root_rot_torch = torch.from_numpy(actual_root_rot_np).to(device=self._device, dtype=torch.float)
            
            self._ref_root_pos[inherit_ids] = actual_root_pos_torch[inherit_ids]
            self._ref_root_rot[inherit_ids] = actual_root_rot_torch[inherit_ids]

            # # ルート速度・角速度の同期
            # actual_root_vel_np = None
            # for method_name in ("get_root_vel", "get_root_velocity", "get_root_linear_velocity"):
                # method = getattr(self._engine, method_name, None)
                # if callable(method):
                    # try: actual_root_vel_np = self._vam_to_numpy(method(char_id)); break
                    # except Exception: pass
            # if actual_root_vel_np is not None:
                # actual_root_vel_torch = torch.from_numpy(actual_root_vel_np).to(device=self._device, dtype=torch.float)
                # self._ref_root_vel[inherit_ids] = actual_root_vel_torch[inherit_ids]


        
        return
        

import envs.task_location_env as task_location_env

#3. run.py が「amp_env.AMPEnv」を呼び出すため、名前をタスク用に差し替える
AMPEnv = task_location_env.TaskLocationEnv
    
    
@torch.jit.script
def compute_disc_vel_obs(ref_root_rot, root_vel, root_ang_vel, dof_vel, global_obs, dof_vel_obs):
    # type: (Tensor, Tensor, Tensor, Tensor, bool, bool) -> Tensor

    if (not global_obs):
        heading_inv_rot = torch_util.calc_heading_quat_inv(ref_root_rot)
        heading_inv_rot_expand = heading_inv_rot.unsqueeze(-2)
        heading_inv_rot_expand = heading_inv_rot_expand.repeat((1, root_vel.shape[1], 1))
        heading_inv_rot_flat = heading_inv_rot_expand.reshape((heading_inv_rot_expand.shape[0] * heading_inv_rot_expand.shape[1], 
                                                               heading_inv_rot_expand.shape[2]))
        
        root_vel_obs_flat = torch.reshape(root_vel, [root_vel.shape[0] * root_vel.shape[1], root_vel.shape[2]])
        root_vel_obs_flat = torch_util.quat_rotate(heading_inv_rot_flat, root_vel_obs_flat)
        root_vel_obs = torch.reshape(root_vel_obs_flat, root_vel.shape)
        
        root_ang_vel_obs_flat = torch.reshape(root_ang_vel, [root_ang_vel.shape[0] * root_ang_vel.shape[1], root_ang_vel.shape[2]])
        root_ang_vel_obs_flat = torch_util.quat_rotate(heading_inv_rot_flat, root_ang_vel_obs_flat)
        root_ang_vel_obs = torch.reshape(root_ang_vel_obs_flat, root_ang_vel.shape)
    else:
        root_vel_obs = root_vel
        root_ang_vel_obs = root_ang_vel

    obs = [root_vel_obs, root_ang_vel_obs]

    if (dof_vel_obs):
        obs = obs + [dof_vel]

    obs = torch.cat(obs, dim=-1)

    return obs

@torch.jit.script
def compute_disc_obs(ref_root_pos, ref_root_rot, root_pos, root_rot, root_vel, root_ang_vel, joint_rot, dof_vel, key_pos, global_obs, root_height_obs, dof_vel_obs):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, bool, bool, bool) -> Tensor

    pos_obs = deepmimic_env.compute_tar_obs(ref_root_pos=ref_root_pos, 
                                               ref_root_rot=ref_root_rot, 
                                               root_pos=root_pos, 
                                               root_rot=root_rot, 
                                               joint_rot=joint_rot, 
                                               key_pos=key_pos,
                                               global_obs=global_obs, 
                                               root_height_obs=root_height_obs)

    vel_obs = compute_disc_vel_obs(ref_root_rot=ref_root_rot, 
                                  root_vel=root_vel, 
                                  root_ang_vel=root_ang_vel, 
                                  dof_vel=dof_vel,
                                  global_obs=global_obs,
                                  dof_vel_obs=dof_vel_obs)

    disc_obs = torch.cat([pos_obs, vel_obs], dim=-1)
    disc_obs = torch.reshape(disc_obs, [disc_obs.shape[0], -1])

    return disc_obs
    
    

