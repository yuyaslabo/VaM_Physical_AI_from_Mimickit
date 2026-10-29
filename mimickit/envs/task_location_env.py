# Original code authored by Xue Bin Peng (MimicKit)
# Modified by [LocalAIVRLab] in 2026 for integration with Virt-A-Mate (VaM).
# Licensed under the Apache License, Version 2.0.
import numpy as np
import torch
import engines.engine as engine
#import envs.smp_env as smp_env
import envs.amp_env as amp_env
import util.torch_util as torch_util
from util.logger import Logger
from engines.newton_engine import SimState

class TaskLocationEnv(amp_env.AMPEnv):
    def __init__(self, env_config, engine_config, num_envs, device, visualize, record_video=False):
        self._tar_speed = env_config["tar_speed"]
        self._tar_change_time_min = env_config["tar_change_time_min"]
        self._tar_change_time_max = env_config["tar_change_time_max"]
        self._tar_dist_min = env_config.get("tar_dist_min", 0.0)
        self._tar_dist_max = env_config["tar_dist_max"]
        assert(self._tar_dist_max >= self._tar_dist_min)

        self._dist_threshold = env_config.get("dist_threshold", 0.5)

        self._pos_err_scale = env_config.get("pos_err_scale", 0.5)
        self._vel_err_scale = env_config.get("vel_err_scale", 4.0)
        self._pos_reward_w = env_config.get("pos_reward_w", 0.5)
        self._vel_reward_w = env_config.get("vel_reward_w", 0.4)
        self._face_reward_w = env_config.get("face_reward_w", 0.1)
        self._vrhand_position_offset = np.array([0.0, 0.18, 0.0], dtype=np.float32)
        self._vrhand_triggers = {"left": 0.0, "right": 0.0}
        self._vrhand_grabbed_targets = {"left": None, "right": None}
        
        

        super().__init__(env_config=env_config, engine_config=engine_config,
                         num_envs=num_envs, device=device, visualize=visualize,
                         record_video=record_video)
        if hasattr(self, "_engine") and hasattr(self._engine, "_sim_state"):
            self._sim_state = self._engine._sim_state
        # if hasattr(self, "_engine") and hasattr(self._engine, "_sim_model"):
            # self._sim_state = SimState(self._engine._sim_model, num_envs)
        # else:
            # # 万が一、engine側にもない場合は親クラスがすでに _sim_state を持っていないか確認
            # if hasattr(self, "_engine") and hasattr(self._engine, "_sim_state"):
                # self._sim_state = self._engine._sim_state
        # if hasattr(self._sim_state, '_wp_body_force') and self._sim_state._wp_body_force is not None:
            # if hasattr(self._sim_state._wp_body_force, "zero_"):
                # self._sim_state._wp_body_force.zero_()
        return
    
    def _build_envs(self, config, num_envs):
        if (self._visualize):
            self._marker_ids = []

        super()._build_envs(config, num_envs)
        return
    
    def _build_env(self, env_id, config):
        super()._build_env(env_id, config)
        
        if (self._visualize):
            marker_id = self._build_marker(env_id)

            if (env_id == 0):
                self._marker_ids.append(marker_id)
            else:
                marker_id0 = self._marker_ids[0]
                assert(marker_id0 == marker_id)
        return

    def _build_marker(self, env_id):
        asset_file = "data/assets/objects/location_marker.xml"

        marker_id = self._engine.create_obj(env_id=env_id, 
                                            obj_type=engine.ObjType.rigid,
                                            asset_file=asset_file, 
                                            name="marker",
                                            is_visual=True,
                                            fix_root=True,
                                            color=[0.5, 0.5, 0.5])
        
        return marker_id

    def _build_sim_tensors(self, config):
        super()._build_sim_tensors(config)
        
        num_envs = self.get_num_envs()
        self._tar_pos = torch.zeros([num_envs, 3], device=self._device, dtype=torch.float)
        self._tar_change_times = torch.zeros([num_envs], device=self._device, dtype=torch.float)
        self._prev_root_pos = torch.zeros([num_envs, 3], device=self._device, dtype=torch.float)
        return
    
    def _get_marker_id(self):
        return self._marker_ids[0]

    def _pre_physics_step(self, actions):
        super()._pre_physics_step(actions)
        self._record_char_root()
        return

    def _record_char_root(self):
        char_id = self._get_char_id()
        root_pos = self._engine.get_root_pos(char_id)
        self._prev_root_pos[:] = root_pos
        return

    def _update_marker(self, env_ids):
        marker_id = self._get_marker_id()
        rot = torch.tensor([0, 0, 0, 1], device=self._device, dtype=torch.float32)

        self._engine.set_root_pos(env_ids, marker_id, self._tar_pos[env_ids])
        self._engine.set_root_rot(env_ids, marker_id, rot)
        self._engine.set_root_vel(env_ids, marker_id, 0.0)
        self._engine.set_root_ang_vel(env_ids, marker_id, 0.0)
        return
    
    def _update_task(self):
        super()._update_task()
        reset_task_mask = self._time_buf >= self._tar_change_times
        rest_env_ids = reset_task_mask.nonzero(as_tuple=False).flatten()

        if len(rest_env_ids) > 0:
            self._reset_task(rest_env_ids)
        return

    def _reset_envs(self, env_ids):
        super()._reset_envs(env_ids)

        if (len(env_ids) > 0):
            self._reset_task(env_ids)
        return

    def _reset_task(self, env_ids):
        self._reset_tar(env_ids)

        rand_dt = torch.rand(env_ids.shape[0], device=self._device, dtype=torch.float)
        rand_dt = (self._tar_change_time_max - self._tar_change_time_min) * rand_dt + self._tar_change_time_min
        self._tar_change_times[env_ids] = self._time_buf[env_ids] + rand_dt
        
        if (self._visualize):
            self._update_marker(env_ids)
        return
        
    def _reset_tar(self, env_ids):


        # 1. HMDの位置とクォータニオンを取得 (1次元テンソルとして扱う)
        hmd_pos = self._latest_hmd_pos.squeeze()   # [x, y, z]
        hmd_quat = self._latest_hmd_quat.squeeze() # [x, y, z, w] -> 現在 [0, 0, z, w] の状態

        # 2. クォータニオンから水平な前方ベクトル（XY平面）を計算
        z, w = hmd_quat[2], hmd_quat[3]

        # 💡 [重要修正] [0, 0, z, w] のクォータニオンから正しく水平前方を取り出す数式に変更
        # 通常の回転行列から、Z軸回転のみが残った際の前方ベクトルを計算します
        # ※ 環境の基準方向（前）がX軸かY軸かによって符号や入れ替えが必要になる場合があります
        forward_x = 1.0 - 2.0 * (z ** 2)
        forward_y = 2.0 * w * z

        forward_dir_2d = torch.stack([forward_x, forward_y]) # [x, y]
        
        # 水平ベクトルの長さを計算して正規化
        norm = torch.norm(forward_dir_2d)
        if norm > 1e-6:
            forward_dir_2d = forward_dir_2d / norm

        # 3. HMDの水平位置（XY）から、その前方へ 50cm (0.5m) 進めた位置を計算
        distance = 0.5  # 50センチ
        target_pos_2d = hmd_pos[0:2] + forward_dir_2d * distance

        # 4. 指定された env_ids の XY座標（0:2）に代入
        self._tar_pos[env_ids, 0:2] = target_pos_2d
        return
    # def _reset_tar(self, env_ids):
        # char_id = self._get_char_id()
        # root_pos = self._engine.get_root_pos(char_id)

        # char_root_pos = root_pos[env_ids, 0:2]

        # rand_dist = (self._tar_dist_max - self._tar_dist_min) * torch.rand_like(char_root_pos[..., 0]) + self._tar_dist_min
        # rand_theta = 2 * np.pi * torch.rand_like(char_root_pos[..., 0])
        # #rand_pos = char_root_pos.clone()
        # rand_pos = torch.zeros_like(char_root_pos)
        # rand_pos[..., 0] += rand_dist * torch.cos(rand_theta)
        # rand_pos[..., 1] += rand_dist * torch.sin(rand_theta)

        # self._tar_pos[env_ids, 0:2] = rand_pos
        # return

    def _compute_obs(self, env_ids=None):
        obs = super()._compute_obs(env_ids)
        
        char_id = self._get_char_id()
        char_root_pos = self._engine.get_root_pos(char_id)
        char_root_rot = self._engine.get_root_rot(char_id)
        tar_pos = self._tar_pos

        if (env_ids is not None):
            char_root_pos = char_root_pos[env_ids]
            char_root_rot = char_root_rot[env_ids]
            tar_pos = tar_pos[env_ids]
        
        task_obs = compute_location_observations(char_root_pos, char_root_rot, tar_pos)
        obs = torch.cat([obs, task_obs], dim=-1)

        return obs
    
    def _update_reward(self):
        char_id = self._get_char_id()
        char_root_pos = self._engine.get_root_pos(char_id)
        char_root_rot = self._engine.get_root_rot(char_id)
        self._reward_buf[:] = compute_location_reward(root_pos=char_root_pos,
                                                      prev_root_pos=self._prev_root_pos,
                                                      root_rot=char_root_rot,
                                                      tar_pos=self._tar_pos, 
                                                      tar_speed=self._tar_speed,
                                                      dt=self._engine.get_timestep(),
                                                      dist_threshold=self._dist_threshold,
                                                      pos_err_scale=self._pos_err_scale,
                                                      vel_err_scale=self._vel_err_scale,
                                                      pos_reward_w=self._pos_reward_w,
                                                      vel_reward_w=self._vel_reward_w,
                                                      face_reward_w=self._face_reward_w
                                                      )
        #Logger.print(f"reward:{self._reward_buf[0].item():.4f}",end='\r')
        return

    def _update_misc(self):
        super()._update_misc()

        self._update_task()
        return

    def _render_scene(self):
        super()._render_scene()
        self._render_location_lines()
        return

    def _render_location_lines(self):
        line_width = 2.0
        col = np.array([[0.5, 0.5, 0.5, 0.5]], dtype=np.float32)
        
        char_id = self._get_char_id()
        char_root_pos = self._engine.get_root_pos(char_id)

        starts = char_root_pos.cpu().numpy()
        ends = self._tar_pos.cpu().numpy()
        
        num_envs = self.get_num_envs()
        for i in range(num_envs):
            curr_start = starts[i:i + 1]
            curr_end = ends[i:i + 1]
            self._engine.draw_lines(i, curr_start, curr_end, col, line_width)

        return
        
    def _vam_pos_to_engine(self, pos_vam):
        p = np.asarray(pos_vam, dtype=np.float32) - self._vrhand_position_offset
        return np.array([p[2], -p[0], p[1]], dtype=np.float32)
        


    def _vam_quat_to_engine(self, quat_vam_xyzw):
        qx, qy, qz, qw = np.asarray(quat_vam_xyzw, dtype=np.float64)
        q = np.asarray([-qx, -qz, -qy, qw], dtype=np.float64)
        norm = np.linalg.norm(q)
        if not np.isfinite(q).all() or norm < 1e-8:
            return np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
        return (q / norm).astype(np.float32)
    

#####################################################################
###=========================jit functions=========================###
#####################################################################

@torch.jit.script
def compute_location_observations(root_pos, root_rot, tar_pos):
    # type: (Tensor, Tensor, Tensor) -> Tensor
    heading_rot = torch_util.calc_heading_quat_inv(root_rot)
    local_tar_pos = torch_util.quat_rotate(heading_rot, tar_pos - root_pos)
    obs = local_tar_pos[..., 0:2]
    return obs

@torch.jit.script
def compute_location_reward(root_pos, prev_root_pos, root_rot, tar_pos, tar_speed, dt, dist_threshold, pos_err_scale, vel_err_scale, pos_reward_w, vel_reward_w, face_reward_w):
    # type: (Tensor, Tensor, Tensor, Tensor, float, float, float, float, float, float, float, float) -> Tensor
    
    pos_diff = tar_pos - root_pos
    pos_err = pos_diff[..., 0] * pos_diff[..., 0] + pos_diff[..., 1] * pos_diff[..., 1]
    pos_reward = torch.exp(-pos_err_scale * pos_err)

    tar_dir = tar_pos - root_pos
    tar_dir = tar_dir[..., 0:2]
    tar_dir = torch.nn.functional.normalize(tar_dir, dim=-1)
    
    delta_root_pos = root_pos - prev_root_pos
    root_vel = delta_root_pos / dt
    tar_dir_speed = torch.sum(tar_dir * root_vel[..., :2], dim=-1)
    tar_vel_err = tar_speed - tar_dir_speed
    tar_vel_err = torch.clamp_min(tar_vel_err, 0.0)
    vel_reward = torch.exp(-vel_err_scale * (tar_vel_err * tar_vel_err))

    wrong_dir = tar_dir_speed <= 0
    vel_reward[wrong_dir] = 0

    heading_rot = torch_util.calc_heading_quat(root_rot)
    facing_dir = torch.zeros_like(root_pos)
    facing_dir[..., 0] = 1.0
    facing_dir = torch_util.quat_rotate(heading_rot, facing_dir)
    facing_err = torch.sum(tar_dir * facing_dir[..., 0:2], dim=-1)
    facing_reward = torch.clamp_min(facing_err, 0.0)

    dist_mask = pos_err < dist_threshold
    facing_reward[dist_mask] = 1.0
    vel_reward[dist_mask] = 1.0

    reward = pos_reward_w * pos_reward + vel_reward_w * vel_reward + face_reward_w * facing_reward

    return reward