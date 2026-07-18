"""PolaRiS client for the remote EgoVerse HPT inference server."""

from __future__ import annotations

import pickle

import numpy as np
import torch
import zmq
from scipy.spatial.transform import Rotation

from polaris.client.abstract_client import InferenceClient
from polaris.config import PolicyArgs


@InferenceClient.register(client_name="EgoVerse")
class EgoVerseClient(InferenceClient):
    """Send the trained third-person+wrist views and execute Cartesian chunks."""

    def __init__(self, args: PolicyArgs) -> None:
        self.args = args
        self.open_loop_horizon = int(args.open_loop_horizon or 1)
        if self.open_loop_horizon < 1:
            raise ValueError("open_loop_horizon must be >= 1")

        self.camera_keys = {
            # The training dataset's Azure Kinect left view matches the
            # simulator's cam1 rendering (verified from crop previews).
            "front_img_1": getattr(args, "cam1_key", "cam1"),
            "wrist_img": getattr(args, "wrist_cam_key", "wrist_cam"),
        }
        context = zmq.Context.instance()
        self.socket = context.socket(zmq.REQ)
        self.socket.connect(f"tcp://{args.host}:{args.port}")
        print(
            f"[EgoVerseClient] connected to {args.host}:{args.port}; "
            f"cameras={self.camera_keys}; open_loop={self.open_loop_horizon}"
        )

        from polaris.utils_.planner_utils import setup_curobo_ik

        self.ik_solver = setup_curobo_ik()
        self.action_chunk: np.ndarray | None = None
        self.chunk_index = 0
        self.last_viz: np.ndarray | None = None

    @property
    def rerender(self) -> bool:
        return self.action_chunk is None or self.chunk_index >= len(self.action_chunk)

    def reset(self):
        self.action_chunk = None
        self.chunk_index = 0
        self.last_viz = None
        self.socket.send(pickle.dumps({"reset": True}))
        response = pickle.loads(self.socket.recv())
        if "error" in response:
            raise RuntimeError(response["error"])

    def _request_chunk(self, obs: dict) -> None:
        splat = obs["splat"]
        missing = [source for source in self.camera_keys.values() if source not in splat]
        if missing:
            raise KeyError(f"Missing simulator camera(s) {missing}; available={list(splat)}")

        ee_pose = obs["policy"]["ee_pose"][0]
        if isinstance(ee_pose, torch.Tensor):
            ee_pose = ee_pose.detach().cpu().numpy()
        ee_pose = np.asarray(ee_pose, dtype=np.float32)
        if ee_pose.shape != (8,):
            raise ValueError(f"Expected ee_pose pos|quat_wxyz|gripper shape (8,), got {ee_pose.shape}")
        quat_xyzw = ee_pose[[4, 5, 6, 3]]
        ypr = Rotation.from_quat(quat_xyzw).as_euler("ZYX").astype(np.float32)
        state_ee_pose = np.concatenate([ee_pose[:3], ypr, ee_pose[7:8]])

        request = {
            "images": {
                target: np.ascontiguousarray(splat[source])
                for target, source in self.camera_keys.items()
            },
            "state_ee_pose": state_ee_pose,
            "open_loop_horizon": self.open_loop_horizon,
        }
        self.socket.send(pickle.dumps(request, protocol=pickle.HIGHEST_PROTOCOL))
        response = pickle.loads(self.socket.recv())
        if "error" in response:
            raise RuntimeError(f"EgoVerse server: {response['error']}")
        chunk = np.asarray(response["action_chunk"], dtype=np.float32)
        if chunk.ndim != 2 or chunk.shape[1] != 7 or len(chunk) == 0:
            raise ValueError(f"Expected non-empty EgoVerse action chunk (T, 7), got {chunk.shape}")
        self.action_chunk = chunk
        self.chunk_index = 0
        if "viz_bytes" in response:
            self.last_viz = np.frombuffer(
                response["viz_bytes"], dtype=np.dtype(response["viz_dtype"])
            ).reshape(response["viz_shape"])
        else:
            # Backward compatibility with servers using the original response.
            self.last_viz = response.get("viz")

    def _to_joint_action(self, action: np.ndarray, obs: dict) -> np.ndarray:
        from curobo.types.math import Pose

        position = torch.from_numpy(action[:3]).float().view(1, 3).to(self.args.device)
        quat_xyzw = Rotation.from_euler("ZYX", action[3:6]).as_quat()
        quat_wxyz = np.asarray(quat_xyzw[[3, 0, 1, 2]], dtype=np.float32)
        quaternion = torch.from_numpy(quat_wxyz).view(1, 4).to(self.args.device)
        current_joints = (
            obs["policy"]["arm_joint_pos"][0]
            .detach()
            .to(device=self.args.device, dtype=torch.float32)
            .view(1, 7)
        )
        # CuRobo's Franka+Robotiq configuration has 13 c-space entries: the
        # seven Panda joints followed by six gripper/mimic joints. Build a
        # full-DOF seed in its declared order and retain configured values for
        # joints that do not affect the grasp-frame IK target.
        current_cspace = self.ik_solver.get_retract_config().detach().clone().view(1, -1)
        current_cspace[:, :7] = current_joints
        num_seeds = self.ik_solver.num_seeds
        local_seeds = current_cspace.unsqueeze(1).repeat(1, num_seeds, 1)
        if num_seeds > 1:
            # Keep every seed in the current IK branch. The first seed is the
            # exact current state; the rest provide small local alternatives.
            generator = torch.Generator(device=self.args.device).manual_seed(0)
            perturbation = torch.randn(
                (1, num_seeds - 1, 7),
                device=self.args.device,
                dtype=torch.float32,
                generator=generator,
            ) * 0.025
            local_seeds[:, 1:, :7] += perturbation
        # Franka has multiple IK branches. Seed and regularize at the current
        # configuration, then choose the closest successful solution instead
        # of allowing independent random-seed branch changes each timestep.
        result = self.ik_solver.solve_single(
            Pose(position=position, quaternion=quaternion),
            retract_config=current_cspace,
            seed_config=local_seeds,
            return_seeds=num_seeds,
        )
        success = result.success.reshape(-1)
        solutions = result.solution.reshape(-1, result.solution.shape[-1])[:, :7]
        if bool(success.any()):
            valid = solutions[success]
            distance = torch.linalg.vector_norm(valid[:, :7] - current_joints, dim=-1)
            closest_index = torch.argmin(distance)
            closest = valid[closest_index, :7]
            joints = closest.detach().cpu().numpy()
        else:
            print("[EgoVerseClient] IK failed; holding current arm joints")
            joints = current_joints[0].detach().cpu().numpy()
        gripper = np.float32(1.0 if action[6] >= 0.2 else 0.0)
        return np.concatenate([joints, [gripper]]).astype(np.float32)

    def infer(
        self, obs: dict, instruction: str, return_viz: bool = False
    ) -> tuple[np.ndarray, np.ndarray | None]:
        del instruction  # This checkpoint is not language-conditioned.
        if self.action_chunk is None or self.chunk_index >= len(self.action_chunk):
            self._request_chunk(obs)
        action = self.action_chunk[self.chunk_index]
        self.chunk_index += 1
        env_action = self._to_joint_action(action, obs)
        return env_action, self.last_viz if return_viz else None
