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
    """Send three RGB views + EEF state and execute returned Cartesian chunks."""

    def __init__(self, args: PolicyArgs) -> None:
        self.args = args
        self.open_loop_horizon = int(args.open_loop_horizon or 1)
        if self.open_loop_horizon < 1:
            raise ValueError("open_loop_horizon must be >= 1")

        self.camera_keys = {
            "front_img_1": getattr(args, "cam0_key", "cam0"),
            "front_img_2": getattr(args, "cam1_key", "cam1"),
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
        result = self.ik_solver.solve_single(Pose(position=position, quaternion=quaternion))
        if result.success.item():
            joints = result.solution.squeeze(0)[0, :7].detach().cpu().numpy()
        else:
            print("[EgoVerseClient] IK failed; holding current arm joints")
            joints = obs["policy"]["arm_joint_pos"][0].detach().cpu().numpy()
        gripper = np.float32(1.0 if action[6] >= 0.5 else 0.0)
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
