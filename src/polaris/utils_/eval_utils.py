import numpy as np
from scipy.spatial.transform import Rotation
import torch
import random
from dataclasses import dataclass

def set_seed(seed: int):
    """Set seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Optional: for deterministic behavior (may slow down)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False

def _sample_from_zones(zones: list) -> float:
    """Randomly pick one zone from a list of [min, max] pairs, then sample within it."""
    zone = zones[np.random.randint(len(zones))]   # pick a zone uniformly
    return np.random.uniform(zone[0], zone[1])     # sample within that zone


def randomize_object_poses(randomization_setting: dict, initial_conditions: list, min_distance: float = 0.05, max_retries: int = 100) -> list:
    result = dict(initial_conditions[0])
    placed_positions = []  # Track (x, y, z) of newly placed objects only

    for obj, ranges in randomization_setting.items():
        for attempt in range(max_retries):
            x = _sample_from_zones(ranges["x"])
            y = _sample_from_zones(ranges["y"])
            z = _sample_from_zones(ranges["z"])

            # Check distance against all previously placed objects
            too_close = any(
                np.linalg.norm(np.array([x, y, z]) - np.array(pos)) < min_distance
                for pos in placed_positions
            )

            if not too_close:
                break
        else:
            raise RuntimeError(
                f"Could not place '{obj}' at least {min_distance*100:.0f}cm away from "
                f"other objects after {max_retries} attempts. "
                f"Consider expanding the randomization zones."
            )

        placed_positions.append([x, y, z])

        # ori ranges are still flat [min, max] — keep as-is
        rx = np.random.uniform(*ranges.get("ori_x", [0, 0]))
        ry = np.random.uniform(*ranges.get("ori_y", [0, 0]))
        rz = np.random.uniform(*ranges.get("ori_z", [0, 0]))

        ori_pose = initial_conditions[0][obj]       # [x, y, z, qx, qy, qz, qw]
        orig_q   = Rotation.from_quat(ori_pose[3:7])
        rel_q    = Rotation.from_euler("xyz", [rx, ry, rz])
        final_q  = orig_q * rel_q
        q        = final_q.as_quat()                # xyzw

        result[obj] = [x, y, z, q[3], q[0], q[1], q[2]]  # wxyz for Isaac Sim

        print(f"[{obj}] pos=({x:.3f}, {y:.3f}, {z:.3f})  quat(wxyz)=({q[3]:.3f}, {q[0]:.3f}, {q[1]:.3f}, {q[2]:.3f})")

    return [result]

def is_success(info) -> bool:
    try:
        val = info["rubric"]["success"]
        v = val[0] if hasattr(val, "__len__") else val
        if isinstance(v, torch.Tensor):
            return bool(v.item())
        return bool(v)
    except (KeyError, TypeError, IndexError) as e:
        print(f"  [warn] Could not read info['rubric']['success']: {e}")
        return False


@dataclass
class PickMetricsTracker:
    """Track grasp attempts and classify the first successful mug pick.

    The mug-local anchors match the red-cup task convention documented in
    task_config.yaml: +x is the handle and +/-y are the two rim sides.
    """

    object_name: str = "red_cup"
    lift_threshold: float = 0.04
    handle_offset: float = 0.055
    side_offset: float = 0.05
    close_threshold: float = 0.5

    def reset(self, env) -> None:
        self.initial_height = self._object_pose(env)[0][2]
        self.previous_closed = False
        self.attempts = 0
        self.active_attempt = 0
        self.active_pick_type = "none"
        self.successful_attempt = 0
        self.pick_type = "none"

    def _object_pose(self, env) -> tuple[np.ndarray, np.ndarray]:
        state = env.scene[self.object_name].data.root_state_w[0]
        state = state.detach().cpu().numpy()
        return state[:3], state[3:7]  # position, quaternion wxyz

    def _classify_pick(self, env) -> str:
        object_pos, object_quat = self._object_pose(env)
        # FrameTransformer target positions may be (N, T, 3), whereas other
        # pose sources use (N, 3). Flatten the singleton target dimension so
        # the position is always a plain xyz vector.
        ee_pos = (
            env.scene["ee_frame"]
            .data.target_pos_w[0]
            .detach()
            .cpu()
            .numpy()
            .reshape(-1, 3)[0]
        )
        rotation = Rotation.from_quat(object_quat[[1, 2, 3, 0]])
        local_xy = rotation.inv().apply(ee_pos - object_pos)[:2]
        anchors = {
            "through_handle": np.array([self.handle_offset, 0.0]),
            "left_side": np.array([0.0, -self.side_offset]),
            "right_side": np.array([0.0, self.side_offset]),
        }
        return min(anchors, key=lambda name: np.linalg.norm(local_xy - anchors[name]))

    def update(self, env, action) -> None:
        command = float(torch.as_tensor(action).reshape(-1)[-1].detach().cpu().item())
        closed = command >= self.close_threshold

        if closed and not self.previous_closed:
            self.attempts += 1
            self.active_attempt = self.attempts
            self.active_pick_type = self._classify_pick(env)

        object_height = self._object_pose(env)[0][2]
        lifted = object_height - self.initial_height > self.lift_threshold
        if closed and lifted and self.successful_attempt == 0:
            self.successful_attempt = self.active_attempt
            self.pick_type = self.active_pick_type

        self.previous_closed = closed

    def metrics(self) -> dict[str, object]:
        picked = self.successful_attempt > 0
        return {
            "grasp_attempts": self.attempts,
            "grasp_succeeded": picked,
            "grasp_succeeded_first_attempt": self.successful_attempt == 1,
            "successful_grasp_attempt": self.successful_attempt,
            "pick_type": self.pick_type,
            "pick_left_side": self.pick_type == "left_side",
            "pick_right_side": self.pick_type == "right_side",
            "pick_through_handle": self.pick_type == "through_handle",
        }
