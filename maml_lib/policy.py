"""Single-step policy MLP (T=1) for MAML.

Replaces the position+velocity PIDs of the firmware. Queried at NN_FREQ.

Input  (B, 12 + 3) : state + 3D relative target (target - current_pos)
Output (B,      4) : [thrust_u16, roll_deg, pitch_deg, yaw_rate_deg/s]

Hover initialisation uses the *baseline* drone mass; MAML adaptation is
expected to fix the residual offset for heavier or off-centred tasks.
"""
import numpy as np
import torch
import torch.nn as nn

from . import config as C


class PolicyMLP(nn.Module):
    """T=1 MLP controller, drop-in replacement for ConcurrentPolicyMLP."""

    def __init__(self, hidden: int = 64):
        super().__init__()
        self.register_buffer("x_scale",
                             torch.tensor(C.X_SCALE, dtype=torch.float32))
        self.register_buffer("pos_scale",
                             torch.tensor(C.POS_SCALE, dtype=torch.float32))

        input_dim  = 12 + 3 # 12D state + 3D position target
        output_dim = 4 # thrust roll pitch yawrate
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, output_dim),
        )
        self._init_hover()

    def _init_hover(self):
        last = self.net[-1]
        nn.init.normal_(last.weight, std=0.01)
        nn.init.zeros_(last.bias)
        # bias on thrust so sigmoid(b) * UINT16_MAX = baseline hover thrust
        hover_ratio = C.HOVER_THRUST_U16_BASE / C.UINT16_MAX
        thrust_bias = float(np.log(hover_ratio / (1.0 - hover_ratio)))
        last.bias.data[0] = thrust_bias
        # roll, pitch, yaw_rate biases stay at 0 -> tanh(0) * max = 0

    def forward(self, state: torch.Tensor,
                target_rel: torch.Tensor) -> torch.Tensor:
        x_n = state / self.x_scale
        t_n = target_rel / self.pos_scale
        z = torch.cat([x_n, t_n], dim=-1)
        raw = self.net(z)

        thrust = torch.sigmoid(raw[..., 0]) * C.UINT16_MAX
        roll   = torch.tanh(raw[..., 1]) * C.PID_VEL_ROLL_MAX
        pitch  = torch.tanh(raw[..., 2]) * C.PID_VEL_PITCH_MAX
        yaw_r  = torch.tanh(raw[..., 3]) * C.YAW_RATE_MAX
        return torch.stack([thrust, roll, pitch, yaw_r], dim=-1)
