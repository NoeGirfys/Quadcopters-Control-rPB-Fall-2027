"""Circle flight — simulation or real Crazyflie 2.1+ with Flow Deck v2.

Both modes share the same sequence:
    1. Takeoff  : climb from z=0 to H
    2. Circle   : fly one full horizontal circle of radius R at height H
    3. Landing  : descend from H back to z=0

The RPM dispatch to either PyBullet or cflib is done directly in the
respective sim/real loops.
In real mode, PyBullet is never instantiated.

Usage
-----
Simulation (default):
    python circle_flight.py

Real drone (Flow Deck v2 required):
    python circle_flight.py --real

Real drone, custom URI:
    python circle_flight.py --real --uri radio://0/80/2M/E7E7E7E7E7
"""

import argparse
import time
import math
import numpy as np
from threading import Event
from scipy.spatial.transform import Rotation

# ── gym-pybullet-drones (only imported/used in sim mode) ──────────────────────
from gym_pybullet_drones.utils.enums import DroneModel, Physics
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl
from gym_pybullet_drones.utils.Logger import Logger

# ── flight parameters ──────────────────────────────────────────────────────────
DEFAULT_URI        = 'radio://0/80/2M/E7E7E7E7E7'
DRONE_MODEL        = DroneModel('cf2x')
SIMULATION_FREQ_HZ = 240
CONTROL_FREQ_HZ    = 48
CTRL_TIMESTEP      = 1.0 / CONTROL_FREQ_HZ

H          = 0.4    # cruise height          [m]
R          = 0.3    # circle radius          [m]
PERIOD     = 10     # circle period          [s]

# takeoff / landing shared parameters
TAKEOFF_SEC  = 3.0  # duration of climb from 0 to H [s]
LANDING_SEC  = 3.0  # duration of descent from H to 0 [s]

# ── PWM (from DSLPIDControl) ──────────────────────────────────────────
MIN_PWM       = 20000
MAX_PWM       = 65535

# ── cflib thrust range ─────────────────────────────────────────────────────────
CF_THRUST_MIN = 10001
CF_THRUST_MAX = 60000

# ── CF2X physical parameters (from cf2x.urdf) ────────────────────────────────
KF    = 3.16e-10   # thrust coefficient     [N / RPM^2]  (also used for HOVER_RPM)
KM    = 7.94e-12   # torque coefficient     [N·m / RPM^2]
ARM_L = 0.0397     # motor arm length       [m]
IXX   = 1.4e-5     # roll  inertia          [kg·m²]
IYY   = 1.4e-5     # pitch inertia          [kg·m²]
IZZ   = 2.17e-5    # yaw   inertia          [kg·m²]
MASS  = 0.027      # vehicle mass           [kg] MAYBE MORE WITH FLOWDECK V2
G     = 9.8        # gravitational accel    [m/s²]
T2W   = 2.25

MAX_RATE_DPS = 200.0   # safety clip for body rate setpoints [deg/s]

# hover RPM placeholder used when building obs in real mode
HOVER_RPM = math.sqrt((MASS * G) / (4 * KF))


# ══════════════════════════════════════════════════════════════════════════════
#  TRAJECTORY HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def build_circle_waypoints():
    """Return (NUM_WP, 2) array of xy waypoints for one full circle."""
    NUM_WP = CONTROL_FREQ_HZ * PERIOD
    wps = np.zeros((NUM_WP, 2))
    for i in range(NUM_WP):
        angle = (i / NUM_WP) * 2 * math.pi
        wps[i, 0] = R * math.cos(angle)
        wps[i, 1] = R * math.sin(angle) - R   # starts at (R, -R)
    return wps


# ══════════════════════════════════════════════════════════════════════════════
#  REAL STATE — cflib async logging
# ══════════════════════════════════════════════════════════════════════════════

_real_state = dict(x=0., y=0., z=0.,
                   vx=0., vy=0., vz=0.,
                   roll=0., pitch=0., yaw=0.,
                   gx=0., gy=0., gz=0.,
                   qx=0., qy=0., qz=0., qw=1.)
_state_ready = Event()


def _state_cb(timestamp, data, logconf):
    _real_state['x']     = data.get('stateEstimate.x',     0.) # meters, world frame
    _real_state['y']     = data.get('stateEstimate.y',     0.)
    _real_state['z']     = data.get('stateEstimate.z',     0.)
    _real_state['vx']    = data.get('stateEstimate.vx',    0.) #m/s, world frame
    _real_state['vy']    = data.get('stateEstimate.vy',    0.)
    _real_state['vz']    = data.get('stateEstimate.vz',    0.)
    _real_state['roll']  = data.get('stateEstimate.roll',  0.) # degrees
    _real_state['pitch'] = data.get('stateEstimate.pitch', 0.) # MAYBE INVERTED SIGN ?
    _real_state['yaw']   = data.get('stateEstimate.yaw',   0.)
    _real_state['gx']    = data.get('gyro.x',              0.) # deg/s, body frame
    _real_state['gy']    = data.get('gyro.y',              0.)
    _real_state['gz']    = data.get('gyro.z',              0.)
    _real_state['qx']    = data.get('stateEstimate.qx',    0.)
    _real_state['qy']    = data.get('stateEstimate.qy',    0.)
    _real_state['qz']    = data.get('stateEstimate.qz',    0.)
    _real_state['qw']    = data.get('stateEstimate.qw',    1.)
    _state_ready.set()


def start_logging(cf):
    """Open two 6-float log blocks at 50 Hz (max packet = 26 bytes)."""
    from cflib.crazyflie.log import LogConfig

    lc1 = LogConfig('StatePos', period_in_ms=20) # create a log block named 'StatePos' that sends data every 20 ms (50 Hz)
    for v in ['stateEstimate.x', 'stateEstimate.y', 'stateEstimate.z',
              'stateEstimate.vx', 'stateEstimate.vy', 'stateEstimate.vz']:
        lc1.add_variable(v, 'float') # add position and velocity variables to the log block
    lc1.data_received_cb.add_callback(_state_cb) # each time a log packet is received, _state_cb is called to update _real_state
    cf.cf.log.add_config(lc1) # add the log block to the Crazyflie log subsystem that checks that everything is in order
    lc1.start() # start the log block, which begins the periodic logging and calls _state_cb every 20 ms with the latest state estimate and gyro data

    lc2 = LogConfig('StateAtt', period_in_ms=20) # create a log block named 'StateAtt' that sends data every 20 ms (50 Hz)
    for v in ['stateEstimate.roll', 'stateEstimate.pitch', 'stateEstimate.yaw',
              'gyro.x', 'gyro.y', 'gyro.z']:
        lc2.add_variable(v, 'float') # add attitude and angular velocity variables to the log block
    lc2.data_received_cb.add_callback(_state_cb)
    cf.cf.log.add_config(lc2)
    lc2.start()

    lc3 = LogConfig('StateQuat', period_in_ms=20)
    for v in ['stateEstimate.qx', 'stateEstimate.qy',
              'stateEstimate.qz', 'stateEstimate.qw']:
        lc3.add_variable(v, 'float')
    lc3.data_received_cb.add_callback(_state_cb)
    cf.cf.log.add_config(lc3)
    lc3.start()

    return [lc1, lc2, lc3]


def real_state_to_obs() -> np.ndarray:
    """Build a (20,) obs vector in gym-pybullet-drones format from _real_state.

    Layout (matches _getDroneStateVector in BaseAviary):
        [0:3]   x, y, z          [m]
        [3:7]   qx, qy, qz, qw  (directly from firmware Kalman filter)
        [7:10]  roll, pitch, yaw [rad]
        [10:13] vx, vy, vz       [m/s]  world frame
        [13:16] wx, wy, wz       [rad/s] world frame
        [16:20] rpm0..3          (hover placeholder, never used by PID)
    """
    # Quaternion directly from firmware — no Euler -> quat conversion needed
    quat = np.array([_real_state['qx'], _real_state['qy'],
                     _real_state['qz'], _real_state['qw']])  # [qx, qy, qz, qw]

    # Euler angles in rad (still needed for obs[7:10])
    r = math.radians(_real_state['roll'])
    p = math.radians(_real_state['pitch'])
    y = math.radians(_real_state['yaw'])

    # gyro deg/s -> rad/s, body frame -> world frame
    rot    = Rotation.from_quat(quat)
    w_body = np.radians([_real_state['gx'], _real_state['gy'], _real_state['gz']])
    w_world = rot.as_matrix() @ w_body

    return np.concatenate([
        [_real_state['x'], _real_state['y'], _real_state['z']],
        quat,
        [r, p, y],
        [_real_state['vx'], _real_state['vy'], _real_state['vz']],
        w_world,
        [HOVER_RPM] * 4,
    ])


# ══════════════════════════════════════════════════════════════════════════════
#  TAKEOFF & LANDING
# ══════════════════════════════════════════════════════════════════════════════

def takeoff_sim(env, ctrl, logger, start_xy: np.ndarray) -> tuple:
    """Climb from z=0 to H inside PyBullet using the PID.

    The PID is given a fixed target at (start_xy, H) until the drone
    reaches within 2 cm of the target height.

    Parameters
    ----------
    start_xy : (2,) xy position of the drone at spawn (= first circle waypoint)

    Returns
    -------
    obs        : (1, 20) last observation after takeoff
    step_count : number of steps taken (for timestamp continuity)
    """
    from gym_pybullet_drones.utils.utils import sync

    target     = np.hstack([start_xy, H])
    target_rpy = np.zeros(3)

    obs, _, _, _, _ = env.step(np.zeros((1, 4)))  # obs is (1, 20)
    step  = 0
    START = time.time()

    print('[TAKEOFF-SIM] Climbing ...')
    while True:
        rpms, _, _ = ctrl.computeControl(
            control_timestep=CTRL_TIMESTEP,
            cur_pos=obs[0][0:3],
            cur_quat=obs[0][3:7],
            cur_vel=obs[0][10:13],
            cur_ang_vel=obs[0][13:16],
            target_pos=target,
            target_rpy=target_rpy,
        )
        obs, _, _, _, _ = env.step(rpms.reshape(1, 4))
        logger.log(drone=0, timestamp=step / CONTROL_FREQ_HZ,
                   state=obs[0],
                   control=np.hstack([target, target_rpy, np.zeros(6)]))
        env.render()
        sync(step, START, CTRL_TIMESTEP)
        step += 1

        if obs[0][2] >= H - 0.02:
            break

        # Safety: bail out after 2 * TAKEOFF_SEC
        if step > int(2 * TAKEOFF_SEC * CONTROL_FREQ_HZ):
            print('[TAKEOFF-SIM] Warning: timeout before target height')
            break

    print(f'[TAKEOFF-SIM] Reached z = {obs[0][2]:.3f} m')
    return obs, step


def landing_sim(env, ctrl, logger, step_offset: int, land_xy: np.ndarray,
                obs: np.ndarray) -> int:
    """Descend from H to z=0 inside PyBullet using the PID.

    The target z is linearly ramped from H down to 0 over LANDING_SEC seconds.
    The xy target stays fixed at land_xy (last circle waypoint position).

    Parameters
    ----------
    step_offset : timestamp offset for the logger (steps already elapsed)
    land_xy     : (2,) xy position to hold during descent
    obs         : (1, 20) last observation from the circle phase

    Returns
    -------
    step_count : total number of landing steps taken
    """
    from gym_pybullet_drones.utils.utils import sync

    steps      = int(LANDING_SEC * CONTROL_FREQ_HZ)
    target_rpy = np.zeros(3)
    START      = time.time()

    print('[LANDING-SIM] Descending ...')
    for i in range(steps):
        z_target = H * (1 - (i + 1) / steps)   # H -> 0
        target   = np.hstack([land_xy, z_target])

        rpms, _, _ = ctrl.computeControl(
            control_timestep=CTRL_TIMESTEP,
            cur_pos=obs[0][0:3],
            cur_quat=obs[0][3:7],
            cur_vel=obs[0][10:13],
            cur_ang_vel=obs[0][13:16],
            target_pos=target,
            target_rpy=target_rpy,
        )
        obs, _, _, _, _ = env.step(rpms.reshape(1, 4))
        logger.log(drone=0,
                   timestamp=(step_offset + i) / CONTROL_FREQ_HZ,
                   state=obs[0],
                   control=np.hstack([target, target_rpy, np.zeros(6)]))
        env.render()
        sync(i, START, CTRL_TIMESTEP)

    print(f'[LANDING-SIM] Reached z = {obs[0][2]:.3f} m')
    return steps


def takeoff_real(cf) -> int:
    """Climb from ground to H using send_hover_setpoint (ToF-based).

    The height target is linearly ramped over TAKEOFF_SEC seconds.
    Returns the number of steps taken (for timestamp continuity).
    """
    steps = int(TAKEOFF_SEC * CONTROL_FREQ_HZ)
    START = time.time()
    print(f'[TAKEOFF-REAL] Climbing to {H:.2f} m ...')

    for i in range(steps):
        z_target = H * (i + 1) / steps
        cf.cf.commander.send_hover_setpoint(0.0, 0.0, 0.0, z_target)
        elapsed = time.time() - START - i * CTRL_TIMESTEP
        sleep_t = CTRL_TIMESTEP - elapsed
        if sleep_t > 0:
            time.sleep(sleep_t)

    print(f'[TAKEOFF-REAL] Done  z = {_real_state["z"]:.3f} m')
    return steps


def landing_real(cf, step_offset: int, land_xy: np.ndarray, logger) -> int:
    """Descend from H to 0 using send_hover_setpoint (ToF-based).

    The height target is linearly ramped from H down to 0 over LANDING_SEC
    seconds, mirroring takeoff_real() in reverse.

    Parameters
    ----------
    step_offset : timestamp offset for the logger
    land_xy     : (2,) xy position at end of circle (unused by hover setpoint
                  but logged for consistency)

    Returns
    -------
    step_count : number of landing steps taken
    """
    steps = int(LANDING_SEC * CONTROL_FREQ_HZ)
    START = time.time()
    print(f'[LANDING-REAL] Descending to 0 m ...')

    for i in range(steps):
        z_target = H * (1 - (i + 1) / steps)   # H -> 0
        cf.cf.commander.send_hover_setpoint(0.0, 0.0, 0.0, z_target)

        obs = real_state_to_obs()
        logger.log(drone=0,
                   timestamp=(step_offset + i) / CONTROL_FREQ_HZ,
                   state=obs,
                   control=np.hstack([land_xy, z_target, np.zeros(9)]))

        elapsed = time.time() - START - i * CTRL_TIMESTEP
        sleep_t = CTRL_TIMESTEP - elapsed
        if sleep_t > 0:
            time.sleep(sleep_t)

    print(f'[LANDING-REAL] Done  z = {_real_state["z"]:.3f} m')
    return steps


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def run(real: bool = False, uri: str = DEFAULT_URI,
        gui: bool = True, plot: bool = True,
        rate: bool = False):

    # ── Circle waypoints ──────────────────────────────────────────────────────
    circle_wps = build_circle_waypoints()   # (NUM_WP, 2) xy only, starts at (R, -R)
    NUM_WP     = circle_wps.shape[0]
    wp_counter = 0

    # Starting xy position = first waypoint = (R, -R)
    start_xy = circle_wps[0, :].copy()

    # ── Logger ────────────────────────────────────────────────────────────────
    logger = Logger(logging_freq_hz=CONTROL_FREQ_HZ, num_drones=1,
                    output_folder='results')

    # ── PID ───────────────────────────────────────────────────────────────────
    ctrl = DSLPIDControl(drone_model=DRONE_MODEL)

    # ══════════════════════════════════════════════════════════════════════════
    #  SIMULATION MODE
    # ══════════════════════════════════════════════════════════════════════════
    if not real:
        from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
        from gym_pybullet_drones.utils.utils import sync

        # Spawn drone at z=0, at the start of the circle
        INIT_XYZS = np.array([[start_xy[0], start_xy[1], 0.0]])
        INIT_RPYS = np.zeros((1, 3))

        env = CtrlAviary(
            drone_model=DRONE_MODEL,
            num_drones=1,
            initial_xyzs=INIT_XYZS,
            initial_rpys=INIT_RPYS,
            physics=Physics('pyb'),
            neighbourhood_radius=1,
            pyb_freq=SIMULATION_FREQ_HZ,
            ctrl_freq=CONTROL_FREQ_HZ,
            gui=gui,
            record=False,
            obstacles=False,
            user_debug_gui=False,
        )

        # ── Takeoff ───────────────────────────────────────────────────────────
        obs, step_offset = takeoff_sim(env, ctrl, logger, start_xy=start_xy)
        ctrl.reset() # ctrl.last_rpy = 0 and ctrl.pos_e = 0 and especially ctrl.integral_pos_e = 0 after reset,
        # which cleans up the transition between takeoff and circle, otherwise the PID would have accumulated errors
        # during takeoff and would apply a nonzero control at the first step of the circle, which would make the drone
        # deviate from the ideal trajectory right from the start. With the reset, the PID starts fresh at the beginning
        # of the circle, which is what we want for a clean comparison between sim and real.

        # ── Circle — stop after exactly one full lap ───────────────────────────
        print('[CIRCLE-SIM] Starting circle ...')
        START = time.time()
        i     = 0

        while True:
            rpms, _, _ = ctrl.computeControl(
                control_timestep=CTRL_TIMESTEP,
                cur_pos=obs[0][0:3],
                cur_quat=obs[0][3:7],
                cur_vel=obs[0][10:13],
                cur_ang_vel=obs[0][13:16],
                target_pos=np.hstack([circle_wps[wp_counter], H]),
                target_rpy=np.zeros(3),
            )
            obs, _, _, _, _ = env.step(rpms.reshape(1, 4))
            logger.log(drone=0,
                       timestamp=(step_offset + i) / CONTROL_FREQ_HZ,
                       state=obs[0],
                       control=np.hstack([circle_wps[wp_counter], H,
                                          np.zeros(9)]))
            env.render()
            sync(i, START, CTRL_TIMESTEP)

            wp_counter = (wp_counter + 1) % NUM_WP
            i += 1

            # One full lap completed when wp_counter wraps back to 0
            if wp_counter == 0:
                print('[CIRCLE-SIM] One full lap completed.')
                break

        # ── Landing ───────────────────────────────────────────────────────────
        land_xy      = circle_wps[0]   # back at start of circle
        circle_steps = i
        landing_sim(env, ctrl, logger,
                    step_offset=step_offset + circle_steps,
                    land_xy=land_xy, obs=obs)

        env.close()

    # ══════════════════════════════════════════════════════════════════════════
    #  REAL MODE  (PyBullet never instantiated)
    # ══════════════════════════════════════════════════════════════════════════
    else:
        import cflib.crtp
        from cflib.crazyflie import Crazyflie
        from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
        from threading import Event as TEvent

        # ── Connect ───────────────────────────────────────────────────────────
        deck_event = TEvent()
        def _deck_cb(_, value_str):
            if int(value_str):
                deck_event.set()

        cflib.crtp.init_drivers()
        cf_ctx = SyncCrazyflie(uri, cf=Crazyflie(rw_cache='./cache'))
        cf = cf_ctx.__enter__()
        print(f'[INFO] Connected on {uri}')
        """
        The two followings are equivalents : 
        with SyncCrazyflie(uri, cf=Crazyflie(rw_cache='./cache')) as cf:
            # cf is connected here
            cf.cf.commander.send_setpoint(...)

        cf_ctx = SyncCrazyflie(uri, cf=Crazyflie(rw_cache='./cache'))
        cf = cf_ctx.__enter__()   # equivalent to the "as cf" of the with
        # here cf is connected
        cf.cf.commander.send_setpoint(...)
        cf_ctx.__exit__(None, None, None)  # equivalent to exiting the with block, which disconnects cf

        Here we chose to connect and disconnect synchronously so at the end we control perfectly the order : stop motors, stop logs, disconnect.
        """

        # ── Check Flow Deck ───────────────────────────────────────────────────
        cf.cf.param.add_update_callback(group='deck', name='bcFlow2', cb=_deck_cb)
        # deck.bcFlow2 is a firmware setting (not a log variable). Its value is 1 if the Flow Deck v2 is physically connected
        # and detected at startup; otherwise, it is 0.
        # add_update_callback triggers the radio request and store the callback for when we will receive the info from the drone
        # and when it happens it calls _deck_cb, which happens shortly after connection. If the value is 1,
        # _deck_cb sets deck_event, otherwise deck_event remains unset and we can detect that the Flow Deck v2 is not present.
        if not deck_event.wait(timeout=5):
            cf_ctx.__exit__(None, None, None)
            raise RuntimeError('Flow Deck v2 not detected!')
        print('[INFO] Flow Deck v2 OK')

        # ── Start logging ─────────────────────────────────────────────────────
        log_cfgs = start_logging(cf)
        """
        _real_state = dict(x=0., y=0., z=0.,
                   vx=0., vy=0., vz=0.,
                   roll=0., pitch=0., yaw=0.,
                   gx=0., gy=0., gz=0.)
        _state_ready = Event()


        def _state_cb(timestamp, data, logconf):
            _real_state['x']     = data.get('stateEstimate.x',     0.)
            _real_state['y']     = data.get('stateEstimate.y',     0.)
            _real_state['z']     = data.get('stateEstimate.z',     0.)
            _real_state['vx']    = data.get('stateEstimate.vx',    0.)
            _real_state['vy']    = data.get('stateEstimate.vy',    0.)
            _real_state['vz']    = data.get('stateEstimate.vz',    0.)
            _real_state['roll']  = data.get('stateEstimate.roll',  0.)
            _real_state['pitch'] = data.get('stateEstimate.pitch', 0.)
            _real_state['yaw']   = data.get('stateEstimate.yaw',   0.)
            _real_state['gx']    = data.get('gyro.x',              0.)
            _real_state['gy']    = data.get('gyro.y',              0.)
            _real_state['gz']    = data.get('gyro.z',              0.)
            _state_ready.set()
            
        def start_logging(cf):
            #Open two 6-float log blocks at 50 Hz (max packet = 26 bytes).
            from cflib.crazyflie.log import LogConfig

            lc1 = LogConfig('StatePos', period_in_ms=20) # create a log block named 'StatePos' that sends data every 20 ms (50 Hz)
            for v in ['stateEstimate.x', 'stateEstimate.y', 'stateEstimate.z',
                    'stateEstimate.vx', 'stateEstimate.vy', 'stateEstimate.vz']:
                lc1.add_variable(v, 'float') # add position and velocity variables to the log block
            lc1.data_received_cb.add_callback(_state_cb) # each time a log packet is received, _state_cb is called to update _real_state
            cf.cf.log.add_config(lc1) # add the log block to the Crazyflie log subsystem that checks that everything is in order
            lc1.start() # start the log block, which begins the periodic logging and calls _state_cb every 20 ms with the latest state estimate and gyro data

            lc2 = LogConfig('StateAtt', period_in_ms=20) # create a log block named 'StateAtt' that sends data every 20 ms (50 Hz)
            for v in ['stateEstimate.roll', 'stateEstimate.pitch', 'stateEstimate.yaw',
                    'gyro.x', 'gyro.y', 'gyro.z']:
                lc2.add_variable(v, 'float') # add attitude and angular velocity variables to the log block
            lc2.data_received_cb.add_callback(_state_cb)
            cf.cf.log.add_config(lc2)
            lc2.start()

            return [lc1, lc2]
        """
        print('[INFO] Waiting for state estimator ...')
        if not _state_ready.wait(timeout=10):
            cf_ctx.__exit__(None, None, None)
            raise RuntimeError('State estimator timeout')
        print(f'[INFO] State ready  z = {_real_state["z"]:.3f} m')

        # ── Arm + takeoff ─────────────────────────────────────────────────────
        is_flying = False   # tracks whether the drone is airborne
        try:
            cf.cf.platform.send_arming_request(True) # mandatory arming before sending any setpoint, otherwise the firmware will ignore them
            time.sleep(1.0)
            is_flying = True
            step_offset = takeoff_real(cf)
            time.sleep(0.5)

            ctrl.reset()

            # ── Configure firmware control mode ───────────────────────────────
            if rate:
                # Rate mode: bypasses the firmware attitude PID.
                # The firmware only runs its fast gyro rate controller (~1 kHz)
                # to track the body-rate setpoints we send.
                cf.cf.param.set_value('flightmode.stabModeRoll',  '0')
                cf.cf.param.set_value('flightmode.stabModePitch', '0')
                cf.cf.param.set_value('flightmode.stabModeYaw',   '0')
                print('[INFO] Rate mode enabled')
            else:
                # Angle mode (default): send_setpoint takes absolute roll/pitch
                # angles [deg], which the firmware attitude PID tracks.
                cf.cf.param.set_value('flightmode.stabModeRoll',  '1')
                cf.cf.param.set_value('flightmode.stabModePitch', '1')
                cf.cf.param.set_value('flightmode.stabModeYaw',   '1')
                print('[INFO] Angle mode enabled')

            # ── Circle — stop after exactly one full lap ───────────────────────
            print('[CIRCLE-REAL] Starting circle ...')
            START = time.time()
            i     = 0

            while True:
                obs = real_state_to_obs()

                if not rate:
                    # ── ANGLE MODE ────────────────────────────────────────────
                    # _dslPIDPositionControl returns thrust [PWM] and target
                    # roll/pitch/yaw [rad] directly — no RPM conversion needed.
                    thrust_pwm, target_euler, _ = ctrl._dslPIDPositionControl(
                        control_timestep=CTRL_TIMESTEP,
                        cur_pos=obs[0:3],
                        cur_quat=obs[3:7],
                        cur_vel=obs[10:13],
                        target_pos=np.hstack([circle_wps[wp_counter], H]),
                        target_rpy=np.zeros(3),
                        target_vel=np.zeros(3),
                    )
                    # thrust_pwm [MIN_PWM, MAX_PWM] -> cflib integer [10001, 60000]
                    thrust_cf = int(np.clip(
                        CF_THRUST_MIN + (thrust_pwm - MIN_PWM) / (MAX_PWM - MIN_PWM) * (CF_THRUST_MAX - CF_THRUST_MIN),
                        CF_THRUST_MIN, CF_THRUST_MAX
                    ))
                    # target_euler [rad] -> [deg]
                    roll_deg  = float(np.clip(np.degrees(target_euler[0]), -30., 30.))
                    pitch_deg = float(np.clip(np.degrees(target_euler[1]), -30., 30.))
                    yaw_deg   = float(np.clip(np.degrees(target_euler[2]), -30., 30.))
                    cf.cf.commander.send_setpoint(roll_deg, pitch_deg, yaw_deg, thrust_cf)

                else:
                    # ── RATE MODE ─────────────────────────────────────────────
                    # computeControl returns RPMs; convert to physical forces
                    # and torques, then to body angular rate setpoints.
                    rpms, _, _ = ctrl.computeControl(
                        control_timestep=CTRL_TIMESTEP,
                        cur_pos=obs[0:3],
                        cur_quat=obs[3:7],
                        cur_vel=obs[10:13],
                        cur_ang_vel=obs[13:16],
                        target_pos=np.hstack([circle_wps[wp_counter], H]),
                        target_rpy=np.zeros(3),
                    )
                    # Step 1 — RPM -> forces per motor [N]
                    forces = rpms**2 * KF                            # (4,) [N]

                    # Step 2 — Forces -> physical torques [N·m]
                    # CF2X: motors at 45 deg, projected arm = L/sqrt(2)
                    # Motor layout (top view):
                    #   0(CCW) 1(CW)
                    #   3(CW)  2(CCW)
                    L_eff        = ARM_L / np.sqrt(2)
                    tau_roll_Nm  = (forces[0] + forces[1]
                                    - forces[2] - forces[3]) * L_eff # [N·m]
                    tau_pitch_Nm = (-forces[0] + forces[1]
                                    + forces[2] - forces[3]) * L_eff # [N·m]
                    tau_yaw_Nm   = (-rpms[0]**2 + rpms[1]**2
                                    - rpms[2]**2 + rpms[3]**2) * KM # [N·m]

                    # Step 3 — Collective thrust -> cflib integer [10001, 60000]
                    F_total   = float(np.sum(forces))
                    F_hover   = MASS * G
                    thrust_cf = int(np.clip(
                        CF_THRUST_MIN + (F_total / (T2W * F_hover)) * (CF_THRUST_MAX - CF_THRUST_MIN),
                        CF_THRUST_MIN, CF_THRUST_MAX
                    ))

                    # Step 4 — Full Newton-Euler: alpha = J^-1*(tau - omega x J*omega)
                    cur_rates = np.radians([_real_state['gx'],
                                            _real_state['gy'],
                                            _real_state['gz']])
                    J    = np.diag([IXX, IYY, IZZ])
                    tau  = np.array([tau_roll_Nm, tau_pitch_Nm, tau_yaw_Nm])
                    gyro = np.cross(cur_rates, J @ cur_rates)
                    alpha = np.linalg.solve(J, tau - gyro)           # [rad/s²]

                    # Step 5 — Euler integration: omega_des = omega_cur + alpha*dt
                    rates_des = cur_rates + alpha * CTRL_TIMESTEP    # [rad/s]
                    rollrate_dps  = float(np.clip(np.degrees(rates_des[0]),
                                                  -MAX_RATE_DPS, MAX_RATE_DPS))
                    pitchrate_dps = float(np.clip(np.degrees(rates_des[1]),
                                                  -MAX_RATE_DPS, MAX_RATE_DPS))
                    yawrate_dps   = float(np.clip(np.degrees(rates_des[2]),
                                                  -MAX_RATE_DPS, MAX_RATE_DPS))
                    cf.cf.commander.send_setpoint(rollrate_dps, pitchrate_dps, yawrate_dps, thrust_cf)

                logger.log(drone=0,
                           timestamp=(step_offset + i) / CONTROL_FREQ_HZ,
                           state=real_state_to_obs(),
                           control=np.hstack([circle_wps[wp_counter], H,
                                              np.zeros(9)]))

                wp_counter = (wp_counter + 1) % NUM_WP
                i += 1

                # One full lap completed when wp_counter wraps back to 0
                if wp_counter == 0:
                    print('[CIRCLE-REAL] One full lap completed.')
                    break

                elapsed = time.time() - START - i * CTRL_TIMESTEP
                sleep_t = CTRL_TIMESTEP - elapsed
                if sleep_t > 0:
                    time.sleep(sleep_t)

            # ── Landing ───────────────────────────────────────────────────────
            circle_steps = i
            landing_real(cf,
                         step_offset=step_offset + circle_steps,
                         land_xy=circle_wps[0],
                         logger=logger)
            is_flying = False

        finally:
            print('[INFO] Stopping ...')
            if is_flying:
                # Ctrl+C or exception while airborne: land gently before closing
                print('[INFO] Emergency landing ...')
                landing_real(cf,
                             step_offset=0,
                             land_xy=np.array([_real_state['x'],
                                               _real_state['y']]),
                             logger=logger)
            cf.cf.commander.send_setpoint(0, 0, 0, 0)
            time.sleep(0.1)
            for lc in log_cfgs:
                lc.stop()
            cf_ctx.__exit__(None, None, None)
            print('[INFO] Disconnected.')

    # ── Save & plot ───────────────────────────────────────────────────────────
    logger.save()
    logger.save_as_csv('circle')
    if plot:
        logger.plot()


# ══════════════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Circle flight — sim or real CF2X')
    parser.add_argument('--real',    action='store_true',
                        help='Real Crazyflie (default: simulation)')
    parser.add_argument('--uri',     default=DEFAULT_URI,
                        help=f'Crazyflie URI (default: {DEFAULT_URI})')
    parser.add_argument('--no-gui',  dest='gui',  action='store_false',
                        help='Disable PyBullet GUI in sim mode')
    parser.add_argument('--no-plot', dest='plot', action='store_false',
                        help='Disable end-of-run plots')
    parser.add_argument('--rate', action='store_true',
                        help='Rate mode: send body rates instead of angles (default: angle mode)')
    parser.set_defaults(gui=True, plot=True)
    args = parser.parse_args()
    run(real=args.real, uri=args.uri, gui=args.gui, plot=args.plot, rate=args.rate)