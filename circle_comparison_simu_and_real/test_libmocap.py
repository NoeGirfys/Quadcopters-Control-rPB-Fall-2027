import motioncapture
import time

mc = motioncapture.connect("optitrack", {"hostname": "192.168.0.56"})
print("Connexion établie, attente de frames...")

t_start = time.time()
n_frames = 0
while time.time() - t_start < 5.0:
    mc.waitForNextFrame()
    n_frames += 1
    bodies = mc.rigidBodies
    print(f"Frame {n_frames}: {len(bodies)} rigid bodies — {list(bodies.keys())}")
    if n_frames >= 10:
        break

if n_frames == 0:
    print("ÉCHEC : 0 frames reçues")
else:
    print(f"OK : {n_frames} frames reçues")