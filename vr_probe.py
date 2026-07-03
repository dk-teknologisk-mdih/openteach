"""Standalone diagnostic: check whether the VR headset is actually delivering
controller/pause packets to this PC.

Run this with teleop.py STOPPED (it binds the same ports the detector uses),
while the VR app is running and pointed at this PC's IP.

    python vr_probe.py

Expected: you should see 'remote' and 'pause' messages streaming in.
If it just sits at 'Waiting...', the headset is NOT reaching this machine
(wrong IP entered in the app, headset on a different subnet, or app not sending).
"""
import zmq

HOST = "0.0.0.0"
REMOTE_PORT = 8125   # network.yaml: remote_port
RESET_PORT = 8100    # network.yaml: teleop_reset_port

ctx = zmq.Context()

remote = ctx.socket(zmq.PULL)
remote.setsockopt(zmq.CONFLATE, 1)
remote.bind(f"tcp://{HOST}:{REMOTE_PORT}")

reset = ctx.socket(zmq.PULL)
reset.setsockopt(zmq.CONFLATE, 1)
reset.bind(f"tcp://{HOST}:{RESET_PORT}")

poller = zmq.Poller()
poller.register(remote, zmq.POLLIN)
poller.register(reset, zmq.POLLIN)

print(f"Listening for VR data on ports {REMOTE_PORT} (remote) and {RESET_PORT} (pause)...")
print("Move the controller / press the teleop button now. Ctrl-C to stop.\n")

got_any = False
last_gripper = None
while True:
    try:
        socks = dict(poller.poll(timeout=1000))
        if not socks:
            print("Waiting... no VR packets yet." if not got_any else "(no new packets in last 1s)")
            continue
        if remote in socks:
            got_any = True
            msg = remote.recv().decode(errors="replace").strip()
            # Format: TypeMarker|x,y,z|q1,q2,q3,q4|gripper|off_f|off_r|off_u
            # The gripper/index-trigger field is index 3.
            parts = msg.split("|")
            gripper = parts[3] if len(parts) > 3 else "?"
            flag = ""
            if gripper != last_gripper:
                flag = "   <-- GRIPPER CHANGED"
                last_gripper = gripper
            print("remote: gripper={}{}".format(gripper, flag))
        if reset in socks:
            got_any = True
            print("pause :", reset.recv().decode(errors="replace"))
    except KeyboardInterrupt:
        break

print("\nStopping probe.")
remote.close()
reset.close()
ctx.term()
