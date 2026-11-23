# Live ultrasound and LD-19 overlay

From the platform repository, load `source setupenv` to use the GNA tooling.
Run the browser server using the tracker environment:

```sh
cd ultra_track_and_map
.venv/bin/python audiodevice_server.py --lidar-port /dev/ttyUSB0 --lidar-root
```

Open <http://localhost:8765/>. **Start LiDAR** reads only the LD-19; **Start**
starts ultrasound playback and calibration independently. Keep the room still
and remove the target during ultrasound calibration. The existing acoustic
safety warning still applies.

The port is on the ADB target, not the host. If only one ADB device is connected,
it is detected automatically; use `--serial` to select a device when multiple
ones are connected. LD-19 uses 230400 baud,
8N1, raw input. `--lidar-root` explicitly opts into target `su 0` for ports
inaccessible to the ADB shell. It does not change permissions. Serial settings
are saved and restored on Stop, failure, or normal server shutdown. Do not
run another reader on the same port. Forced termination or an ADB disconnect
can prevent restoration; the server reports cleanup failures. After reconnecting,
restore the previous port configuration before starting another reader.
Without `--lidar-port`, ultrasound remains usable on its own.

Green points are raw LiDAR ranges, not identified targets or a fitted room map.
Each scan contains one revolution; corrupt packets are rejected with CRC-8
and counted. Zero ranges are omitted. Missing scans time out after three
seconds with a visible error; stale returns disappear from the browser after
1.5 seconds. Stop LiDAR does not stop ultrasound and vice versa.

Set forward/right offsets in metres and yaw in degrees to match the physical
mounting pose. Zero yaw points the LiDAR's zero-angle ray forward; positive
angles rotate clockwise towards the right. Defaults assume coincident sensor
origins and aligned axes, which must be checked on hardware. The pose affects
only the browser overlay. Ultrasound estimates remain experimental: no
automatic alignment, time synchronization, sensor fusion, or accuracy claim.
Different sensor heights may see different objects.

Layer toggles and 3/6/12-metre view radii help compare returns with the ultrasound
evidence grid. The wider view does not extend the ultrasound detection region.

## Recordings

`audiodevice_recording.py` records ultrasound and LD-19 scans together and replays
them through `audiodevice_mapper.AudiodeviceMapper` against the LiDAR walker track:

```sh
.venv/bin/python audiodevice_recording.py record recordings/walk2 --seconds 30 \
    --lidar-port /dev/ttyUSB0 --lidar-root --note "what happened"
.venv/bin/python audiodevice_recording.py replay recordings/walk1
```

To watch a recording in a GUI, replay it in place of the device, with no
device or ADB needed and no audio played. Then press **Start** (and **Start
LiDAR** for the recorded scans) as in a live session:

```sh
.venv/bin/python audiodevice_server.py --recording recordings/walk1      # browser, http://localhost:8765/
.venv/bin/python audiodevice_live.py --recording recordings/walk1_sim    # Matplotlib window
```

`--speed 2` replays twice as fast. Both GUIs run their own live tracker
(`audiodevice_tracking`), not the `AudiodeviceMapper` that `replay` scores.

Audio is stored losslessly as 24-bit six-channel FLAC (`STEM.flac`); chunk host
times, LiDAR scans, an optional walker track and metadata go in `STEM.npz`
(see the module docstring). LiDAR scans are timestamped on arrival, about 0.1 s
after the matching ping; replay compensates for this.

`recordings/walk1` (55 s, 18–22 kHz at 20%, LiDAR at the device origin,
zero angle forward) holds a still device with the room empty for 0–5 s, one person
walking in front of it for 5–29 s, and an empty room after about 30 s. Its
`track` was extracted from LiDAR points that differ from the empty-room scans.
Replay gives 69% of target pings tracked, range error median 0.05 m (p90
0.16 m), lateral error median 0.22 m (p90 0.56 m), no false tracks, and a
facing wall at 4.32 m (LiDAR about 4.3 m). Before the mic-layout correction and
the new angle estimate (see the README), lateral error was 0.27 m (p90 0.97 m)
and 2% of target-absent pings had tracks.

Protocol reference: [LDROBOT SDK packet layout and decoder](https://github.com/ldrobotSensorTeam/ldlidar_stl_sdk/tree/master/ldlidar_driver/src/dataprocess).
