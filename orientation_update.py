import numpy as np

def orientation_update(accel, mag):
    """Compute orientation from accelerometer and magnetometer.

    Parameters
    ----------
    accel : array_like
        3-element iterable with accelerometer readings in m/s^2.
    mag : array_like
        3-element iterable with magnetometer readings in microtesla.

    Returns
    -------
    tuple of float
        roll, pitch and yaw angles in degrees.
    """
    a = np.asarray(accel, dtype=float)
    m = np.asarray(mag, dtype=float)

    if np.linalg.norm(a) == 0 or np.linalg.norm(m) == 0:
        raise ValueError("Acceleration or magnetometer vector has zero magnitude")

    a /= np.linalg.norm(a)
    m /= np.linalg.norm(m)

    ax, ay, az = a

    # roll and pitch from accelerometer
    roll = np.arctan2(ay, az)
    pitch = np.arctan2(-ax, np.sqrt(ay**2 + az**2))

    mx, my, mz = m

    # tilt compensated magnetometer
    mag_x = mx * np.cos(pitch) + my * np.sin(roll) * np.sin(pitch) + mz * np.cos(roll) * np.sin(pitch)
    mag_y = my * np.cos(roll) - mz * np.sin(roll)
    yaw = np.arctan2(-mag_y, mag_x)

    return np.degrees([roll, pitch, yaw])


if __name__ == "__main__":
    # Example usage
    accel_sample = [0.0, 0.0, 1.0]
    mag_sample = [1.0, 0.0, 0.0]
    rpy = orientation_update(accel_sample, mag_sample)
    print(f"Roll: {rpy[0]:.2f} deg, Pitch: {rpy[1]:.2f} deg, Yaw: {rpy[2]:.2f} deg")
