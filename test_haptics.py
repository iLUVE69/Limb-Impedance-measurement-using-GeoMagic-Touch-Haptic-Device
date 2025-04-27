#!/usr/bin/env python
""" Haptic Path Following Experiment with Drift (Batched Trials, Separate Files, Single Straight Line Path, Assumed User Force)

This program implements a haptic experiment with multiple batches of trials.
Each batch can have a different force/drift condition. The user follows a
single straight line path from start to end with a haptic device. A sudden force
drift is applied in specific batches. Data is logged into separate CSV files
for each batch, organized into folders. Uses a single straight line for the path.

ASSUMPTION: This code assumes, for logging purposes, that the force applied
by the user is equal in magnitude and opposite in direction to the commanded
force set by the haptic device (F_user = -F_commanded). This is a
simplification that ignores device passive dynamics and inertia, and will
impact the accuracy of impedance analysis.

"""

import pygame
from pyOpenHaptics.hd_device import HapticDevice
import pyOpenHaptics.hd as hd
import time
from dataclasses import dataclass, field
from pyOpenHaptics.hd_callback import hd_callback
import random
import math
import csv
import numpy as np
import os # Import os for file operations

# --- Configuration Constants ---
SCREEN_WIDTH = 1200
SCREEN_HEIGHT = 900
FPS = 100 # Pygame frame rate (Data logging rate will be tied to this)

# Haptic device scaling: How haptic device units map to screen pixels
# Tune these based on your device's workspace and desired game area.
# DECREASE these values if the cursor is too sensitive (moves too fast).
# INCREASE these values if the cursor is not sensitive enough (moves too slow).
HAPTIC_SCALE_X = 9 # pixels per meter (example value, ADJUST THIS)
HAPTIC_SCALE_Y = 9 # pixels per meter (example value, ADJUST THIS)
HAPTIC_SCALE_Z = 0    # We are doing a 2D task, so Z scaling is 0 for cursor movement mapping

# Offset for mapping haptic device space to screen space.
# This point in HAPTIC DEVICE SPACE (in meters) will map to (HAPTIC_SCREEN_ORIGIN_X, HAPTIC_SCREEN_ORIGIN_Y) on the screen.
# Often, haptic device (0,0,0) is the neutral point. If you want the neutral point
# to map to the screen center, set these to SCREEN_WIDTH/2, SCREEN_HEIGHT/2.
HAPTIC_SCREEN_ORIGIN_X = SCREEN_WIDTH // 2
HAPTIC_SCREEN_ORIGIN_Y = SCREEN_HEIGHT // 2

# --- Path Definition (Waypoints for Straight Line) ---
# Define the path using only the start and end points.
# The path will be a single straight line between these two points.
SIMPLE_PATH_WAYPOINTS_SCREEN = [
    (50, SCREEN_HEIGHT // 2),         # Start Point
    (SCREEN_WIDTH - 50, SCREEN_HEIGHT // 2) # End Point
]
PATH_POINTS_PER_SEGMENT = 100 # Density of points generated along the straight line for drawing/calculations (increased for smoother drawing of a single line)

# Haptic Force Parameters
BASE_DRIFT_FORCE_MAGNITUDE = 3.5 # Magnitude of the constant drift force when enabled (in device units, e.g., Newtons) - ADJUST
DRIFT_FORCE_ANGLE_DEG = 90 # Angle of the drift force (in degrees, relative to +x axis)
DRIFT_START_TIME = 0.3 # Time in seconds from experiment start when the drift force starts - ADJUST

# Visuals
COLOR_BG = (10, 10, 30) # Dark Blue
COLOR_PATH = (100, 100, 255) # Light Blue
COLOR_WAYPOINTS = (200, 200, 0) # Yellowish for waypoints
COLOR_START = (0, 255, 0) # Green
COLOR_END = (255, 0, 0) # Red
COLOR_CURSOR = (255, 255, 0) # Yellow
CURSOR_RADIUS = 10

# Data Logging
BASE_LOG_FOLDER = "haptic_experiment_data" # Changed base folder name

# --- Global State ---
@dataclass
class DeviceState:
    button: bool = False
    position: list = field(default_factory=lambda: [0, 0, 0]) # 3D position from device (meters)
    velocity: list = field(default_factory=lambda: [0, 0, 0]) # 3D velocity from device (m/s, calculated)
    commanded_force: list = field(default_factory=lambda: [0, 0, 0])    # 3D force commanded to device (Newtons)

    # State for relative mapping and velocity calculation
    initial_haptic_position: list = field(default_factory=lambda: [0, 0, 0])
    _last_position: list = field(default_factory=lambda: [0, 0, 0])
    _last_callback_time: float = 0.0 # Timestamp of the last callback

# Global instance of DeviceState
device_state = DeviceState()

# --- Haptic Callback Function (Runs at Device Frequency) ---
@hd_callback
def state_callback():
    global device_state

    transform = hd.get_transform()
    button = hd.get_buttons()

    current_time = time.perf_counter()
    dt = current_time - device_state._last_callback_time if device_state._last_callback_time else 0.0

    current_pos = [transform[3][0], transform[3][1], transform[3][2]]
    device_state.position = current_pos

    if dt > 0:
        device_state.velocity = [
            (current_pos[0] - device_state._last_position[0]) / dt,
            (current_pos[1] - device_state._last_position[1]) / dt,
            (current_pos[2] - device_state._last_position[2]) / dt
        ]
    else:
         device_state.velocity = [0, 0, 0]

    device_state._last_position = current_pos
    device_state._last_callback_time = current_time

    device_state.button = True if button == 1 else False

    hd.set_force(device_state.commanded_force)


# --- Linear Path Generation ---
# Generates points along a single straight line segment connecting the two waypoints.
def generate_linear_path_points(waypoints_screen, points_per_segment):
    path_points = []
    num_waypoints = len(waypoints_screen)

    # This function is now specifically for a single line between the first and last waypoint
    if num_waypoints < 2:
        return waypoints_screen # Need at least two points for a line

    p1 = np.array(waypoints_screen[0]) # Use the first waypoint as the start
    p2 = np.array(waypoints_screen[-1]) # Use the last waypoint as the end

    # Generate points along the segment from p1 to p2
    for j in range(points_per_segment + 1): # Include both p1 and p2
        t = j / points_per_segment
        point = p1 + t * (p2 - p1)
        path_points.append(tuple(point.tolist())) # Store as tuple

    return path_points


# --- Path Following Error Calculation ---
# Finds the closest point on the path to the cursor and calculates errors
# This logic works on the dense list of path points generated by the linear function.
def calculate_path_errors(cursor_pos, path_points):
    if not path_points:
        return None, 0, 0

    cursor_np = np.array(cursor_pos)
    path_np = np.array(path_points)

    squared_distances = np.sum((path_np - cursor_np)**2, axis=1)
    closest_point_index = np.argmin(squared_distances)
    closest_point_sampled = path_points[closest_point_index]

    tracking_error = np.sqrt(squared_distances[closest_point_index])

    # Simplified Cross Error (using segment between sampled points near closest)
    p1_idx = max(0, closest_point_index - 1)
    p2_idx = min(len(path_points) - 1, closest_point_index + 1)

    if p1_idx == p2_idx:
        return closest_point_sampled, tracking_error, 0.0

    p1 = np.array(path_points[p1_idx])
    p2 = np.array(path_points[p2_idx])

    line_vec = p2 - p1
    point_vec = cursor_np - p1
    line_len_sq = np.sum(line_vec**2)

    if line_len_sq == 0:
         cross_error = np.linalg.norm(point_vec)
    else:
        t = np.dot(point_vec, line_vec) / line_len_sq
        t = max(0, min(1, t))
        projection_point = p1 + t * line_vec
        cross_error = np.linalg.norm(cursor_np - projection_point)

    # Calculate signed cross error using the 2D cross product relative to the segment direction
    segment_vec_normalized = line_vec / np.linalg.norm(line_vec) if np.linalg.norm(line_vec) > 0 else np.array([0.0, 0.0])
    vec_to_cursor_from_segment_start = cursor_np - p1
    cross_product_z = segment_vec_normalized[0] * vec_to_cursor_from_segment_start[1] - segment_vec_normalized[1] * vec_to_cursor_from_segment_start[0]

    signed_cross_error = cross_error * np.sign(cross_product_z)

    return closest_point_sampled, tracking_error, signed_cross_error

# --- Haptic Feedback Calculation (Runs in Main Loop) ---
def calculate_haptic_feedback(current_time, start_time, apply_drift_this_trial):
    commanded_force = [0, 0, 0]
    drift_active = False

    if apply_drift_this_trial and (current_time - start_time >= DRIFT_START_TIME):
        current_drift_vector = [
            BASE_DRIFT_FORCE_MAGNITUDE * math.cos(math.radians(DRIFT_FORCE_ANGLE_DEG)),
            BASE_DRIFT_FORCE_MAGNITUDE * math.sin(math.radians(DRIFT_FORCE_ANGLE_DEG)),
            0
        ]
        commanded_force[0] += current_drift_vector[0]
        commanded_force[1] += current_drift_vector[1]
        commanded_force[2] += current_drift_vector[2]
        drift_active = True

    device_state.commanded_force = commanded_force

    return drift_active

# --- Data Logging ---
class HapticDataLogger:
    def __init__(self, filepath):
        self.filepath = filepath # Full path including folder and filename
        self.file = None
        self.writer = None
        self._start_time = None
        self._header_written = False # Flag to ensure header is written only once per file

    def start(self):
        # Ensure the directory exists
        directory = os.path.dirname(self.filepath)
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"Created directory: {directory}")

        # Open file in write mode ('w') for a new file per batch
        self.file = open(self.filepath, 'w', newline='')
        self.writer = csv.writer(self.file)

        # Always write header for a new file
        self.writer.writerow([
            'Trial', # Added trial number (within the batch)
            'Batch', # Added batch number/description (e.g., "NoDrift1", "Drift", "NoDrift2")
            'Timestamp', # Time since experiment started (seconds)
            'DevicePosX', 'DevicePosY', 'DevicePosZ', # Haptic device position (meters)
            'DeviceVelX', 'DeviceVelY', 'DeviceVelZ', # Haptic device velocity (m/s)
            'AssumedUserForceX', 'AssumedUserForceY', 'AssumedUserForceZ', # Assumed user force (Newtons, = -CommandedForce)
            'CommandedForceX', 'CommandedForceY', 'CommandedForceZ', # Force commanded by the program (Newtons)
            'CursorPosX', 'CursorPosY', # Cursor position on screen (pixels)
            'ClosestPathX', 'ClosestPathY', # Closest point on path (pixels)
            'TrackingError', # Distance from cursor to closest path point (pixels)
            'CrossError', # Perpendicular distance from cursor to path segment (pixels)
            'Button', # Haptic device button state (boolean)
            'DriftActive' # Is the drift force currently applied (boolean)
        ])
        self._header_written = True # Set flag after writing header

        self._start_time = time.perf_counter()
        print(f"Logging data to {self.filepath}")

    def log(self, trial_num_in_batch, batch_desc, device_state: DeviceState, cursor_pos, closest_path_point, tracking_error, cross_error, drift_active):
        if not self.writer or self._start_time is None:
            return

        timestamp = time.perf_counter() - self._start_time

        assumed_user_force = [
            -device_state.commanded_force[0],
            -device_state.commanded_force[1],
            -device_state.commanded_force[2]
        ]

        row = [
            trial_num_in_batch, # Log trial number within the batch
            batch_desc, # Log batch description
            timestamp,
            device_state.position[0], device_state.position[1], device_state.position[2],
            device_state.velocity[0], device_state.velocity[1], device_state.velocity[2],
            assumed_user_force[0], assumed_user_force[1], assumed_user_force[2],
            device_state.commanded_force[0], device_state.commanded_force[1], device_state.commanded_force[2],
            cursor_pos[0], cursor_pos[1],
            closest_path_point[0] if closest_path_point else '',
            closest_path_point[1] if closest_path_point else '',
            tracking_error, cross_error,
            device_state.button,
            drift_active
        ]

        self.writer.writerow(row)

    def stop(self):
        if self.file:
            self.file.close()
            print(f"Finished logging data for this batch to {self.filepath}")

# --- Main Game and Experiment Loop (Runs a single trial) ---
def run_single_trial(trial_num_in_batch, batch_desc, logger, apply_drift_this_trial):
    global device_state

    pygame.init()
    surface = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption(f"Haptic Path Following (Batch: {batch_desc}, Trial {trial_num_in_batch})")

    clock = pygame.time.Clock()
    running = True

    # Generate the linear path points from the waypoints (now just start and end)
    path_points_screen = generate_linear_path_points(SIMPLE_PATH_WAYPOINTS_SCREEN, PATH_POINTS_PER_SEGMENT)
    start_point_screen = np.array(SIMPLE_PATH_WAYPOINTS_SCREEN[0])
    end_point_screen = np.array(SIMPLE_PATH_WAYPOINTS_SCREEN[-1])

    # --- Initial Haptic State and Cursor Mapping ---
    print("Move device to a comfortable starting position for the next trial and wait...")
    time.sleep(1.5)

    device_state.initial_haptic_position = list(device_state.position)
    print(f"Initial haptic position recorded: {device_state.initial_haptic_position}")

    cursor_pos = list(start_point_screen)
    print(f"Cursor starting at screen position: {cursor_pos}")

    experiment_start_time = time.perf_counter()

    print(f"Starting Trial {trial_num_in_batch} in Batch '{batch_desc}'. Follow the path.")
    if apply_drift_this_trial:
        print(f"Drift force will activate after {DRIFT_START_TIME} seconds.")
    else:
        print("No drift force in this trial.")


    # --- Main Loop ---
    while running:
        clock.tick(FPS)

        current_experiment_time = time.perf_counter()

        # --- Read Haptic Input (State updated by state_callback) ---
        haptic_pos = device_state.position

        # --- Update Cursor Position (Relative Mapping) ---
        haptic_displacement = [
            haptic_pos[0] - device_state.initial_haptic_position[0],
            haptic_pos[1] - device_state.initial_haptic_position[1],
            haptic_pos[2] - device_state.initial_haptic_position[2]
        ]

        cursor_pos[0] = start_point_screen[0] + haptic_displacement[0] * HAPTIC_SCALE_X
        cursor_pos[1] = start_point_screen[1] + (-haptic_displacement[1]) * HAPTIC_SCALE_Y

        cursor_pos[0] = int(max(0.0, min(SCREEN_WIDTH - 1.0, cursor_pos[0])))
        cursor_pos[1] = int(max(0.0, min(SCREEN_HEIGHT - 1.0, cursor_pos[1])))


        # --- Game Logic / Experiment State Update ---
        if math.dist(cursor_pos, end_point_screen) < CURSOR_RADIUS * 2:
            print("End reached!")
            running = False


        # --- Calculate Haptic Feedback ---
        drift_active = calculate_haptic_feedback(current_experiment_time, experiment_start_time, apply_drift_this_trial)


        # --- Calculate Experiment Parameters ---
        closest_point_on_path, tracking_error, cross_error = calculate_path_errors(cursor_pos, path_points_screen)

        # --- Log Data ---
        # Log trial number within the batch
        logger.log(trial_num_in_batch, batch_desc, device_state, cursor_pos, closest_point_on_path, tracking_error, cross_error, drift_active)


        # --- Drawing ---
        surface.fill(COLOR_BG)

        if len(path_points_screen) > 1:
            # Draw lines using the dense list of points for the linear path
            # Drawing just between the two original waypoints might look cleaner for a single line
            pygame.draw.line(surface, COLOR_PATH, SIMPLE_PATH_WAYPOINTS_SCREEN[0], SIMPLE_PATH_WAYPOINTS_SCREEN[-1], 3)
            # Or continue drawing the dense points:
            # pygame.draw.lines(surface, COLOR_PATH, False, path_points_screen, 3)


        # Optionally draw original waypoints
        # for waypoint in SIMPLE_PATH_WAYPOINTS_SCREEN:
        #    pygame.draw.circle(surface, COLOR_WAYPOINTS, waypoint, 5)

        pygame.draw.circle(surface, COLOR_START, (int(start_point_screen[0]), int(start_point_screen[1])), CURSOR_RADIUS * 1.5)
        pygame.draw.circle(surface, COLOR_END, (int(end_point_screen[0]), int(end_point_screen[1])), CURSOR_RADIUS * 1.5)

        pygame.draw.circle(surface, COLOR_CURSOR, (int(cursor_pos[0]), int(cursor_pos[1])), CURSOR_RADIUS)

        # --- Event Handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                if event.key == pygame.K_s:
                     print("Skipping trial...")
                     running = False # Skip current trial


        pygame.display.flip()

    # --- End of Trial ---
    print(f"Trial {trial_num_in_batch} in Batch '{batch_desc}' finished.")
    # logger.stop() is called after the batch loop
    pygame.quit() # Quit Pygame for this trial window


# --- Main Execution ---
if __name__ == "__main__":
    # Define the experiment batches: (number_of_trials, apply_drift_flag, batch_description)
    experiment_batches = [
        (10, False, "NoDrift_Batch1"),
        (30, True, "Drift_Batch"),
        (10, False, "NoDrift_Batch2")
    ]

    device = None # Ensure device is None before attempting creation

    try:
        print("Initializing haptic device...")
        # Initialize the haptic device ONCE before all trials
        device_state = DeviceState() # Reset global state for the entire experiment run
        device = HapticDevice(device_name="Default Device", callback=state_callback)
        print("Haptic device initialized.")

        # Initial sleep to allow device and callback to settle before the very first trial
        time.sleep(1.5)

        # Loop through each batch
        for batch_index, (num_trials_in_batch, apply_drift_in_batch, batch_desc) in enumerate(experiment_batches):
            print(f"\n--- Starting Batch {batch_index + 1}: {batch_desc} ({num_trials_in_batch} trials) ---")

            # Create the folder for this batch if it doesn't exist
            batch_folder = os.path.join(BASE_LOG_FOLDER, batch_desc)

            # Define the filename for this batch's data
            batch_log_filename = os.path.join(batch_folder, f"{batch_desc}_data.csv")

            # Initialize a NEW logger for each batch
            logger = HapticDataLogger(batch_log_filename)
            logger.start() # This will create the folder and file, and write the header

            # Loop through trials within the current batch
            for i in range(num_trials_in_batch):
                trial_num_in_batch = i + 1 # Trial number within the current batch
                print(f"\n-- Starting Trial {trial_num_in_batch}/{num_trials_in_batch} (Batch {batch_index + 1}: {batch_desc}) --")

                # Reset device state for each trial (important for initial position tracking)
                device_state.commanded_force = [0, 0, 0]
                device_state.button = False
                # Position and velocity will be updated by the callback
                # initial_haptic_position will be set in run_single_trial

                # Run a single trial
                run_single_trial(trial_num_in_batch, batch_desc, logger, apply_drift_in_batch)

                # Small delay between trials
                time.sleep(1.0) # Pause briefly before the next trial starts

            # Stop the logger for the current batch after all its trials are done
            logger.stop()


    except Exception as e:
        print(f"\nAn error occurred during the experiment: {e}")
        # Optionally log the error or specific state at the time

    finally:
        # Ensure the haptic device is closed ONCE after all batches are done
        if device:
            device.close()
            print("\nHaptic device closed.")
        # Pygame is quit after each trial within run_single_trial


    print("\nProgram finished.")
