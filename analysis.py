#!/usr/bin/env python
""" Haptic Experiment Data Analysis Script (Simplified Impedance Calculation & Ellipsoid Plotting - Drift Batch Only)

This script loads data from the haptic path-following experiment,
performs analysis on performance metrics (tracking error, cross error, velocity),
examines commanded forces for all batches.
It calculates 'apparent' impedance matrices (Stiffness K, Damping D, Inertia M)
ONLY for batches where drift force was applied, based on the simplifying assumption
that F_user = -F_commanded.
It also plots the estimated Stiffness, Damping, and Inertia ellipsoids in 3D,
and their projections onto the XY, XZ, and YZ planes in 2D.
Uses an adjusted plotting method for matrices that may not be positive definite.

It expects data files in subfolders within BASE_LOG_FOLDER, named after batches.

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D # Import for 3D plotting
import os
import math
import statsmodels.api as sm # Import statsmodels for regression

# --- Configuration ---
BASE_LOG_FOLDER = "haptic_experiment_data" # Base folder where batch folders are located (Updated to match latest experiment script)
# Define the batches and their descriptions as used in the experiment script
EXPERIMENT_BATCHES = [
    (10, False, "NoDrift_Batch1"),
    (30, True, "Drift_Batch"),
    (10, False, "NoDrift_Batch2")
]

# --- Analysis Parameters ---
# Time window for calculating average metrics (e.g., exclude initial/final seconds)
ANALYSIS_START_TIME = 0.2 # seconds (allow user to settle on path)
ANALYSIS_END_MARGIN = 0.2 # seconds from end of trial (exclude reaching target)

# Parameters for numerical differentiation (to get acceleration from velocity)
# Be cautious with numerical differentiation as it amplifies noise.
# A small window size or smoothing might be needed.
VELOCITY_SMOOTHING_WINDOW = 11 # Window size for smoothing velocity before calculating acceleration (must be odd)

# Ellipsoid Plotting Parameters
ELLIPSOID_POINTS = 30 # Number of points for drawing the ellipsoid surface


# --- Data Loading and Processing ---
def load_batch_data(base_folder, batches):
    """Loads data from CSV files for each batch."""
    all_batch_data = {}
    for num_trials, apply_drift, batch_desc in batches:
        batch_folder = os.path.join(base_folder, batch_desc)
        batch_filename = os.path.join(batch_folder, f"{batch_desc}_data.csv")

        if not os.path.exists(batch_filename):
            print(f"Warning: Data file not found for batch '{batch_desc}': {batch_filename}")
            all_batch_data[batch_desc] = None
            continue

        print(f"Loading data for batch '{batch_desc}' from {batch_filename}...")
        try:
            # Read the CSV into a pandas DataFrame
            df = pd.read_csv(batch_filename)
            all_batch_data[batch_desc] = df
            print(f"Successfully loaded {len(df)} rows.")
        except Exception as e:
            print(f"Error loading data for batch '{batch_desc}': {e}")
            all_batch_data[batch_desc] = None

    return all_batch_data

def process_trial_data(df_trial):
    """Processes data for a single trial (adds calculated columns)."""
    if df_trial.empty:
        return df_trial.copy() # Return a copy even if empty

    df_processed = df_trial.copy() # Work on a copy

    # Calculate velocity magnitude
    df_processed['DeviceVelMagnitude'] = np.linalg.norm(df_processed[['DeviceVelX', 'DeviceVelY', 'DeviceVelZ']].values, axis=1)

    # Calculate acceleration
    # First, smooth velocity components using a rolling mean
    if len(df_processed) >= VELOCITY_SMOOTHING_WINDOW:
        df_processed['DeviceVelX_smooth'] = df_processed['DeviceVelX'].rolling(window=VELOCITY_SMOOTHING_WINDOW, center=True, min_periods=1).mean()
        df_processed['DeviceVelY_smooth'] = df_processed['DeviceVelY'].rolling(window=VELOCITY_SMOOTHING_WINDOW, center=True, min_periods=1).mean()
        df_processed['DeviceVelZ_smooth'] = df_processed['DeviceVelZ'].rolling(window=VELOCITY_SMOOTHING_WINDOW, center=True, min_periods=1).mean()

        # Calculate acceleration using gradient on smoothed velocity
        dt = df_processed['Timestamp'].diff().mean() # Average time step
        if dt > 0:
            df_processed['DeviceAccX'] = np.gradient(df_processed['DeviceVelX_smooth'], dt)
            df_processed['DeviceAccY'] = np.gradient(df_processed['DeviceVelY_smooth'], dt)
            df_processed['DeviceAccZ'] = np.gradient(df_processed['DeviceVelZ_smooth'], dt)
        else:
             df_processed['DeviceAccX'] = 0
             df_processed['DeviceAccY'] = 0
             df_processed['DeviceAccZ'] = 0
    else:
        # Not enough data points for smoothing window, set acceleration to 0
        df_processed['DeviceVelX_smooth'] = df_processed['DeviceVelX']
        df_processed['DeviceVelY_smooth'] = df_processed['DeviceVelY']
        df_processed['DeviceVelZ_smooth'] = df_processed['DeviceVelZ']
        df_processed['DeviceAccX'] = 0
        df_processed['DeviceAccY'] = 0
        df_processed['DeviceAccZ'] = 0


    df_processed['DeviceAccMagnitude'] = np.linalg.norm(df_processed[['DeviceAccX', 'DeviceAccY', 'DeviceAccZ']].values, axis=1)


    # Calculate assumed user force magnitude
    df_processed['AssumedUserForceMagnitude'] = np.linalg.norm(df_processed[['AssumedUserForceX', 'AssumedUserForceY', 'AssumedUserForceZ']].values, axis=1)

    # Calculate commanded force magnitude
    df_processed['CommandedForceMagnitude'] = np.linalg.norm(df_processed[['CommandedForceX', 'CommandedForceY', 'CommandedForceZ']].values, axis=1)


    return df_processed

# --- Performance Analysis ---
def analyze_performance(batch_data):
    """Calculates average performance metrics per trial and per batch."""
    performance_summary = {} # Stores average metrics per trial for plotting
    batch_averages = {}      # Stores overall average metrics per batch

    for batch_desc, df_batch in batch_data.items():
        if df_batch is None or df_batch.empty:
            performance_summary[batch_desc] = None
            batch_averages[batch_desc] = None
            continue

        # Get unique trial numbers in this batch
        trials_in_batch = df_batch['Trial'].unique()
        trial_metrics = [] # List to store average metrics for each trial

        for trial_num in trials_in_batch:
            df_trial = df_batch[df_batch['Trial'] == trial_num].copy() # Get data for this trial

            # Process trial data (calculate velocity magnitude, acceleration, force magnitudes)
            df_trial = process_trial_data(df_trial)

            # Filter data for analysis window (excluding start and end)
            trial_duration = df_trial['Timestamp'].max()
            analysis_window_df = df_trial[
                (df_trial['Timestamp'] >= ANALYSIS_START_TIME) &
                (df_trial['Timestamp'] <= trial_duration - ANALYSIS_END_MARGIN)
            ].copy() # Work on a copy of the filtered data

            if analysis_window_df.empty:
                 print(f"Warning: Analysis window is empty for Batch '{batch_desc}', Trial {trial_num}. Skipping performance calculation for this trial.")
                 continue

            # Calculate average metrics within the analysis window for this trial
            avg_tracking_error = analysis_window_df['TrackingError'].mean()
            avg_abs_cross_error = analysis_window_df['CrossError'].abs().mean() # Use absolute cross error for average deviation
            avg_velocity_magnitude = analysis_window_df['DeviceVelMagnitude'].mean()
            avg_commanded_force_magnitude = analysis_window_df['CommandedForceMagnitude'].mean()
            avg_assumed_user_force_magnitude = analysis_window_df['AssumedUserForceMagnitude'].mean()


            trial_metrics.append({
                'Trial': trial_num,
                'AvgTrackingError': avg_tracking_error,
                'AvgAbsCrossError': avg_abs_cross_error,
                'AvgVelocityMagnitude': avg_velocity_magnitude,
                'AvgCommandedForceMagnitude': avg_commanded_force_magnitude,
                'AvgAssumedUserForceMagnitude': avg_assumed_user_force_magnitude
            })

        # Convert trial metrics to a DataFrame for easier handling
        if trial_metrics:
            performance_summary[batch_desc] = pd.DataFrame(trial_metrics)

            # Calculate overall batch averages
            batch_averages[batch_desc] = {
                'OverallAvgTrackingError': performance_summary[batch_desc]['AvgTrackingError'].mean(),
                'OverallAvgAbsCrossError': performance_summary[batch_desc]['AvgAbsCrossError'].mean(),
                'OverallAvgVelocityMagnitude': performance_summary[batch_desc]['AvgVelocityMagnitude'].mean(),
                'OverallAvgCommandedForceMagnitude': performance_summary[batch_desc]['AvgCommandedForceMagnitude'].mean(),
                'OverallAvgAssumedUserForceMagnitude': performance_summary[batch_desc]['AvgAssumedUserForceMagnitude'].mean()
            }
        else:
            performance_summary[batch_desc] = None
            batch_averages[batch_desc] = None


    return performance_summary, batch_averages

# --- Visualization ---
def plot_performance_summary(performance_summary):
    """Plots average performance metrics across trials for each batch."""
    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(10, 12), sharex=True)
    fig.suptitle('Performance Metrics Across Trials per Batch')

    metrics_to_plot = {
        'AvgTrackingError': 'Average Tracking Error (pixels)',
        'AvgAbsCrossError': 'Average Absolute Cross Error (pixels)',
        'AvgVelocityMagnitude': 'Average Device Velocity Magnitude (m/s)'
    }

    for i, (metric_key, ylabel) in enumerate(metrics_to_plot.items()):
        ax = axes[i]
        for batch_desc, df_metrics in performance_summary.items():
            if df_metrics is not None and not df_metrics.empty:
                ax.plot(df_metrics['Trial'], df_metrics[metric_key], marker='o', linestyle='-', label=batch_desc)
        ax.set_ylabel(ylabel)
        ax.grid(True)
        ax.legend()

    axes[-1].set_xlabel('Trial Number')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to prevent title overlap
    plt.show()

def plot_force_summary(performance_summary):
    """Plots average force magnitudes across trials for each batch."""
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(10, 8), sharex=True)
    fig.suptitle('Average Force Magnitudes Across Trials per Batch')

    metrics_to_plot = {
        'AvgCommandedForceMagnitude': 'Average Commanded Force Magnitude (N)',
        'AvgAssumedUserForceMagnitude': 'Average Assumed User Force Magnitude (N)'
    }

    for i, (metric_key, ylabel) in enumerate(metrics_to_plot.items()):
        ax = axes[i]
        for batch_desc, df_metrics in performance_summary.items():
            if df_metrics is not None and not df_metrics.empty:
                ax.plot(df_metrics['Trial'], df_metrics[metric_key], marker='o', linestyle='-', label=batch_desc)
        ax.set_ylabel(ylabel)
        ax.grid(True)
        ax.legend()

    axes[-1].set_xlabel('Trial Number')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


def plot_trial_time_series(batch_data, batch_desc, trial_num):
    """Plots time series data for a specific trial."""
    if batch_desc not in batch_data or batch_data[batch_desc] is None:
        print(f"Batch '{batch_desc}' not found or has no data.")
        return

    df_batch = batch_data[batch_desc]
    df_trial = df_batch[df_batch['Trial'] == trial_num].copy()

    if df_trial.empty:
        print(f"Trial {trial_num} not found in batch '{batch_desc}'.")
        return

    df_trial = process_trial_data(df_trial) # Process this trial's data

    fig, axes = plt.subplots(nrows=4, ncols=1, figsize=(10, 12), sharex=True)
    fig.suptitle(f'Time Series Data for Batch: {batch_desc}, Trial {trial_num}')

    # Plot Tracking and Cross Error
    axes[0].plot(df_trial['Timestamp'], df_trial['TrackingError'], label='Tracking Error')
    axes[0].plot(df_trial['Timestamp'], df_trial['CrossError'], label='Cross Error')
    axes[0].set_ylabel('Error (pixels)')
    axes[0].legend()
    axes[0].grid(True)

    # Plot Velocity Magnitude
    axes[1].plot(df_trial['Timestamp'], df_trial['DeviceVelMagnitude'], label='Velocity Magnitude')
    axes[1].set_ylabel('Velocity Magnitude (m/s)')
    axes[1].legend()
    axes[1].grid(True)

    # Plot Commanded Force Components and Magnitude
    axes[2].plot(df_trial['Timestamp'], df_trial['CommandedForceX'], label='Commanded Force X')
    axes[2].plot(df_trial['Timestamp'], df_trial['CommandedForceY'], label='Commanded Force Y')
    axes[2].plot(df_trial['Timestamp'], df_trial['CommandedForceZ'], label='Commanded Force Z')
    axes[2].plot(df_trial['Timestamp'], df_trial['CommandedForceMagnitude'], label='Commanded Force Magnitude', linestyle='--')
    axes[2].set_ylabel('Commanded Force (N)')
    axes[2].legend()
    axes[2].grid(True)

    # Plot Assumed User Force Components and Magnitude
    axes[3].plot(df_trial['Timestamp'], df_trial['AssumedUserForceX'], label='Assumed User Force X')
    axes[3].plot(df_trial['Timestamp'], df_trial['AssumedUserForceY'], label='Assumed User Force Y')
    axes[3].plot(df_trial['Timestamp'], df_trial['AssumedUserForceZ'], label='Assumed User Force Z')
    axes[3].plot(df_trial['Timestamp'], df_trial['AssumedUserForceMagnitude'], label='Assumed User Force Magnitude', linestyle='--')
    axes[3].set_ylabel('Assumed User Force (N)')
    axes[3].set_xlabel('Time (s)')
    axes[3].legend()
    axes[3].grid(True)


    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

# --- Ellipsoid Plotting Function (3D) ---
def plot_ellipsoid_3d(ax, matrix, color, label):
    """
    Plots a 3D ellipsoid based on a symmetric matrix.
    Uses sqrt(abs(eigenvalue)) for semi-axes scaling, suitable for non-positive definite matrices.
    """
    # Ensure the matrix is symmetric
    symmetric_matrix = (matrix + matrix.T) / 2

    # Calculate eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eig(symmetric_matrix)

    # Use sqrt(abs(eigenvalue)) for semi-axes lengths
    # This visualizes directional 'strength' but is not a true impedance ellipsoid if eigenvalues are negative
    semi_axes = np.sqrt(np.abs(eigenvalues))

    # Check for any zero semi-axes (degenerate cases)
    if np.any(semi_axes == 0):
         print(f"Warning: One or more semi-axes for '{label}' are zero. 3D plotting may be degenerate.")


    # Generate points on a unit sphere
    u = np.linspace(0, 2 * np.pi, ELLIPSOID_POINTS)
    v = np.linspace(0, np.pi, ELLIPSOID_POINTS)
    x = 1 * np.outer(np.cos(u), np.sin(v))
    y = 1 * np.outer(np.sin(u), np.sin(v))
    z = 1 * np.outer(np.ones(np.size(u)), np.cos(v))

    # Apply scaling and rotation to transform the unit sphere into the ellipsoid
    # The transformation is eigenvectors @ diag(semi_axes) @ sphere_points
    # We need to reshape the sphere points (x, y, z) into a list of 3D vectors
    sphere_points = np.array([x.flatten(), y.flatten(), z.flatten()])

    # Apply scaling (multiply each row by the corresponding semi-axis length)
    scaled_points = np.diag(semi_axes) @ sphere_points

    # Apply rotation (multiply by the eigenvector matrix)
    ellipsoid_points = eigenvectors @ scaled_points

    # Reshape back to surface format
    x_ellipsoid = ellipsoid_points[0].reshape(x.shape)
    y_ellipsoid = ellipsoid_points[1].reshape(y.shape)
    z_ellipsoid = ellipsoid_points[2].reshape(z.shape)

    # Plot the ellipsoid surface
    ax.plot_surface(x_ellipsoid, y_ellipsoid, z_ellipsoid,  rstride=4, cstride=4, color=color, alpha=0.5)

    # Add a fake point for the legend (surface plots don't automatically get legend entries)
    ax.plot([], [], [], color=color, label=label)

# --- Ellipsoid Plotting Function (2D Projection) ---
def plot_ellipsoid_2d(ax, matrix_2d, color, label, plane):
    """
    Plots a 2D ellipse based on a 2x2 symmetric matrix (projection).
    Uses sqrt(abs(eigenvalue)) for semi-axes scaling.
    """
    # Ensure the matrix is symmetric
    symmetric_matrix_2d = (matrix_2d + matrix_2d.T) / 2

    # Calculate eigenvalues and eigenvectors for the 2x2 matrix
    eigenvalues_2d, eigenvectors_2d = np.linalg.eig(symmetric_matrix_2d)

    # Use sqrt(abs(eigenvalue)) for semi-axes lengths
    semi_axes_2d = np.sqrt(np.abs(eigenvalues_2d))

    # Check for any zero semi-axes (degenerate cases)
    if np.any(semi_axes_2d == 0):
         print(f"Warning: One or more semi-axes for 2D '{label}' ({plane} plane) are zero. Plotting may be degenerate.")


    # Generate points on a unit circle
    theta = np.linspace(0, 2 * np.pi, ELLIPSOID_POINTS)
    circle_points = np.array([np.cos(theta), np.sin(theta)])

    # Apply scaling and rotation to transform the unit circle into the ellipse
    # The transformation is eigenvectors_2d @ diag(semi_axes_2d) @ circle_points
    scaled_points_2d = np.diag(semi_axes_2d) @ circle_points
    ellipse_points_2d = eigenvectors_2d @ scaled_points_2d

    # Plot the ellipse
    ax.plot(ellipse_points_2d[0], ellipse_points_2d[1], color=color, label=label)

    # Set equal aspect ratio
    ax.set_aspect('equal', adjustable='box')
    ax.grid(True)
    ax.set_title(f'Estimated Apparent {label} Ellipse ({plane} Plane)')
    ax.legend()


# --- Simplified Impedance Analysis (Based on Assumption F_user = -F_commanded) ---
def perform_simplified_impedance_analysis(batch_data, experiment_batches_config):
    """
    Performs simplified impedance analysis (K, D, M matrices) for specified batches
    based on the assumption F_user = -F_commanded and plots ellipsoids.
    """
    print("\n--- Simplified Impedance Analysis (Based on F_user = -F_commanded Assumption) ---")
    print("Interpreting these results as true user impedance requires caution.")
    print("They represent the system's apparent impedance under the commanded force.")

    # Create a set of batch descriptions where drift is applied
    drift_batches = {batch_desc for num_trials, apply_drift, batch_desc in experiment_batches_config if apply_drift}

    impedance_matrices_to_plot = {} # Store K, D, and M matrices for plotting ellipsoids

    for batch_desc, df_batch in batch_data.items():
        # Only perform impedance analysis for batches where drift is applied
        if batch_desc not in drift_batches:
            print(f"\nSkipping impedance analysis for batch '{batch_desc}' (No drift applied in this batch).")
            continue

        if df_batch is None or df_batch.empty:
            print(f"\nBatch '{batch_desc}': No data available for impedance analysis.")
            continue

        print(f"\nAnalyzing batch: '{batch_desc}' for impedance")

        # Process all trials in the batch to get acceleration etc.
        df_processed_trials = []
        trials_in_batch = df_batch['Trial'].unique()
        for trial_num in trials_in_batch:
            df_trial = df_batch[df_batch['Trial'] == trial_num].copy()
            df_processed_trials.append(process_trial_data(df_trial))

        if not df_processed_trials:
            print(f"  No valid trials found for impedance analysis in batch '{batch_desc}'.")
            continue

        df_processed_batch = pd.concat(df_processed_trials, ignore_index=True)

        # Filter data for analysis window across all trials in the batch
        analysis_data = []
        for trial_num in trials_in_batch:
            df_trial = df_processed_batch[df_processed_batch['Trial'] == trial_num].copy()
            trial_duration = df_trial['Timestamp'].max()
            analysis_window_df = df_trial[
                (df_trial['Timestamp'] >= ANALYSIS_START_TIME) &
                (df_trial['Timestamp'] <= trial_duration - ANALYSIS_END_MARGIN)
            ].copy() # Work on a copy

            # Ensure required columns exist and are not NaN after filtering/processing
            required_cols = ['AssumedUserForceX', 'AssumedUserForceY', 'AssumedUserForceZ',
                             'DevicePosX', 'DevicePosY', 'DevicePosZ',
                             'DeviceVelX', 'DeviceVelY', 'DeviceVelZ',
                             'DeviceAccX', 'DeviceAccY', 'DeviceAccZ']
            if not all(col in analysis_window_df.columns for col in required_cols):
                 print(f"  Warning: Missing required columns in analysis window for trial {trial_num} in batch '{batch_desc}'. Skipping this trial for impedance analysis.")
                 continue
            if analysis_window_df[required_cols].isnull().any().any():
                 print(f"  Warning: NaN values found in required columns in analysis window for trial {trial_num} in batch '{batch_desc}'. Dropping rows with NaN.")
                 analysis_window_df.dropna(subset=required_cols, inplace=True)


            if not analysis_window_df.empty:
                analysis_data.append(analysis_window_df)

        if not analysis_data:
            print(f"  No data in analysis window for batch '{batch_desc}'. Cannot perform impedance analysis.")
            continue

        df_analysis = pd.concat(analysis_data, ignore_index=True)

        # --- Prepare Data for Regression ---
        # Independent variables (predictors): Position, Velocity, Acceleration
        X = df_analysis[['DevicePosX', 'DevicePosY', 'DevicePosZ',
                         'DeviceVelX', 'DeviceVelY', 'DeviceVelZ',
                         'DeviceAccX', 'DeviceAccY', 'DeviceAccZ']].values

        # Dependent variables (responses): Assumed User Force components
        Y_fx = df_analysis['AssumedUserForceX'].values
        Y_fy = df_analysis['AssumedUserForceY'].values
        Y_fz = df_analysis['AssumedUserForceZ'].values

        # Add a constant (intercept) term to the independent variables
        X = sm.add_constant(X)

        # --- Perform Regression for each Force Component ---
        print("  Performing regression for Fx...")
        model_fx = sm.OLS(Y_fx, X).fit()
        print(model_fx.summary())

        print("\n  Performing regression for Fy...")
        model_fy = sm.OLS(Y_fy, X).fit()
        print(model_fy.summary())

        print("\n  Performing regression for Fz...")
        model_fz = sm.OLS(Y_fz, X).fit()
        print(model_fz.summary())

        # --- Extract Coefficients and Build Matrices ---
        # The coefficients are in the order: [const, PosX, PosY, PosZ, VelX, VelY, VelZ, AccX, AccY, AccZ]
        coef_fx = model_fx.params
        coef_fy = model_fy.params
        coef_fz = model_fz.params

        # Extract bias terms
        bias = np.array([coef_fx[0], coef_fy[0], coef_fz[0]])

        # Extract matrix coefficients (skip the first coefficient which is the constant)
        K_flat = np.array([coef_fx[1:4], coef_fy[1:4], coef_fz[1:4]])
        D_flat = np.array([coef_fx[4:7], coef_fy[4:7], coef_fz[4:7]])
        M_flat = np.array([coef_fx[7:10], coef_fy[7:10], coef_fz[7:10]])

        # Reshape into 3x3 matrices
        K_matrix = K_flat
        D_matrix = D_flat
        M_matrix = M_flat

        # Store matrices for plotting
        impedance_matrices_to_plot[batch_desc] = {'K': K_matrix, 'D': D_matrix, 'M': M_matrix}


        # --- Print Estimated Matrices ---
        print("\n  Estimated Matrices (Based on F_user = -F_commanded Assumption):")
        print("\n  Stiffness Matrix (K):")
        print(K_matrix)
        print("\n  Damping Matrix (D):")
        print(D_matrix)
        print("\n  Inertia Matrix (M):")
        print(M_matrix)
        print("\n  Bias Vector:")
        print(bias)

        print("\n  R-squared values:")
        print(f"    Fx model: {model_fx.rsquared:.4f}")
        print(f"    Fy model: {model_fy.rsquared:.4f}")
        print(f"    Fz model: {model_fz.rsquared:.4f}")

        print("\n  Interpretation Notes:")
        print("  - The diagonal elements (K_xx, D_yy, M_zz, etc.) represent the impedance along each axis.")
        print("  - Off-diagonal elements represent coupling between axes (e.g., K_xy relates Y position to X force).")
        print("  - R-squared indicates how much of the variance in the assumed user force is explained by the model.")
        print("  - These matrices are estimates of the *system's* apparent impedance under the commanded force,")
        print("    not isolated user impedance, due to the simplifying assumption.")

    # --- Plot Ellipsoids for Drift Batches ---
    if impedance_matrices_to_plot:
        print("\nPlotting estimated impedance ellipsoids for drift batches...")

        # 3D Plot
        fig_3d = plt.figure(figsize=(12, 10))
        ax_3d = fig_3d.add_subplot(111, projection='3d')
        ax_3d.set_title('Estimated Apparent Impedance Ellipsoids (Drift Batches) - 3D')
        ax_3d.set_xlabel('X')
        ax_3d.set_ylabel('Y')
        ax_3d.set_zlabel('Z')
        ax_3d.set_box_aspect([1,1,1]) # Equal aspect ratio

        # 2D Plots
        fig_2d_xy, ax_2d_xy = plt.subplots(figsize=(8, 8))
        ax_2d_xy.set_xlabel('X')
        ax_2d_xy.set_ylabel('Y')

        fig_2d_xz, ax_2d_xz = plt.subplots(figsize=(8, 8))
        ax_2d_xz.set_xlabel('X')
        ax_2d_xz.set_ylabel('Z') # Note: Y axis of plot is Z axis of device

        fig_2d_yz, ax_2d_yz = plt.subplots(figsize=(8, 8))
        ax_2d_yz.set_xlabel('Y') # Note: X axis of plot is Y axis of device
        ax_2d_yz.set_ylabel('Z') # Note: Y axis of plot is Z axis of device


        for batch_desc, matrices in impedance_matrices_to_plot.items():
            # Plot Stiffness Ellipsoid (3D)
            plot_ellipsoid_3d(ax_3d, matrices['K'], color='blue', label=f'{batch_desc} - Stiffness')
            # Plot Damping Ellipsoid (3D)
            plot_ellipsoid_3d(ax_3d, matrices['D'], color='red', label=f'{batch_desc} - Damping')
            # Plot Inertia Ellipsoid (3D)
            plot_ellipsoid_3d(ax_3d, matrices['M'], color='green', label=f'{batch_desc} - Inertia')

            # Plot Stiffness Ellipse (2D Projections)
            plot_ellipsoid_2d(ax_2d_xy, matrices['K'][0:2, 0:2], color='blue', label=f'{batch_desc} - Stiffness', plane='XY')
            plot_ellipsoid_2d(ax_2d_xz, np.array([[matrices['K'][0,0], matrices['K'][0,2]], [matrices['K'][2,0], matrices['K'][2,2]]]), color='blue',label=f'{batch_desc} - Stiffness', plane='XZ') # XZ plane
            plot_ellipsoid_2d(ax_2d_yz, np.array([[matrices['K'][1,1], matrices['K'][1,2]], [matrices['K'][2,1], matrices['K'][2,2]]]), color='blue',  label=f'{batch_desc} - Stiffness', plane='YZ') # YZ plane

            # Plot Damping Ellipse (2D Projections)
            plot_ellipsoid_2d(ax_2d_xy, matrices['D'][0:2, 0:2], color='red', label=f'{batch_desc} - Damping', plane='XY')
            plot_ellipsoid_2d(ax_2d_xz, np.array([[matrices['D'][0,0], matrices['D'][0,2]], [matrices['D'][2,0], matrices['D'][2,2]]]), color='red',  label=f'{batch_desc} - Damping', plane='XZ') # XZ plane
            plot_ellipsoid_2d(ax_2d_yz, np.array([[matrices['D'][1,1], matrices['D'][1,2]], [matrices['D'][2,1], matrices['D'][2,2]]]), color='red',  label=f'{batch_desc} - Damping', plane='YZ') # YZ plane

            # Plot Inertia Ellipse (2D Projections)
            plot_ellipsoid_2d(ax_2d_xy, matrices['M'][0:2, 0:2], color='green', label=f'{batch_desc} - Inertia', plane='XY')
            plot_ellipsoid_2d(ax_2d_xz, np.array([[matrices['M'][0,0], matrices['M'][0,2]], [matrices['M'][2,0], matrices['M'][2,2]]]), color='green',  label=f'{batch_desc} - Inertia', plane='XZ') # XZ plane
            plot_ellipsoid_2d(ax_2d_yz, np.array([[matrices['M'][1,1], matrices['M'][1,2]], [matrices['M'][2,1], matrices['M'][2,2]]]), color='green',  label=f'{batch_desc} - Inertia', plane='YZ') # YZ plane


        # Show all plots
        ax_3d.legend() # Add legend to 3D plot
        plt.show() # This will show all generated figures


    else:
        print("\nNo impedance matrices calculated for plotting ellipsoids.")


# --- Main Analysis Execution ---
if __name__ == "__main__":
    # Load data for all batches
    batch_data = load_batch_data(BASE_LOG_FOLDER, EXPERIMENT_BATCHES)

    # Analyze performance metrics across batches and trials
    performance_summary, batch_averages = analyze_performance(batch_data)

    # Print overall batch averages
    print("\n--- Overall Batch Averages (within analysis window) ---")
    for batch_desc, averages in batch_averages.items():
        if averages:
            print(f"\nBatch: {batch_desc}")
            for key, value in averages.items():
                print(f"  {key}: {value:.4f}")
        else:
            print(f"\nBatch: {batch_desc} - No data for averages.")

    # Plot performance summary across trials
    print("\nPlotting performance summary...")
    plot_performance_summary(performance_summary)

    # Plot force summary across trials
    print("\nPlotting force summary...")
    plot_force_summary(performance_summary)

    # --- Optional: Plot time series for a specific trial ---
    # You can uncomment and modify these lines to visualize individual trials
    # print("\nPlotting time series for a sample trial...")
    # sample_batch_desc = "Drift_Batch" # Choose a batch
    # sample_trial_num = 1 # Choose a trial number within that batch
    # plot_trial_time_series(batch_data, sample_batch_desc, sample_trial_num)

    # --- Perform Simplified Impedance Analysis (Only for Drift Batches) ---
    # This will calculate K, D, M matrices only for batches where drift was applied and plot ellipsoids.
    perform_simplified_impedance_analysis(batch_data, EXPERIMENT_BATCHES)


    print("\nAnalysis script finished.")
