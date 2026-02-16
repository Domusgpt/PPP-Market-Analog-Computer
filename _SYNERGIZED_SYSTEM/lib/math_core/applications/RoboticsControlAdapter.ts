/**
 * Robotics Control Adapter
 *
 * Maps IMU sensor data to the Chronomorphic Engine and provides
 * control outputs for robotic systems. Features:
 *
 * - 6-axis IMU → 6D rotation mapping (accelerometer + gyroscope)
 * - Invariant plane definition for stable reference frames
 * - Gyro drift modeling as perturbation matrix
 * - Quaternion-based orientation tracking
 * - Smooth trajectory generation
 *
 * Applications: Drone control, robotic arms, mobile robots,
 * balance systems, motion capture.
 *
 * Ported from CPE repo (Domusgpt/-Chronomorphic-Polytopal-Engine)
 * with import paths adapted for PPP math_core layout.
 */

import type { Vector4D, Bivector4D } from '../geometric_algebra/types';
import {
  rotorInPlane,
  applyRotorToVector,
  combinedRotationMatrix,
  matrixVectorMultiply,
  type Rotor
} from '../geometric_algebra/GeometricAlgebra';

// =============================================================================
// TYPES
// =============================================================================

/** 3D vector */
export type Vector3D = [number, number, number];

/** IMU sensor data */
export interface IMUData {
  /** Accelerometer reading [x, y, z] in m/s² */
  readonly accelerometer: Vector3D;
  /** Gyroscope reading [x, y, z] in rad/s */
  readonly gyroscope: Vector3D;
  /** Optional magnetometer [x, y, z] in μT */
  readonly magnetometer?: Vector3D;
  /** Timestamp in ms */
  readonly timestamp: number;
}

/** Quaternion for 3D rotation */
export interface Quaternion {
  readonly w: number;
  readonly x: number;
  readonly y: number;
  readonly z: number;
}

/** Robot pose in 3D space */
export interface RobotPose {
  /** Position [x, y, z] */
  readonly position: Vector3D;
  /** Orientation as quaternion */
  readonly orientation: Quaternion;
  /** Linear velocity [vx, vy, vz] */
  readonly linearVelocity: Vector3D;
  /** Angular velocity [wx, wy, wz] */
  readonly angularVelocity: Vector3D;
}

/** Control output for motors/actuators */
export interface ControlOutput {
  /** Motor/actuator values (normalized -1 to 1) */
  readonly motorValues: number[];
  /** Target orientation quaternion */
  readonly targetOrientation: Quaternion;
  /** Target angular velocity */
  readonly targetAngularVelocity: Vector3D;
  /** Control mode */
  readonly mode: ControlMode;
  /** Stability metric (0-1) */
  readonly stability: number;
  /** Timestamp */
  readonly timestamp: number;
}

/** Control modes */
export type ControlMode = 'STABILIZE' | 'RATE' | 'POSITION' | 'TRAJECTORY' | 'MANUAL';

/** Trajectory waypoint */
export interface Waypoint {
  /** Target position */
  readonly position: Vector3D;
  /** Target orientation */
  readonly orientation: Quaternion;
  /** Time to reach (ms) */
  readonly duration: number;
  /** Interpolation type */
  readonly interpolation: 'linear' | 'spline' | 'slerp';
}

/** Drift correction parameters */
export interface DriftCorrection {
  /** Gyro bias estimate [x, y, z] */
  readonly gyroBias: Vector3D;
  /** Accelerometer bias [x, y, z] */
  readonly accelBias: Vector3D;
  /** Correction gain */
  readonly correctionGain: number;
  /** Last correction timestamp */
  readonly lastCorrectionTime: number;
}

/** Adapter configuration */
export interface RoboticsConfig {
  /** Number of motors/actuators */
  motorCount: number;
  /** Sample rate (Hz) */
  sampleRate: number;
  /** Control loop rate (Hz) */
  controlRate: number;
  /** Enable drift correction */
  enableDriftCorrection: boolean;
  /** Drift correction interval (ms) */
  driftCorrectionInterval: number;
  /** PID gains [Kp, Ki, Kd] for attitude */
  attitudePID: [number, number, number];
  /** PID gains [Kp, Ki, Kd] for rate */
  ratePID: [number, number, number];
  /** Motor mixing matrix (4xN for quadcopter, etc.) */
  mixingMatrix?: number[][];
  /** Gravity vector */
  gravity: Vector3D;
}

// =============================================================================
// DEFAULTS
// =============================================================================

export const DEFAULT_ROBOTICS_CONFIG: RoboticsConfig = {
  motorCount: 4,
  sampleRate: 100,
  controlRate: 50,
  enableDriftCorrection: true,
  driftCorrectionInterval: 1000,
  attitudePID: [4.0, 0.02, 0.5],
  ratePID: [0.15, 0.002, 0.01],
  gravity: [0, 0, -9.81]
};

/** Quadcopter mixing matrix (X configuration) */
export const QUADCOPTER_X_MIXING: number[][] = [
  [ 1, -1,  1,  1], // Front-right
  [ 1,  1, -1,  1], // Front-left
  [ 1,  1,  1, -1], // Rear-left
  [ 1, -1, -1, -1]  // Rear-right
];

// =============================================================================
// ROBOTICS CONTROL ADAPTER
// =============================================================================

/**
 * Robotics Control Adapter.
 *
 * Integrates IMU data with the Chronomorphic Engine for robotic control.
 */
export class RoboticsControlAdapter {
  private _config: RoboticsConfig;
  private _pose: RobotPose;
  private _targetPose: RobotPose;
  private _driftCorrection: DriftCorrection;
  private _integralError: Vector3D;
  private _lastError: Vector3D;
  private _lastTimestamp: number;
  private _trajectory: Waypoint[];
  private _trajectoryIndex: number;
  private _controlMode: ControlMode;
  private _position4D: Vector4D;
  private _imuHistory: IMUData[];

  constructor(config: Partial<RoboticsConfig> = {}) {
    this._config = { ...DEFAULT_ROBOTICS_CONFIG, ...config };

    // Initialize pose
    this._pose = {
      position: [0, 0, 0],
      orientation: { w: 1, x: 0, y: 0, z: 0 },
      linearVelocity: [0, 0, 0],
      angularVelocity: [0, 0, 0]
    };

    this._targetPose = { ...this._pose };

    this._driftCorrection = {
      gyroBias: [0, 0, 0],
      accelBias: [0, 0, 0],
      correctionGain: 0.01,
      lastCorrectionTime: 0
    };

    this._integralError = [0, 0, 0];
    this._lastError = [0, 0, 0];
    this._lastTimestamp = 0;
    this._trajectory = [];
    this._trajectoryIndex = 0;
    this._controlMode = 'STABILIZE';
    this._position4D = [0, 0, 0, 1];
    this._imuHistory = [];
  }

  // ===========================================================================
  // IMU PROCESSING
  // ===========================================================================

  /**
   * Process IMU data and update pose estimate.
   */
  processIMU(imu: IMUData): void {
    // Calculate delta time
    const dt = this._lastTimestamp > 0
      ? (imu.timestamp - this._lastTimestamp) / 1000
      : 1 / this._config.sampleRate;

    this._lastTimestamp = imu.timestamp;

    // Apply drift correction
    const correctedGyro = this._applyDriftCorrection(imu.gyroscope);

    // Update orientation using gyroscope
    const dq = this._gyroToQuaternionDelta(correctedGyro, dt);
    const newOrientation = this._multiplyQuaternions(this._pose.orientation, dq);
    const normalizedOrientation = this._normalizeQuaternion(newOrientation);

    // Complementary filter with accelerometer
    if (this._config.enableDriftCorrection) {
      const accelOrientation = this._accelToOrientation(imu.accelerometer);
      const fusedOrientation = this._complementaryFilter(
        normalizedOrientation,
        accelOrientation,
        0.98 // Alpha: trust gyro 98%, accelerometer 2%
      );
      this._pose = { ...this._pose, orientation: fusedOrientation };
    } else {
      this._pose = { ...this._pose, orientation: normalizedOrientation };
    }

    // Update angular velocity
    this._pose = {
      ...this._pose,
      angularVelocity: correctedGyro
    };

    // Update linear velocity from accelerometer (in world frame)
    const worldAccel = this._rotateVectorByQuaternion(
      imu.accelerometer,
      this._pose.orientation
    );

    // Remove gravity
    const linearAccel: Vector3D = [
      worldAccel[0] - this._config.gravity[0],
      worldAccel[1] - this._config.gravity[1],
      worldAccel[2] - this._config.gravity[2]
    ];

    // Integrate acceleration to get velocity (with damping)
    const damping = 0.99;
    this._pose = {
      ...this._pose,
      linearVelocity: [
        (this._pose.linearVelocity[0] + linearAccel[0] * dt) * damping,
        (this._pose.linearVelocity[1] + linearAccel[1] * dt) * damping,
        (this._pose.linearVelocity[2] + linearAccel[2] * dt) * damping
      ]
    };

    // Integrate velocity to get position
    this._pose = {
      ...this._pose,
      position: [
        this._pose.position[0] + this._pose.linearVelocity[0] * dt,
        this._pose.position[1] + this._pose.linearVelocity[1] * dt,
        this._pose.position[2] + this._pose.linearVelocity[2] * dt
      ]
    };

    // Update 4D position for polytope navigation
    this._position4D = this._poseToPosition4D(this._pose);

    // Store history
    this._imuHistory.push(imu);
    if (this._imuHistory.length > 100) {
      this._imuHistory.shift();
    }

    // Periodic drift correction update
    if (imu.timestamp - this._driftCorrection.lastCorrectionTime >
        this._config.driftCorrectionInterval) {
      this._updateDriftEstimate();
    }
  }

  /**
   * Apply gyro drift correction.
   */
  private _applyDriftCorrection(gyro: Vector3D): Vector3D {
    return [
      gyro[0] - this._driftCorrection.gyroBias[0],
      gyro[1] - this._driftCorrection.gyroBias[1],
      gyro[2] - this._driftCorrection.gyroBias[2]
    ];
  }

  /**
   * Update drift estimate from recent history.
   */
  private _updateDriftEstimate(): void {
    if (this._imuHistory.length < 10) return;

    // Check if stationary (low accelerometer variance)
    const recentAccel = this._imuHistory.slice(-10).map(i => i.accelerometer);
    const variance = this._calculateVariance(recentAccel);

    if (variance < 0.5) {
      // Stationary - update gyro bias
      const recentGyro = this._imuHistory.slice(-10).map(i => i.gyroscope);
      const meanGyro = this._calculateMean(recentGyro);

      this._driftCorrection = {
        ...this._driftCorrection,
        gyroBias: [
          this._driftCorrection.gyroBias[0] +
            this._driftCorrection.correctionGain * meanGyro[0],
          this._driftCorrection.gyroBias[1] +
            this._driftCorrection.correctionGain * meanGyro[1],
          this._driftCorrection.gyroBias[2] +
            this._driftCorrection.correctionGain * meanGyro[2]
        ],
        lastCorrectionTime: this._lastTimestamp
      };
    }
  }

  // ===========================================================================
  // CONTROL
  // ===========================================================================

  /**
   * Compute control output.
   */
  computeControl(): ControlOutput {
    const timestamp = this._lastTimestamp;

    // Update target based on mode
    this._updateTarget(timestamp);

    // Compute attitude error
    const attitudeError = this._computeAttitudeError();

    // Compute rate error
    const rateError = this._computeRateError();

    // PID control
    const dt = 1 / this._config.controlRate;
    const controlSignal = this._computePID(attitudeError, rateError, dt);

    // Apply motor mixing
    const motorValues = this._applyMixing(controlSignal);

    // Compute stability metric
    const stability = this._computeStability(attitudeError);

    return {
      motorValues,
      targetOrientation: this._targetPose.orientation,
      targetAngularVelocity: this._targetPose.angularVelocity,
      mode: this._controlMode,
      stability,
      timestamp
    };
  }

  /**
   * Update target based on control mode.
   */
  private _updateTarget(timestamp: number): void {
    switch (this._controlMode) {
      case 'TRAJECTORY':
        this._updateTrajectoryTarget(timestamp);
        break;
      case 'STABILIZE':
        this._targetPose = {
          ...this._targetPose,
          orientation: { w: 1, x: 0, y: 0, z: 0 },
          angularVelocity: [0, 0, 0]
        };
        break;
      case 'RATE':
        this._targetPose = {
          ...this._targetPose,
          orientation: this._pose.orientation
        };
        break;
    }
  }

  /**
   * Update target from trajectory.
   */
  private _updateTrajectoryTarget(timestamp: number): void {
    if (this._trajectory.length === 0) {
      this._controlMode = 'STABILIZE';
      return;
    }

    const waypoint = this._trajectory[this._trajectoryIndex];

    // Check if waypoint reached
    const posError = this._vectorDistance(
      this._pose.position,
      waypoint.position
    );

    if (posError < 0.1) {
      this._trajectoryIndex++;
      if (this._trajectoryIndex >= this._trajectory.length) {
        this._trajectoryIndex = 0;
        this._controlMode = 'STABILIZE';
        return;
      }
    }

    this._targetPose = {
      position: waypoint.position,
      orientation: waypoint.orientation,
      linearVelocity: [0, 0, 0],
      angularVelocity: [0, 0, 0]
    };
  }

  /**
   * Compute attitude error quaternion.
   */
  private _computeAttitudeError(): Vector3D {
    const conj = this._conjugateQuaternion(this._pose.orientation);
    const error = this._multiplyQuaternions(this._targetPose.orientation, conj);

    const angle = 2 * Math.acos(Math.min(1, Math.abs(error.w)));
    const s = Math.sqrt(1 - error.w * error.w);

    if (s < 0.001) {
      return [0, 0, 0];
    }

    return [
      (error.x / s) * angle,
      (error.y / s) * angle,
      (error.z / s) * angle
    ];
  }

  /**
   * Compute rate error.
   */
  private _computeRateError(): Vector3D {
    return [
      this._targetPose.angularVelocity[0] - this._pose.angularVelocity[0],
      this._targetPose.angularVelocity[1] - this._pose.angularVelocity[1],
      this._targetPose.angularVelocity[2] - this._pose.angularVelocity[2]
    ];
  }

  /**
   * PID controller.
   */
  private _computePID(
    attitudeError: Vector3D,
    rateError: Vector3D,
    dt: number
  ): Vector3D {
    const [Kp_att, Ki_att, Kd_att] = this._config.attitudePID;
    const [Kp_rate, Ki_rate, Kd_rate] = this._config.ratePID;

    // Attitude loop output becomes rate target
    const rateTarget: Vector3D = [
      Kp_att * attitudeError[0],
      Kp_att * attitudeError[1],
      Kp_att * attitudeError[2]
    ];

    // Rate loop
    const effectiveRateError: Vector3D = [
      rateTarget[0] - this._pose.angularVelocity[0],
      rateTarget[1] - this._pose.angularVelocity[1],
      rateTarget[2] - this._pose.angularVelocity[2]
    ];

    // Update integral
    this._integralError = [
      this._integralError[0] + effectiveRateError[0] * dt,
      this._integralError[1] + effectiveRateError[1] * dt,
      this._integralError[2] + effectiveRateError[2] * dt
    ];

    // Clamp integral
    const maxIntegral = 0.5;
    this._integralError = this._integralError.map(
      i => Math.max(-maxIntegral, Math.min(maxIntegral, i))
    ) as Vector3D;

    // Derivative
    const derivative: Vector3D = [
      (effectiveRateError[0] - this._lastError[0]) / dt,
      (effectiveRateError[1] - this._lastError[1]) / dt,
      (effectiveRateError[2] - this._lastError[2]) / dt
    ];

    this._lastError = effectiveRateError;

    // PID output
    return [
      Kp_rate * effectiveRateError[0] +
        Ki_rate * this._integralError[0] +
        Kd_rate * derivative[0],
      Kp_rate * effectiveRateError[1] +
        Ki_rate * this._integralError[1] +
        Kd_rate * derivative[1],
      Kp_rate * effectiveRateError[2] +
        Ki_rate * this._integralError[2] +
        Kd_rate * derivative[2]
    ];
  }

  /**
   * Apply motor mixing matrix.
   */
  private _applyMixing(control: Vector3D): number[] {
    const mixing = this._config.mixingMatrix || QUADCOPTER_X_MIXING;
    const n = this._config.motorCount;

    const throttle = 0.5;
    const input = [throttle, control[0], control[1], control[2]];

    const motors: number[] = [];
    for (let i = 0; i < n; i++) {
      let value = 0;
      for (let j = 0; j < Math.min(4, mixing[i]?.length || 0); j++) {
        value += mixing[i][j] * input[j];
      }
      motors.push(Math.max(-1, Math.min(1, value)));
    }

    return motors;
  }

  /**
   * Compute stability metric.
   */
  private _computeStability(attitudeError: Vector3D): number {
    const errorMag = Math.sqrt(
      attitudeError[0]**2 + attitudeError[1]**2 + attitudeError[2]**2
    );
    return Math.max(0, 1 - errorMag / Math.PI);
  }

  // ===========================================================================
  // 4D INTEGRATION
  // ===========================================================================

  /**
   * Convert pose to 4D position for polytope navigation.
   */
  private _poseToPosition4D(pose: RobotPose): Vector4D {
    const q = pose.orientation;
    const w = pose.angularVelocity;

    const angularMag = Math.sqrt(w[0]**2 + w[1]**2 + w[2]**2);

    return [
      q.x + w[0] * 0.1,
      q.y + w[1] * 0.1,
      q.z + w[2] * 0.1,
      q.w * (1 - angularMag * 0.05)
    ];
  }

  /**
   * Get current 4D position.
   */
  getPosition4D(): Vector4D {
    return [...this._position4D] as Vector4D;
  }

  /**
   * Get 6D rotation as bivector (for GA operations).
   */
  getRotation6D(): Bivector4D {
    const [wx, wy, wz] = this._pose.angularVelocity;

    return [
      wx, // XY plane
      wy, // XZ plane
      0,  // XW plane
      wz, // YZ plane
      0,  // YW plane
      0   // ZW plane
    ];
  }

  // ===========================================================================
  // API
  // ===========================================================================

  /**
   * Set control mode.
   */
  setMode(mode: ControlMode): void {
    this._controlMode = mode;
    this._integralError = [0, 0, 0];
    this._lastError = [0, 0, 0];
  }

  /**
   * Set target pose.
   */
  setTarget(pose: Partial<RobotPose>): void {
    this._targetPose = { ...this._targetPose, ...pose };
  }

  /**
   * Set trajectory.
   */
  setTrajectory(waypoints: Waypoint[]): void {
    this._trajectory = waypoints;
    this._trajectoryIndex = 0;
    this._controlMode = 'TRAJECTORY';
  }

  /**
   * Get current pose.
   */
  getPose(): RobotPose {
    return { ...this._pose };
  }

  /**
   * Reset adapter.
   */
  reset(): void {
    this._pose = {
      position: [0, 0, 0],
      orientation: { w: 1, x: 0, y: 0, z: 0 },
      linearVelocity: [0, 0, 0],
      angularVelocity: [0, 0, 0]
    };
    this._targetPose = { ...this._pose };
    this._integralError = [0, 0, 0];
    this._lastError = [0, 0, 0];
    this._position4D = [0, 0, 0, 1];
    this._imuHistory = [];
    this._controlMode = 'STABILIZE';
  }

  // ===========================================================================
  // QUATERNION HELPERS
  // ===========================================================================

  private _gyroToQuaternionDelta(gyro: Vector3D, dt: number): Quaternion {
    const [wx, wy, wz] = gyro;
    const halfDt = dt / 2;

    return {
      w: 1,
      x: wx * halfDt,
      y: wy * halfDt,
      z: wz * halfDt
    };
  }

  private _multiplyQuaternions(a: Quaternion, b: Quaternion): Quaternion {
    return {
      w: a.w*b.w - a.x*b.x - a.y*b.y - a.z*b.z,
      x: a.w*b.x + a.x*b.w + a.y*b.z - a.z*b.y,
      y: a.w*b.y - a.x*b.z + a.y*b.w + a.z*b.x,
      z: a.w*b.z + a.x*b.y - a.y*b.x + a.z*b.w
    };
  }

  private _normalizeQuaternion(q: Quaternion): Quaternion {
    const norm = Math.sqrt(q.w*q.w + q.x*q.x + q.y*q.y + q.z*q.z);
    return {
      w: q.w / norm,
      x: q.x / norm,
      y: q.y / norm,
      z: q.z / norm
    };
  }

  private _conjugateQuaternion(q: Quaternion): Quaternion {
    return { w: q.w, x: -q.x, y: -q.y, z: -q.z };
  }

  private _rotateVectorByQuaternion(v: Vector3D, q: Quaternion): Vector3D {
    const vq: Quaternion = { w: 0, x: v[0], y: v[1], z: v[2] };
    const conj = this._conjugateQuaternion(q);
    const result = this._multiplyQuaternions(
      this._multiplyQuaternions(q, vq),
      conj
    );
    return [result.x, result.y, result.z];
  }

  private _accelToOrientation(accel: Vector3D): Quaternion {
    const norm = Math.sqrt(accel[0]**2 + accel[1]**2 + accel[2]**2);
    if (norm < 0.01) return { w: 1, x: 0, y: 0, z: 0 };

    const nx = accel[0] / norm;
    const ny = accel[1] / norm;
    const nz = accel[2] / norm;

    const roll = Math.atan2(ny, nz);
    const pitch = Math.atan2(-nx, Math.sqrt(ny*ny + nz*nz));

    const cr = Math.cos(roll / 2);
    const sr = Math.sin(roll / 2);
    const cp = Math.cos(pitch / 2);
    const sp = Math.sin(pitch / 2);

    return {
      w: cr * cp,
      x: sr * cp,
      y: cr * sp,
      z: -sr * sp
    };
  }

  private _complementaryFilter(
    gyroQ: Quaternion,
    accelQ: Quaternion,
    alpha: number
  ): Quaternion {
    const dot = gyroQ.w*accelQ.w + gyroQ.x*accelQ.x +
                gyroQ.y*accelQ.y + gyroQ.z*accelQ.z;

    const t = 1 - alpha;

    if (dot < 0) {
      return this._normalizeQuaternion({
        w: gyroQ.w * alpha - accelQ.w * t,
        x: gyroQ.x * alpha - accelQ.x * t,
        y: gyroQ.y * alpha - accelQ.y * t,
        z: gyroQ.z * alpha - accelQ.z * t
      });
    }

    return this._normalizeQuaternion({
      w: gyroQ.w * alpha + accelQ.w * t,
      x: gyroQ.x * alpha + accelQ.x * t,
      y: gyroQ.y * alpha + accelQ.y * t,
      z: gyroQ.z * alpha + accelQ.z * t
    });
  }

  // ===========================================================================
  // UTILITY HELPERS
  // ===========================================================================

  private _calculateVariance(vectors: Vector3D[]): number {
    const mean = this._calculateMean(vectors);
    let variance = 0;

    for (const v of vectors) {
      variance += (v[0] - mean[0])**2 + (v[1] - mean[1])**2 + (v[2] - mean[2])**2;
    }

    return variance / vectors.length;
  }

  private _calculateMean(vectors: Vector3D[]): Vector3D {
    const sum: Vector3D = [0, 0, 0];
    for (const v of vectors) {
      sum[0] += v[0];
      sum[1] += v[1];
      sum[2] += v[2];
    }
    const n = vectors.length || 1;
    return [sum[0] / n, sum[1] / n, sum[2] / n];
  }

  private _vectorDistance(a: Vector3D, b: Vector3D): number {
    return Math.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2 + (a[2]-b[2])**2);
  }
}

// =============================================================================
// EXPORTS
// =============================================================================

export default RoboticsControlAdapter;
