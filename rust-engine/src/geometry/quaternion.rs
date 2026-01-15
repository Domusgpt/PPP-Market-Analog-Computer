//! Quaternion implementation for 4D rotations
//!
//! In 4D space, rotations are more complex than in 3D. The rotation group SO(4)
//! factors into two independent 3D rotations: SO(4) ≅ SO(3) × SO(3).
//! We use pairs of quaternions (left and right) to represent general 4D rotations.
//!
//! Isoclinic rotations (where both rotations have the same angle) are particularly
//! important - iterating an isoclinic rotation of a 24-cell generates the 600-cell.

use super::Vec4;
use std::ops::{Add, Mul, Neg};
use serde::{Serialize, Deserialize};

/// A quaternion q = w + xi + yj + zk
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Quaternion {
    pub w: f64, // scalar (real) part
    pub x: f64, // i component
    pub y: f64, // j component
    pub z: f64, // k component
}

impl Quaternion {
    /// Create a new quaternion
    pub const fn new(w: f64, x: f64, y: f64, z: f64) -> Self {
        Self { w, x, y, z }
    }

    /// Identity quaternion (no rotation)
    pub const fn identity() -> Self {
        Self::new(1.0, 0.0, 0.0, 0.0)
    }

    /// Zero quaternion
    pub const fn zero() -> Self {
        Self::new(0.0, 0.0, 0.0, 0.0)
    }

    /// Create from axis-angle representation (3D rotation)
    pub fn from_axis_angle(axis: [f64; 3], angle: f64) -> Self {
        let half_angle = angle / 2.0;
        let s = half_angle.sin();
        let c = half_angle.cos();
        let mag = (axis[0] * axis[0] + axis[1] * axis[1] + axis[2] * axis[2]).sqrt();

        if mag < 1e-10 {
            return Self::identity();
        }

        Self::new(
            c,
            axis[0] / mag * s,
            axis[1] / mag * s,
            axis[2] / mag * s,
        )
    }

    /// Create quaternion for rotation in XY plane (around ZW)
    pub fn rotation_xy(angle: f64) -> Self {
        Self::from_axis_angle([0.0, 0.0, 1.0], angle)
    }

    /// Create quaternion for rotation in XZ plane (around YW)
    pub fn rotation_xz(angle: f64) -> Self {
        Self::from_axis_angle([0.0, 1.0, 0.0], -angle)
    }

    /// Create quaternion for rotation in YZ plane (around XW)
    pub fn rotation_yz(angle: f64) -> Self {
        Self::from_axis_angle([1.0, 0.0, 0.0], angle)
    }

    /// Squared magnitude
    pub fn magnitude_squared(self) -> f64 {
        self.w * self.w + self.x * self.x + self.y * self.y + self.z * self.z
    }

    /// Magnitude (norm)
    pub fn magnitude(self) -> f64 {
        self.magnitude_squared().sqrt()
    }

    /// Normalize to unit quaternion
    pub fn normalize(self) -> Self {
        let mag = self.magnitude();
        if mag > 1e-10 {
            Self::new(
                self.w / mag,
                self.x / mag,
                self.y / mag,
                self.z / mag,
            )
        } else {
            Self::identity()
        }
    }

    /// Conjugate (w, -x, -y, -z)
    pub fn conjugate(self) -> Self {
        Self::new(self.w, -self.x, -self.y, -self.z)
    }

    /// Inverse (for unit quaternions, this equals conjugate)
    pub fn inverse(self) -> Self {
        let mag_sq = self.magnitude_squared();
        if mag_sq > 1e-10 {
            let inv_mag_sq = 1.0 / mag_sq;
            Self::new(
                self.w * inv_mag_sq,
                -self.x * inv_mag_sq,
                -self.y * inv_mag_sq,
                -self.z * inv_mag_sq,
            )
        } else {
            Self::identity()
        }
    }

    /// Quaternion multiplication (Hamilton product)
    pub fn mul(self, other: Self) -> Self {
        Self::new(
            self.w * other.w - self.x * other.x - self.y * other.y - self.z * other.z,
            self.w * other.x + self.x * other.w + self.y * other.z - self.z * other.y,
            self.w * other.y - self.x * other.z + self.y * other.w + self.z * other.x,
            self.w * other.z + self.x * other.y - self.y * other.x + self.z * other.w,
        )
    }

    /// Rotate a 3D point by this quaternion: q * p * q⁻¹
    pub fn rotate_point_3d(self, point: [f64; 3]) -> [f64; 3] {
        let p = Self::new(0.0, point[0], point[1], point[2]);
        let result = self.mul(p).mul(self.conjugate());
        [result.x, result.y, result.z]
    }

    /// Spherical linear interpolation (SLERP)
    pub fn slerp(self, other: Self, t: f64) -> Self {
        let mut dot = self.w * other.w + self.x * other.x + self.y * other.y + self.z * other.z;

        // If dot is negative, negate one quaternion to take shorter path
        let other = if dot < 0.0 {
            dot = -dot;
            -other
        } else {
            other
        };

        // Use linear interpolation for nearly identical quaternions
        if dot > 0.9995 {
            return Self::new(
                self.w + (other.w - self.w) * t,
                self.x + (other.x - self.x) * t,
                self.y + (other.y - self.y) * t,
                self.z + (other.z - self.z) * t,
            ).normalize();
        }

        let theta = dot.acos();
        let sin_theta = theta.sin();
        let t0 = ((1.0 - t) * theta).sin() / sin_theta;
        let t1 = (t * theta).sin() / sin_theta;

        Self::new(
            self.w * t0 + other.w * t1,
            self.x * t0 + other.x * t1,
            self.y * t0 + other.y * t1,
            self.z * t0 + other.z * t1,
        )
    }

    /// Convert to axis-angle representation
    pub fn to_axis_angle(self) -> ([f64; 3], f64) {
        let q = self.normalize();
        let angle = 2.0 * q.w.acos();

        if angle.abs() < 1e-10 {
            return ([1.0, 0.0, 0.0], 0.0);
        }

        let s = (1.0 - q.w * q.w).sqrt();
        if s < 1e-10 {
            return ([1.0, 0.0, 0.0], angle);
        }

        ([q.x / s, q.y / s, q.z / s], angle)
    }
}

impl Default for Quaternion {
    fn default() -> Self {
        Self::identity()
    }
}

impl Add for Quaternion {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        Self::new(
            self.w + other.w,
            self.x + other.x,
            self.y + other.y,
            self.z + other.z,
        )
    }
}

impl Mul for Quaternion {
    type Output = Self;

    fn mul(self, other: Self) -> Self {
        self.mul(other)
    }
}

impl Mul<f64> for Quaternion {
    type Output = Self;

    fn mul(self, scalar: f64) -> Self {
        Self::new(
            self.w * scalar,
            self.x * scalar,
            self.y * scalar,
            self.z * scalar,
        )
    }
}

impl Neg for Quaternion {
    type Output = Self;

    fn neg(self) -> Self {
        Self::new(-self.w, -self.x, -self.y, -self.z)
    }
}

/// 4D rotation represented as a pair of quaternions (left, right)
/// A 4D rotation acts on a point p as: R(p) = L * p * R^(-1)
/// where p is treated as a quaternion
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Rotation4D {
    pub left: Quaternion,
    pub right: Quaternion,
}

impl Rotation4D {
    /// Create a new 4D rotation from left and right quaternions
    pub fn new(left: Quaternion, right: Quaternion) -> Self {
        Self {
            left: left.normalize(),
            right: right.normalize(),
        }
    }

    /// Identity rotation (no rotation)
    pub fn identity() -> Self {
        Self::new(Quaternion::identity(), Quaternion::identity())
    }

    /// Create an isoclinic rotation (same angle in two orthogonal planes)
    /// This is the type of rotation that, when iterated, generates higher polytopes
    pub fn isoclinic(q: Quaternion) -> Self {
        Self::new(q, q)
    }

    /// Create a simple rotation in the XY plane
    pub fn simple_xy(angle: f64) -> Self {
        let q = Quaternion::from_axis_angle([0.0, 0.0, 1.0], angle / 2.0);
        Self::new(q, q.conjugate())
    }

    /// Create a simple rotation in the XZ plane
    pub fn simple_xz(angle: f64) -> Self {
        let q = Quaternion::from_axis_angle([0.0, 1.0, 0.0], angle / 2.0);
        Self::new(q, q.conjugate())
    }

    /// Create a simple rotation in the XW plane
    pub fn simple_xw(angle: f64) -> Self {
        let half = angle / 2.0;
        let c = half.cos();
        let s = half.sin();
        let left = Quaternion::new(c, s, 0.0, 0.0);
        let right = Quaternion::new(c, s, 0.0, 0.0);
        Self::new(left, right)
    }

    /// Create a simple rotation in the YZ plane
    pub fn simple_yz(angle: f64) -> Self {
        let q = Quaternion::from_axis_angle([1.0, 0.0, 0.0], angle / 2.0);
        Self::new(q, q.conjugate())
    }

    /// Create a simple rotation in the YW plane
    pub fn simple_yw(angle: f64) -> Self {
        let half = angle / 2.0;
        let c = half.cos();
        let s = half.sin();
        let left = Quaternion::new(c, 0.0, s, 0.0);
        let right = Quaternion::new(c, 0.0, s, 0.0);
        Self::new(left, right)
    }

    /// Create a simple rotation in the ZW plane
    pub fn simple_zw(angle: f64) -> Self {
        let half = angle / 2.0;
        let c = half.cos();
        let s = half.sin();
        let left = Quaternion::new(c, 0.0, 0.0, s);
        let right = Quaternion::new(c, 0.0, 0.0, s);
        Self::new(left, right)
    }

    /// Compose two 4D rotations
    pub fn compose(self, other: Self) -> Self {
        Self::new(
            self.left.mul(other.left),
            other.right.mul(self.right),
        )
    }

    /// Inverse rotation
    pub fn inverse(self) -> Self {
        Self::new(self.left.inverse(), self.right.inverse())
    }

    /// Apply this rotation to a 4D point
    /// The point is treated as a quaternion: p = x*1 + y*i + z*j + w*k
    pub fn rotate_point(self, point: Vec4) -> Vec4 {
        // Convert Vec4 to quaternion representation
        // We use a specific encoding: the 4D point becomes a quaternion
        let p = Quaternion::new(point.w, point.x, point.y, point.z);

        // Apply rotation: L * p * R^(-1)
        let result = self.left.mul(p).mul(self.right.inverse());

        Vec4::new(result.x, result.y, result.z, result.w)
    }

    /// Interpolate between two 4D rotations
    pub fn slerp(self, other: Self, t: f64) -> Self {
        Self::new(
            self.left.slerp(other.left, t),
            self.right.slerp(other.right, t),
        )
    }
}

impl Default for Rotation4D {
    fn default() -> Self {
        Self::identity()
    }
}

/// Generate the icosian quaternions (vertices of the 600-cell as quaternions)
/// These are the 120 unit quaternions of the binary icosahedral group 2I
pub fn icosian_quaternions() -> Vec<Quaternion> {
    use crate::PHI;

    let mut quats = Vec::with_capacity(120);

    // 8 quaternions: ±1, ±i, ±j, ±k
    for sign in [-1.0, 1.0] {
        quats.push(Quaternion::new(sign, 0.0, 0.0, 0.0));
        quats.push(Quaternion::new(0.0, sign, 0.0, 0.0));
        quats.push(Quaternion::new(0.0, 0.0, sign, 0.0));
        quats.push(Quaternion::new(0.0, 0.0, 0.0, sign));
    }

    // 16 quaternions: (±1/2, ±1/2, ±1/2, ±1/2)
    for s0 in [-0.5, 0.5] {
        for s1 in [-0.5, 0.5] {
            for s2 in [-0.5, 0.5] {
                for s3 in [-0.5, 0.5] {
                    quats.push(Quaternion::new(s0, s1, s2, s3));
                }
            }
        }
    }

    // 96 quaternions from golden ratio combinations
    let phi_half = PHI / 2.0;
    let phi_inv_half = 1.0 / (2.0 * PHI);
    let half = 0.5;

    // These are even permutations of (±φ/2, ±1/2, ±φ⁻¹/2, 0)
    let vals = [
        (phi_half, half, phi_inv_half, 0.0),
        (phi_half, phi_inv_half, 0.0, half),
        (half, phi_half, 0.0, phi_inv_half),
        (half, phi_inv_half, phi_half, 0.0),
        (half, 0.0, phi_inv_half, phi_half),
        (phi_inv_half, phi_half, half, 0.0),
        (phi_inv_half, half, 0.0, phi_half),
        (phi_inv_half, 0.0, phi_half, half),
        (0.0, phi_half, phi_inv_half, half),
        (0.0, half, phi_half, phi_inv_half),
        (0.0, phi_inv_half, half, phi_half),
        (phi_half, 0.0, half, phi_inv_half),
    ];

    for (a, b, c, d) in vals {
        // Generate all sign combinations
        for s0 in [-1.0, 1.0] {
            for s1 in [-1.0, 1.0] {
                for s2 in [-1.0, 1.0] {
                    for s3 in [-1.0, 1.0] {
                        let q = Quaternion::new(s0 * a, s1 * b, s2 * c, s3 * d);
                        // Only add if not already present (avoid duplicates from 0 components)
                        if q.magnitude_squared() > 0.9 {
                            quats.push(q.normalize());
                        }
                    }
                }
            }
        }
    }

    // Remove duplicates (within tolerance)
    let mut unique = Vec::with_capacity(120);
    for q in quats {
        let is_dup = unique.iter().any(|u: &Quaternion| {
            let diff = Quaternion::new(q.w - u.w, q.x - u.x, q.y - u.y, q.z - u.z);
            diff.magnitude_squared() < 1e-6
        });
        if !is_dup {
            unique.push(q);
        }
    }

    unique
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    #[test]
    fn test_quaternion_identity() {
        let q = Quaternion::identity();
        assert_eq!(q.magnitude(), 1.0);
        assert_eq!(q.mul(q), q);
    }

    #[test]
    fn test_quaternion_rotation() {
        let q = Quaternion::from_axis_angle([0.0, 0.0, 1.0], PI / 2.0);
        let point = [1.0, 0.0, 0.0];
        let rotated = q.rotate_point_3d(point);

        // 90° rotation around Z should move (1,0,0) to (0,1,0)
        assert!((rotated[0] - 0.0).abs() < 1e-10);
        assert!((rotated[1] - 1.0).abs() < 1e-10);
        assert!((rotated[2] - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_rotation_4d_identity() {
        let r = Rotation4D::identity();
        let p = Vec4::new(1.0, 2.0, 3.0, 4.0);
        let rotated = r.rotate_point(p);

        assert!((rotated.x - p.x).abs() < 1e-10);
        assert!((rotated.y - p.y).abs() < 1e-10);
        assert!((rotated.z - p.z).abs() < 1e-10);
        assert!((rotated.w - p.w).abs() < 1e-10);
    }

    #[test]
    fn test_quaternion_slerp() {
        let q1 = Quaternion::identity();
        let q2 = Quaternion::from_axis_angle([0.0, 0.0, 1.0], PI);

        let mid = q1.slerp(q2, 0.5);
        let (_, angle) = mid.to_axis_angle();

        assert!((angle - PI / 2.0).abs() < 1e-6);
    }
}
