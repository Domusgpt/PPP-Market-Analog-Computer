//! 4D Vector implementation for geometric computations

use std::ops::{Add, Sub, Mul, Neg, Index, IndexMut};
use serde::{Serialize, Deserialize};

/// A 4-dimensional vector (x, y, z, w)
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Vec4 {
    pub x: f64,
    pub y: f64,
    pub z: f64,
    pub w: f64,
}

impl Vec4 {
    /// Create a new 4D vector
    pub const fn new(x: f64, y: f64, z: f64, w: f64) -> Self {
        Self { x, y, z, w }
    }

    /// Zero vector
    pub const fn zero() -> Self {
        Self::new(0.0, 0.0, 0.0, 0.0)
    }

    /// Unit vector along X axis
    pub const fn unit_x() -> Self {
        Self::new(1.0, 0.0, 0.0, 0.0)
    }

    /// Unit vector along Y axis
    pub const fn unit_y() -> Self {
        Self::new(0.0, 1.0, 0.0, 0.0)
    }

    /// Unit vector along Z axis
    pub const fn unit_z() -> Self {
        Self::new(0.0, 0.0, 1.0, 0.0)
    }

    /// Unit vector along W axis
    pub const fn unit_w() -> Self {
        Self::new(0.0, 0.0, 0.0, 1.0)
    }

    /// Create from array
    pub fn from_array(arr: [f64; 4]) -> Self {
        Self::new(arr[0], arr[1], arr[2], arr[3])
    }

    /// Convert to array
    pub fn to_array(self) -> [f64; 4] {
        [self.x, self.y, self.z, self.w]
    }

    /// Convert to f32 array (for GPU buffers)
    pub fn to_f32_array(self) -> [f32; 4] {
        [self.x as f32, self.y as f32, self.z as f32, self.w as f32]
    }

    /// Dot product with another vector
    pub fn dot(self, other: Self) -> f64 {
        self.x * other.x + self.y * other.y + self.z * other.z + self.w * other.w
    }

    /// Squared magnitude
    pub fn magnitude_squared(self) -> f64 {
        self.dot(self)
    }

    /// Magnitude (length)
    pub fn magnitude(self) -> f64 {
        self.magnitude_squared().sqrt()
    }

    /// Normalize to unit length
    pub fn normalize(self) -> Self {
        let mag = self.magnitude();
        if mag > 1e-10 {
            self * (1.0 / mag)
        } else {
            Self::zero()
        }
    }

    /// Linear interpolation between two vectors
    pub fn lerp(self, other: Self, t: f64) -> Self {
        self * (1.0 - t) + other * t
    }

    /// Distance to another vector
    pub fn distance(self, other: Self) -> f64 {
        (self - other).magnitude()
    }

    /// Squared distance to another vector
    pub fn distance_squared(self, other: Self) -> f64 {
        (self - other).magnitude_squared()
    }

    /// Scale by golden ratio φ
    pub fn scale_phi(self) -> Self {
        self * crate::PHI
    }

    /// Scale by inverse golden ratio φ⁻¹
    pub fn scale_phi_inv(self) -> Self {
        self * crate::PHI_INV
    }

    /// Project to 3D by dropping the W component (orthographic)
    pub fn project_drop_w(self) -> [f64; 3] {
        [self.x, self.y, self.z]
    }

    /// Project to 3D by dropping any axis
    pub fn project_drop_axis(self, axis: usize) -> [f64; 3] {
        match axis {
            0 => [self.y, self.z, self.w],
            1 => [self.x, self.z, self.w],
            2 => [self.x, self.y, self.w],
            _ => [self.x, self.y, self.z],
        }
    }

    /// Stereographic projection from 4D hypersphere to 3D
    /// Projects from the point (0,0,0,1) onto the w=0 hyperplane
    pub fn stereographic_project(self) -> [f64; 3] {
        let denom = 1.0 - self.w;
        if denom.abs() < 1e-10 {
            // Near the projection point, return a large but finite value
            let scale = 1000.0;
            [self.x * scale, self.y * scale, self.z * scale]
        } else {
            let scale = 1.0 / denom;
            [self.x * scale, self.y * scale, self.z * scale]
        }
    }

    /// Perspective projection from 4D
    pub fn perspective_project(self, focal_distance: f64) -> [f64; 3] {
        let w_offset = self.w + focal_distance;
        if w_offset.abs() < 1e-10 {
            [self.x * 1000.0, self.y * 1000.0, self.z * 1000.0]
        } else {
            let scale = focal_distance / w_offset;
            [self.x * scale, self.y * scale, self.z * scale]
        }
    }

    /// Component-wise absolute value
    pub fn abs(self) -> Self {
        Self::new(self.x.abs(), self.y.abs(), self.z.abs(), self.w.abs())
    }

    /// Component-wise minimum
    pub fn min(self, other: Self) -> Self {
        Self::new(
            self.x.min(other.x),
            self.y.min(other.y),
            self.z.min(other.z),
            self.w.min(other.w),
        )
    }

    /// Component-wise maximum
    pub fn max(self, other: Self) -> Self {
        Self::new(
            self.x.max(other.x),
            self.y.max(other.y),
            self.z.max(other.z),
            self.w.max(other.w),
        )
    }

    /// Reflect this vector across a hyperplane with given normal
    pub fn reflect(self, normal: Self) -> Self {
        let n = normal.normalize();
        self - n * (2.0 * self.dot(n))
    }
}

impl Default for Vec4 {
    fn default() -> Self {
        Self::zero()
    }
}

impl Add for Vec4 {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        Self::new(
            self.x + other.x,
            self.y + other.y,
            self.z + other.z,
            self.w + other.w,
        )
    }
}

impl Sub for Vec4 {
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        Self::new(
            self.x - other.x,
            self.y - other.y,
            self.z - other.z,
            self.w - other.w,
        )
    }
}

impl Mul<f64> for Vec4 {
    type Output = Self;

    fn mul(self, scalar: f64) -> Self {
        Self::new(
            self.x * scalar,
            self.y * scalar,
            self.z * scalar,
            self.w * scalar,
        )
    }
}

impl Mul<Vec4> for f64 {
    type Output = Vec4;

    fn mul(self, vec: Vec4) -> Vec4 {
        vec * self
    }
}

impl Neg for Vec4 {
    type Output = Self;

    fn neg(self) -> Self {
        Self::new(-self.x, -self.y, -self.z, -self.w)
    }
}

impl Index<usize> for Vec4 {
    type Output = f64;

    fn index(&self, index: usize) -> &f64 {
        match index {
            0 => &self.x,
            1 => &self.y,
            2 => &self.z,
            3 => &self.w,
            _ => panic!("Vec4 index out of bounds: {}", index),
        }
    }
}

impl IndexMut<usize> for Vec4 {
    fn index_mut(&mut self, index: usize) -> &mut f64 {
        match index {
            0 => &mut self.x,
            1 => &mut self.y,
            2 => &mut self.z,
            3 => &mut self.w,
            _ => panic!("Vec4 index out of bounds: {}", index),
        }
    }
}

impl From<[f64; 4]> for Vec4 {
    fn from(arr: [f64; 4]) -> Self {
        Self::from_array(arr)
    }
}

impl From<Vec4> for [f64; 4] {
    fn from(v: Vec4) -> [f64; 4] {
        v.to_array()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vec4_basics() {
        let v = Vec4::new(1.0, 2.0, 3.0, 4.0);
        assert_eq!(v.x, 1.0);
        assert_eq!(v[2], 3.0);
    }

    #[test]
    fn test_vec4_dot() {
        let a = Vec4::new(1.0, 0.0, 0.0, 0.0);
        let b = Vec4::new(0.0, 1.0, 0.0, 0.0);
        assert_eq!(a.dot(b), 0.0);

        let c = Vec4::new(1.0, 2.0, 3.0, 4.0);
        assert_eq!(c.dot(c), 30.0);
    }

    #[test]
    fn test_vec4_normalize() {
        let v = Vec4::new(3.0, 0.0, 4.0, 0.0);
        let n = v.normalize();
        assert!((n.magnitude() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_vec4_lerp() {
        let a = Vec4::new(0.0, 0.0, 0.0, 0.0);
        let b = Vec4::new(2.0, 4.0, 6.0, 8.0);
        let mid = a.lerp(b, 0.5);
        assert_eq!(mid, Vec4::new(1.0, 2.0, 3.0, 4.0));
    }

    #[test]
    fn test_golden_ratio_scaling() {
        let v = Vec4::new(1.0, 1.0, 1.0, 1.0);
        let scaled = v.scale_phi();
        let unscaled = scaled.scale_phi_inv();
        // φ * φ⁻¹ ≈ 1
        assert!((v.x - unscaled.x).abs() < 1e-10);
    }
}
