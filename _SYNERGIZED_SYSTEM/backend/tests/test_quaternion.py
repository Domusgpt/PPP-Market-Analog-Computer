"""Tests for quaternion 4D rotation algebra."""

import math
import pytest
import numpy as np
from engine.geometry.quaternion_4d import (
    Quaternion4D,
    IsoclinicRotation,
    RotationMode,
    QuaternionRotation,
)


class TestQuaternionBasics:
    def test_identity_quaternion(self):
        q = Quaternion4D(1.0, 0.0, 0.0, 0.0)
        assert abs(q.w - 1.0) < 1e-10
        assert abs(q.x) < 1e-10

    def test_unit_norm(self):
        q = Quaternion4D(1.0, 1.0, 1.0, 1.0)
        # Should be auto-normalized to unit quaternion
        norm = math.sqrt(q.w**2 + q.x**2 + q.y**2 + q.z**2)
        assert abs(norm - 1.0) < 1e-10

    def test_from_axis_angle(self):
        axis = np.array([0.0, 0.0, 1.0])
        angle = math.pi / 2
        q = Quaternion4D.from_axis_angle(axis, angle)
        norm = math.sqrt(q.w**2 + q.x**2 + q.y**2 + q.z**2)
        assert abs(norm - 1.0) < 1e-10

    def test_to_array_roundtrip(self):
        q = Quaternion4D(0.5, 0.5, 0.5, 0.5)
        arr = q.to_array()
        q2 = Quaternion4D.from_array(arr)
        assert abs(q.w - q2.w) < 1e-10
        assert abs(q.x - q2.x) < 1e-10

    def test_identity_class_method(self):
        q = Quaternion4D.identity()
        assert abs(q.w - 1.0) < 1e-10
        assert abs(q.x) < 1e-10
        assert abs(q.y) < 1e-10
        assert abs(q.z) < 1e-10


class TestQuaternionOperations:
    def test_conjugate(self):
        q = Quaternion4D(0.5, 0.5, 0.5, 0.5)
        qc = q.conjugate()
        assert abs(qc.w - q.w) < 1e-10
        assert abs(qc.x + q.x) < 1e-10
        assert abs(qc.y + q.y) < 1e-10
        assert abs(qc.z + q.z) < 1e-10

    def test_multiplication_preserves_norm(self):
        q1 = Quaternion4D(0.5, 0.5, 0.5, 0.5)
        q2 = Quaternion4D.from_axis_angle(np.array([1, 0, 0]), math.pi / 4)
        product = q1 * q2
        norm = product.norm()
        assert abs(norm - 1.0) < 1e-10

    def test_slerp_endpoints(self):
        q1 = Quaternion4D.identity()
        q2 = Quaternion4D.from_axis_angle(np.array([0, 0, 1]), math.pi / 2)
        # t=0 should give q1, t=1 should give q2
        q_start = Quaternion4D.slerp(q1, q2, 0.0)
        assert abs(q_start.w - q1.w) < 1e-6
        q_end = Quaternion4D.slerp(q1, q2, 1.0)
        assert abs(q_end.w - q2.w) < 1e-6

    def test_rotation_matrix_3d(self):
        q = Quaternion4D.from_axis_angle(np.array([0, 0, 1]), math.pi / 2)
        R = q.to_rotation_matrix_3d()
        assert R.shape == (3, 3)
        # Should be orthogonal
        assert np.allclose(R @ R.T, np.eye(3), atol=1e-6)


class TestIsoclinicRotation:
    def test_isoclinic_creation(self):
        """IsoclinicRotation uses axis_pair kwarg, not plane1/plane2."""
        rot = IsoclinicRotation(
            angle=math.pi / 6,
            axis_pair=((0, 1), (2, 3)),  # XY-ZW pair
            chirality="left",
        )
        assert abs(rot.angle - math.pi / 6) < 1e-10

    def test_isoclinic_to_quaternion(self):
        rot = IsoclinicRotation(
            angle=math.pi / 4,
            axis_pair=((0, 1), (2, 3)),
        )
        qr = rot.to_quaternion_rotation()
        assert isinstance(qr, QuaternionRotation)

    def test_isoclinic_apply(self):
        rot = IsoclinicRotation(angle=math.pi / 4)
        point = np.array([1.0, 0.0, 0.0, 0.0])
        result = rot.apply(point.reshape(1, 4))
        assert result.shape == (1, 4)
        # Rotation should preserve norm
        assert abs(np.linalg.norm(result) - 1.0) < 1e-6

    def test_golden_rotation(self):
        """Golden rotation is a special isoclinic rotation."""
        rot = IsoclinicRotation.golden_rotation()
        assert isinstance(rot, IsoclinicRotation)
