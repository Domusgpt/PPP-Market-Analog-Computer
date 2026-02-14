"""Tests for H4 Constellation (5-node 600-cell assembly)."""

import pytest
import numpy as np
from engine.constellation.h4_constellation import (
    ConstellationNode,
    ConstellationNetwork,
    NodePosition,
    PhasonPropagator,
)


class TestConstellationNode:
    @pytest.fixture
    def node(self):
        return ConstellationNode(node_id=0, position=NodePosition.CENTER)

    def test_creation(self, node):
        assert node is not None
        assert node.position == NodePosition.CENTER
        assert node.node_id == 0

    def test_has_vertex_ports(self, node):
        """Each node should have vertex ports after init."""
        assert len(node.vertex_ports) > 0

    def test_get_vertex_positions_4d(self, node):
        """Should return 4D vertex positions."""
        verts = node.get_vertex_positions_4d()
        assert isinstance(verts, np.ndarray)
        assert verts.shape[1] == 4


class TestConstellationNetwork:
    @pytest.fixture
    def network(self):
        return ConstellationNetwork()

    def test_has_five_nodes(self, network):
        assert len(network.nodes) == 5

    def test_node_positions(self, network):
        # network.nodes is Dict[int, ConstellationNode]
        positions = {n.position for n in network.nodes.values()}
        expected = {
            NodePosition.CENTER,
            NodePosition.NORTH,
            NodePosition.EAST,
            NodePosition.SOUTH,
            NodePosition.WEST,
        }
        assert positions == expected

    def test_get_600cell_vertices(self, network):
        """5 nodes should produce 120 4D vertices."""
        verts = network.get_600cell_vertices()
        assert isinstance(verts, np.ndarray)
        assert verts.shape[1] == 4
        assert verts.shape[0] == 120

    def test_state_summary(self, network):
        summary = network.get_state_summary()
        assert isinstance(summary, dict)


class TestNodePositions:
    def test_five_positions(self):
        positions = list(NodePosition)
        assert len(positions) == 5

    def test_center_exists(self):
        assert NodePosition.CENTER is not None


class TestPhasonPropagator:
    def test_creation(self):
        prop = PhasonPropagator()
        assert prop is not None

    def test_compute_strain(self):
        prop = PhasonPropagator()
        node = ConstellationNode(node_id=0, position=NodePosition.CENTER)
        strain = prop.compute_strain(node)
        assert isinstance(strain, float)
