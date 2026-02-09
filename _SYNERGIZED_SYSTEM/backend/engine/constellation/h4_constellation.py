"""
H4 Constellation - Distributed 600-cell assembly and coordination.

The 600-cell can be partitioned into FIVE disjoint 24-cells. This module
implements the networking architecture to coordinate multiple 24-cell
prototype units to form the complete 600-cell hyper-structure.

When units expand to Vertex Size 1 (deployed state), their vertices
can "interlock" with neighboring units, enabling:
1. Data flow across module boundaries
2. Phason strain propagation through the network
3. Collective 4D rotation simulation
4. Distributed quaternion processing

The networking uses magnetic pogo-pin connectors at the hexagonal frame
vertices, establishing data links when modules are physically coupled.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Optional, Tuple, Set
import numpy as np
from ..geometry.h4_geometry import (
    Polytope24Cell, Polytope600Cell, H4Geometry, LayerPlane
)
from ..geometry.quaternion_4d import (
    Quaternion4D, QuaternionRotation, IsoclinicRotation, QuaternionController
)
from ..kirigami.h4_kirigami import H4KirigamiStack, DeploymentState


class ConstellationState(Enum):
    """States of a constellation network."""
    DISCONNECTED = "disconnected"   # Modules not coupled
    PARTIAL = "partial"             # Some connections active
    COUPLED = "coupled"             # Full constellation formed
    COMPUTING = "computing"         # Active data processing
    SYNCHRONIZED = "synchronized"   # All modules in sync


class NodePosition(Enum):
    """Position of a node in the 5-node 600-cell constellation."""
    CENTER = 0      # Central reference node
    NORTH = 1       # First rotated 24-cell
    EAST = 2        # Second rotated 24-cell
    SOUTH = 3       # Third rotated 24-cell
    WEST = 4        # Fourth rotated 24-cell


@dataclass
class VertexPort:
    """
    A vertex port for inter-module communication.

    Physical implementation: Magnetic pogo-pin connector
    at the hexagonal frame vertices.
    """
    port_id: int                    # 0-5 for hexagonal vertices
    position: np.ndarray            # 3D position on frame
    is_connected: bool = False      # Connection status
    connected_node_id: Optional[int] = None
    connected_port_id: Optional[int] = None

    # Data channel
    data_in: np.ndarray = field(default_factory=lambda: np.zeros(4))
    data_out: np.ndarray = field(default_factory=lambda: np.zeros(4))

    # Physical state
    is_extended: bool = False       # Pogo pin extended (at Size 1)

    def extend(self):
        """Extend the pogo pin (when reaching Size 1)."""
        self.is_extended = True

    def retract(self):
        """Retract the pogo pin."""
        self.is_extended = False
        self.is_connected = False
        self.connected_node_id = None
        self.connected_port_id = None

    def connect(self, other_node_id: int, other_port_id: int):
        """Establish connection with another port."""
        if self.is_extended:
            self.is_connected = True
            self.connected_node_id = other_node_id
            self.connected_port_id = other_port_id

    def send(self, data: np.ndarray):
        """Send data through the port."""
        self.data_out = data.copy()

    def receive(self) -> np.ndarray:
        """Receive data from the port."""
        return self.data_in.copy()


@dataclass
class ConstellationNode:
    """
    A single node in the H4 Constellation network.

    Each node represents one physical 24-cell prototype unit,
    containing the kirigami stack, quaternion controller, and
    vertex ports for inter-module communication.
    """
    node_id: int
    position: NodePosition
    kirigami_stack: H4KirigamiStack = field(default_factory=H4KirigamiStack)
    quaternion_controller: QuaternionController = field(
        default_factory=QuaternionController)
    geometry: H4Geometry = field(default_factory=H4Geometry)

    # Network connectivity
    vertex_ports: List[VertexPort] = field(default_factory=list)
    neighbors: Dict[int, int] = field(default_factory=dict)  # port_id -> node_id

    # State
    current_quaternion: Quaternion4D = field(default_factory=Quaternion4D.identity)
    deployment_state: DeploymentState = DeploymentState.LOCKED

    # 4D position within the 600-cell
    rotation_offset: QuaternionRotation = field(
        default_factory=lambda: QuaternionRotation())

    def __post_init__(self):
        """Initialize vertex ports."""
        if not self.vertex_ports:
            self._init_vertex_ports()
        self._compute_rotation_offset()

    def _init_vertex_ports(self):
        """Create the 6 vertex ports on the hexagonal frame."""
        # Hexagonal vertex positions
        for i in range(6):
            angle = i * np.pi / 3
            position = np.array([np.cos(angle), np.sin(angle), 0])
            self.vertex_ports.append(VertexPort(
                port_id=i,
                position=position
            ))

    def _compute_rotation_offset(self):
        """
        Compute the 4D rotation offset for this node's position.

        The five 24-cells in a 600-cell are related by Golden Ratio
        rotations of 72° (360°/5).
        """
        if self.position == NodePosition.CENTER:
            # Center node has identity rotation
            self.rotation_offset = QuaternionRotation()
        else:
            # Other nodes are rotated by multiples of 72°
            k = self.position.value  # 1, 2, 3, or 4
            angle = k * 2 * np.pi / 5  # 72°, 144°, 216°, 288°

            # Golden isoclinic rotation
            iso_rot = IsoclinicRotation.golden_rotation()
            iso_rot.angle = angle
            self.rotation_offset = iso_rot.to_quaternion_rotation()

    def set_deployment(self, state: DeploymentState):
        """Set the deployment state of the kirigami stack."""
        self.deployment_state = state
        self.kirigami_stack.set_all_states(state.value)

        # Extend/retract vertex ports based on deployment
        for port in self.vertex_ports:
            if state == DeploymentState.DEPLOYED:
                port.extend()
            else:
                port.retract()

    def apply_quaternion(self, q: Quaternion4D):
        """Apply a quaternion rotation to the node."""
        self.current_quaternion = q

        # Apply node's offset rotation
        effective_q = self.rotation_offset.q_left * q

        # Compute layer actuation
        self.quaternion_controller.update_state(effective_q)
        layer_values = self.quaternion_controller.compute_layer_actuation(effective_q)

        # Apply to kirigami stack
        self.kirigami_stack.apply_quaternion_control(layer_values)

    def get_vertex_positions_4d(self) -> np.ndarray:
        """Get the 4D positions of this node's 24-cell vertices."""
        base_vertices = self.geometry.polytope_24.get_vertex_array()

        # Apply rotation offset
        rotated = self.rotation_offset.apply_to_points(base_vertices)

        # Scale based on deployment
        scale = 1.0 + self.deployment_state.value * 0.618
        return rotated * scale

    def get_moire_output(self) -> np.ndarray:
        """Get the current moiré pattern output."""
        return self.kirigami_stack.compute_spectral_moire()

    def broadcast_state(self):
        """Broadcast current quaternion state through all connected ports."""
        state_data = self.current_quaternion.to_array()
        for port in self.vertex_ports:
            if port.is_connected:
                port.send(state_data)

    def receive_neighbor_states(self) -> Dict[int, Quaternion4D]:
        """Receive quaternion states from connected neighbors."""
        states = {}
        for port in self.vertex_ports:
            if port.is_connected and port.connected_node_id is not None:
                data = port.receive()
                states[port.connected_node_id] = Quaternion4D.from_array(data)
        return states


class PhasonPropagator:
    """
    Propagates phason strain waves across the constellation network.

    When one node experiences a 4D distortion (deviation from ideal
    24-cell projection), the strain propagates to neighboring nodes
    through the vertex connections, simulating the elastic behavior
    of the 600-cell hyper-structure.
    """

    def __init__(self, damping: float = 0.1, wave_speed: float = 1.0):
        """
        Initialize the phason propagator.

        Args:
            damping: Energy dissipation rate
            wave_speed: Propagation velocity (normalized)
        """
        self.damping = damping
        self.wave_speed = wave_speed

    def compute_strain(self, node: ConstellationNode) -> float:
        """
        Compute the phason strain at a node.

        Strain is measured as deviation from ideal quaternion alignment
        with the node's rotation offset.
        """
        # Expected quaternion based on node position
        expected = node.rotation_offset.q_left

        # Actual quaternion state
        actual = node.current_quaternion

        # Geodesic distance on quaternion sphere
        dot = actual.dot(expected)
        angle = 2 * np.arccos(min(abs(dot), 1.0))

        return angle

    def propagate(self,
                   network: "ConstellationNetwork",
                   source_node_id: int,
                   strain_magnitude: float,
                   steps: int = 10) -> Dict[int, List[float]]:
        """
        Propagate strain from a source node through the network.

        Args:
            network: The constellation network
            source_node_id: ID of the strain source
            strain_magnitude: Initial strain magnitude
            steps: Number of propagation steps

        Returns:
            Dictionary mapping node_id to strain history
        """
        strain_history = {nid: [0.0] for nid in network.nodes.keys()}
        strain_history[source_node_id][0] = strain_magnitude

        current_strain = {nid: 0.0 for nid in network.nodes.keys()}
        current_strain[source_node_id] = strain_magnitude

        for _ in range(steps):
            new_strain = {nid: 0.0 for nid in network.nodes.keys()}

            for node_id, node in network.nodes.items():
                # Receive strain from neighbors
                neighbor_strain = 0.0
                neighbor_count = 0

                for port in node.vertex_ports:
                    if port.is_connected and port.connected_node_id is not None:
                        neighbor_id = port.connected_node_id
                        neighbor_strain += current_strain.get(neighbor_id, 0)
                        neighbor_count += 1

                if neighbor_count > 0:
                    # Wave equation update
                    incoming = neighbor_strain / neighbor_count
                    propagated = self.wave_speed * (incoming - current_strain[node_id])
                    damped = propagated * (1 - self.damping)
                    new_strain[node_id] = current_strain[node_id] + damped

            current_strain = new_strain
            for nid, strain in current_strain.items():
                strain_history[nid].append(strain)

        return strain_history

    def compute_collective_strain(self,
                                    network: "ConstellationNetwork") -> float:
        """
        Compute the total phason strain across the constellation.

        Returns:
            Sum of individual node strains
        """
        total = 0.0
        for node in network.nodes.values():
            total += self.compute_strain(node)
        return total


class ConstellationNetwork:
    """
    The complete H4 Constellation network.

    Manages five 24-cell nodes arranged to form the 600-cell structure
    when fully deployed and connected.
    """

    def __init__(self, auto_init: bool = True):
        """
        Initialize the constellation network.

        Args:
            auto_init: If True, automatically create the 5 nodes
        """
        self.nodes: Dict[int, ConstellationNode] = {}
        self.connections: List[Tuple[int, int, int, int]] = []  # (node1, port1, node2, port2)
        self.state = ConstellationState.DISCONNECTED
        self.phason_propagator = PhasonPropagator()

        if auto_init:
            self._init_nodes()

    def _init_nodes(self):
        """Create the five constellation nodes."""
        for position in NodePosition:
            node = ConstellationNode(
                node_id=position.value,
                position=position
            )
            self.nodes[position.value] = node

    def get_node(self, node_id: int) -> ConstellationNode:
        """Get a node by ID."""
        return self.nodes[node_id]

    def connect_nodes(self,
                       node1_id: int, port1_id: int,
                       node2_id: int, port2_id: int):
        """
        Establish connection between two node ports.

        Connection can only be made when both ports are extended
        (nodes at deployment state 1).
        """
        node1 = self.nodes[node1_id]
        node2 = self.nodes[node2_id]

        port1 = node1.vertex_ports[port1_id]
        port2 = node2.vertex_ports[port2_id]

        if port1.is_extended and port2.is_extended:
            port1.connect(node2_id, port2_id)
            port2.connect(node1_id, port1_id)
            self.connections.append((node1_id, port1_id, node2_id, port2_id))

            # Update neighbor maps
            node1.neighbors[port1_id] = node2_id
            node2.neighbors[port2_id] = node1_id

            self._update_state()

    def disconnect_nodes(self, node1_id: int, port1_id: int):
        """Disconnect a specific port."""
        node1 = self.nodes[node1_id]
        port1 = node1.vertex_ports[port1_id]

        if port1.is_connected:
            node2_id = port1.connected_node_id
            port2_id = port1.connected_port_id

            # Clear connection on both ends
            port1.retract()
            if node2_id is not None and port2_id is not None:
                self.nodes[node2_id].vertex_ports[port2_id].retract()

            # Remove from connection list
            self.connections = [
                c for c in self.connections
                if not (c[0] == node1_id and c[1] == port1_id)
                and not (c[2] == node1_id and c[3] == port1_id)
            ]

            self._update_state()

    def _update_state(self):
        """Update the network state based on connections."""
        total_connections = len(self.connections)

        if total_connections == 0:
            self.state = ConstellationState.DISCONNECTED
        elif total_connections < 10:  # 5 nodes * 2 connections each minimum
            self.state = ConstellationState.PARTIAL
        else:
            self.state = ConstellationState.COUPLED

    def deploy_all(self, state: DeploymentState):
        """Set deployment state for all nodes."""
        for node in self.nodes.values():
            node.set_deployment(state)

        # Auto-connect when deployed
        if state == DeploymentState.DEPLOYED:
            self._auto_connect()

    def _auto_connect(self):
        """Automatically connect adjacent nodes when deployed."""
        # Connect center to all peripheral nodes
        center = self.nodes[NodePosition.CENTER.value]
        peripherals = [
            NodePosition.NORTH, NodePosition.EAST,
            NodePosition.SOUTH, NodePosition.WEST
        ]

        for i, pos in enumerate(peripherals):
            peripheral_node = self.nodes[pos.value]
            # Connect opposite ports
            self.connect_nodes(
                NodePosition.CENTER.value, i,
                pos.value, (i + 3) % 6
            )

    def apply_global_quaternion(self, q: Quaternion4D):
        """Apply a quaternion rotation to all nodes."""
        for node in self.nodes.values():
            node.apply_quaternion(q)

        # Broadcast states through connections
        for node in self.nodes.values():
            node.broadcast_state()

        self.state = ConstellationState.COMPUTING

    def synchronize(self):
        """
        Synchronize all nodes to achieve collective coherence.

        Uses averaging of neighbor states to converge on a common
        quaternion orientation.
        """
        for _ in range(10):  # Synchronization iterations
            new_quaternions = {}

            for node_id, node in self.nodes.items():
                # Get neighbor states
                neighbor_states = node.receive_neighbor_states()

                if neighbor_states:
                    # Average with current state
                    total = node.current_quaternion.to_array()
                    for nq in neighbor_states.values():
                        total += nq.to_array()
                    avg = total / (len(neighbor_states) + 1)

                    new_quaternions[node_id] = Quaternion4D.from_array(avg)
                else:
                    new_quaternions[node_id] = node.current_quaternion

            # Apply new quaternions
            for node_id, q in new_quaternions.items():
                self.nodes[node_id].apply_quaternion(q)
                self.nodes[node_id].broadcast_state()

        self.state = ConstellationState.SYNCHRONIZED

    def get_600cell_vertices(self) -> np.ndarray:
        """
        Get all vertices of the assembled 600-cell.

        Returns:
            Array of 120 4D vertices (or fewer if not all nodes active)
        """
        all_vertices = []

        for node in self.nodes.values():
            vertices = node.get_vertex_positions_4d()
            all_vertices.append(vertices)

        return np.vstack(all_vertices)

    def get_collective_moire(self) -> np.ndarray:
        """
        Compute combined moiré output from all nodes.

        Returns:
            Composite moiré pattern
        """
        patterns = []

        for node in self.nodes.values():
            pattern = node.get_moire_output()
            patterns.append(pattern)

        # Combine patterns (average for now)
        return np.mean(patterns, axis=0)

    def compute_network_strain(self) -> float:
        """Compute total phason strain across the network."""
        return self.phason_propagator.compute_collective_strain(self)

    def get_state_summary(self) -> Dict:
        """Get summary of network state."""
        return {
            "state": self.state.value,
            "num_nodes": len(self.nodes),
            "num_connections": len(self.connections),
            "total_strain": self.compute_network_strain(),
            "node_deployments": {
                nid: node.deployment_state.value
                for nid, node in self.nodes.items()
            }
        }


class H4ConstellationController:
    """
    High-level controller for the H4 Constellation system.

    Integrates all subsystems:
    - Constellation network (5 nodes)
    - Quaternion control
    - Kirigami actuation
    - LPQ feedback
    - Phason propagation

    Provides a unified API for 4D rotation simulation and data processing.
    """

    def __init__(self):
        """Initialize the constellation controller."""
        self.network = ConstellationNetwork()
        self.geometry = H4Geometry()

        # Control parameters
        self.feedback_gain = 0.1
        self.damping = 0.05

        # State history
        self.quaternion_history: List[Quaternion4D] = []
        self.strain_history: List[float] = []

    def initialize(self):
        """Initialize the system to default state."""
        self.network.deploy_all(DeploymentState.LOCKED)

    def deploy(self):
        """Deploy to 600-cell constellation state."""
        # Gradual deployment sequence
        for state in [DeploymentState.AUXETIC, DeploymentState.DEPLOYED]:
            self.network.deploy_all(state)

        # Synchronize nodes
        self.network.synchronize()

    def retract(self):
        """Retract to 24-cell locked state."""
        self.network.deploy_all(DeploymentState.LOCKED)

    def process_4d_input(self, input_vector: np.ndarray) -> Dict:
        """
        Process a 4D input vector through the constellation.

        Args:
            input_vector: 4D data vector [w, x, y, z]

        Returns:
            Processing results including moiré output and feedback
        """
        # Normalize to quaternion
        q = Quaternion4D.from_array(input_vector)

        # Apply to network
        self.network.apply_global_quaternion(q)

        # Get outputs
        moire_output = self.network.get_collective_moire()
        strain = self.network.compute_network_strain()

        # Store history
        self.quaternion_history.append(q)
        self.strain_history.append(strain)

        return {
            "quaternion": q.to_array(),
            "moire_pattern": moire_output,
            "phason_strain": strain,
            "network_state": self.network.state.value,
            "vertices_4d": self.network.get_600cell_vertices()
        }

    def process_data_stream(self, data: np.ndarray) -> List[Dict]:
        """
        Process a stream of 4D data vectors.

        Args:
            data: Array of 4D vectors (N x 4)

        Returns:
            List of processing results
        """
        results = []
        for vec in data:
            result = self.process_4d_input(vec)
            results.append(result)
        return results

    def perform_isoclinic_rotation(self,
                                    angle: float,
                                    chirality: str = "left") -> Dict:
        """
        Perform an isoclinic (Clifford) rotation.

        Args:
            angle: Rotation angle in radians
            chirality: "left" or "right"

        Returns:
            Rotation result
        """
        iso = IsoclinicRotation(angle=angle, chirality=chirality)
        q = iso.to_quaternion_rotation().q_left

        return self.process_4d_input(q.to_array())

    def get_palindrome_sequence(self, steps: int = 20) -> List[Dict]:
        """
        Generate the H4 palindrome transformation sequence.

        24-cell → 600-cell → 120-cell (dual) → 24-cell

        Args:
            steps: Number of interpolation steps per transition

        Returns:
            List of states through the palindrome
        """
        sequence = []

        # Phase 1: 24-cell → 600-cell (expansion)
        for i in range(steps):
            t = i / steps
            q = Quaternion4D(
                w=np.cos(t * np.pi / 4),
                x=np.sin(t * np.pi / 4) / np.sqrt(3),
                y=np.sin(t * np.pi / 4) / np.sqrt(3),
                z=np.sin(t * np.pi / 4) / np.sqrt(3)
            )
            result = self.process_4d_input(q.to_array())
            result["phase"] = "expansion"
            result["progress"] = t
            sequence.append(result)

        # Phase 2: 600-cell → 120-cell (dual)
        for i in range(steps):
            t = i / steps
            # Dual transformation: invert the quaternion
            q = Quaternion4D(
                w=np.cos((1 - t) * np.pi / 4),
                x=-np.sin(t * np.pi / 4) / np.sqrt(3),
                y=-np.sin(t * np.pi / 4) / np.sqrt(3),
                z=-np.sin(t * np.pi / 4) / np.sqrt(3)
            )
            result = self.process_4d_input(q.to_array())
            result["phase"] = "dual"
            result["progress"] = t
            sequence.append(result)

        # Phase 3: 120-cell → 24-cell (return)
        for i in range(steps):
            t = i / steps
            q = Quaternion4D(
                w=np.cos((1 - t) * np.pi / 4),
                x=0,
                y=0,
                z=0
            )
            result = self.process_4d_input(q.to_array())
            result["phase"] = "return"
            result["progress"] = t
            sequence.append(result)

        return sequence

    def get_system_state(self) -> Dict:
        """Get complete system state."""
        return {
            "network": self.network.get_state_summary(),
            "geometry": {
                "24cell_vertices": self.geometry.polytope_24.get_vertex_array().shape,
                "trilatic_channels": list(self.geometry.trilatic.get_all_vertices().shape)
            },
            "history": {
                "quaternion_count": len(self.quaternion_history),
                "strain_mean": np.mean(self.strain_history) if self.strain_history else 0,
                "strain_max": np.max(self.strain_history) if self.strain_history else 0
            }
        }
