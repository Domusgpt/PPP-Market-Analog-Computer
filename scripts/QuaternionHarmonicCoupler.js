const clamp = (value, min, max) => {
    if (!Number.isFinite(value)) {
        return min;
    }
    if (value < min) {
        return min;
    }
    if (value > max) {
        return max;
    }
    return value;
};

const normalizeVector = (vector) => {
    if (!Array.isArray(vector) || !vector.length) {
        return null;
    }
    const lengthSquared = vector.reduce((sum, value) => sum + value * value, 0);
    if (lengthSquared <= 0) {
        return null;
    }
    const length = Math.sqrt(lengthSquared);
    return vector.map((value) => value / length);
};

const cross = (a, b) => {
    if (!a || !b) {
        return [0, 0, 0];
    }
    return [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0]
    ];
};

const wrapUnit = (value) => {
    if (!Number.isFinite(value)) {
        return 0;
    }
    let wrapped = value % 1;
    if (wrapped < 0) {
        wrapped += 1;
    }
    return wrapped;
};

const toPhaseOrbit = (angle) => wrapUnit(angle / (Math.PI * 2));

const computeRatio = (component, fiber, axis, crossComponent) => {
    const exponent = (component * 0.85) + (fiber * 0.35) + (axis * 0.25) + (crossComponent * 0.2);
    return Math.pow(2, exponent);
};

const computePanContribution = (component, fiber, crossComponent, coherence) => {
    const weighted = (component * 0.5) + (fiber * 0.35) + (crossComponent * 0.25);
    return clamp(weighted * (0.5 + 0.5 * coherence), -1, 1) * 0.7;
};

export const deriveSpinorHarmonics = (telemetry) => {
    if (!telemetry || !Array.isArray(telemetry.leftQuaternion) || !Array.isArray(telemetry.rightQuaternion)) {
        return null;
    }
    const leftAxis = normalizeVector(telemetry.leftQuaternion.slice(1));
    const rightAxis = normalizeVector(telemetry.rightQuaternion.slice(1));
    const axisDot = leftAxis && rightAxis
        ? clamp(leftAxis.reduce((sum, value, index) => sum + value * rightAxis[index], 0), -1, 1)
        : 0;
    const coherence = (axisDot + 1) / 2;
    const hopfFiber = Array.isArray(telemetry.hopfFiber) ? telemetry.hopfFiber : [];
    const normalizedBridge = Array.isArray(telemetry.normalizedBridge) ? telemetry.normalizedBridge : [];
    const axisCross = cross(leftAxis, rightAxis);

    const ratios = normalizedBridge.length
        ? normalizedBridge.map((component, index) => {
            const fiberComponent = hopfFiber.length ? hopfFiber[index % hopfFiber.length] : 0;
            const axisComponent = leftAxis ? leftAxis[index % leftAxis.length] : 0;
            const crossComponent = axisCross[index % axisCross.length] || 0;
            return computeRatio(component, fiberComponent, axisComponent, crossComponent);
        })
        : [1, 1, 1, 1];

    const panOrbit = normalizedBridge.length
        ? normalizedBridge.map((component, index) => {
            const fiberComponent = hopfFiber.length ? hopfFiber[index % hopfFiber.length] : 0;
            const crossComponent = axisCross[index % axisCross.length] || 0;
            return computePanContribution(component, fiberComponent, crossComponent, coherence);
        })
        : [];

    const bridgeMagnitude = Number.isFinite(telemetry.bridgeMagnitude) ? telemetry.bridgeMagnitude : 0;
    const braidDensity = normalizedBridge.length
        ? normalizedBridge.reduce((sum, value) => sum + value * value, 0) / normalizedBridge.length
        : 0;

    const phaseOrbit = [
        toPhaseOrbit(telemetry.leftAngle || 0),
        toPhaseOrbit(telemetry.rightAngle || 0)
    ];

    const pitchLattice = ratios.map((ratio, index) => ({
        index,
        ratio,
        cents: Math.log2(ratio) * 1200,
        pan: panOrbit.length ? panOrbit[index % panOrbit.length] : 0
    }));

    return {
        ratios,
        panOrbit,
        phaseOrbit,
        axis: {
            left: leftAxis ? leftAxis.slice() : null,
            right: rightAxis ? rightAxis.slice() : null,
            dot: axisDot,
            cross: axisCross
        },
        coherence,
        braidDensity,
        bridgeMagnitude,
        fiber: hopfFiber.slice(),
        pitchLattice
    };
};
