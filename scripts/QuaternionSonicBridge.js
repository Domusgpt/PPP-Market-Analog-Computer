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

const createIdentityMatrix = (size) => Array.from({ length: size }, (_, row) => (
    Array.from({ length: size }, (_, col) => (row === col ? 1 : 0))
));

const multiplyMatrices = (a, b) => {
    const rows = a.length;
    const cols = b[0].length;
    const shared = b.length;
    const result = Array.from({ length: rows }, () => new Array(cols).fill(0));
    for (let i = 0; i < rows; i += 1) {
        for (let k = 0; k < shared; k += 1) {
            const aik = a[i][k];
            if (aik === 0) {
                continue;
            }
            for (let j = 0; j < cols; j += 1) {
                result[i][j] += aik * b[k][j];
            }
        }
    }
    return result;
};

const multiplyMatrixVector = (matrix, vector) => {
    const result = new Array(matrix.length).fill(0);
    for (let row = 0; row < matrix.length; row += 1) {
        const rowValues = matrix[row];
        let sum = 0;
        for (let col = 0; col < rowValues.length; col += 1) {
            sum += rowValues[col] * vector[col];
        }
        result[row] = sum;
    }
    return result;
};

const transpose = (matrix) => matrix[0].map((_, index) => matrix.map((row) => row[index]));

const flattenMatrix = (matrix) => {
    const values = [];
    for (let row = 0; row < matrix.length; row += 1) {
        for (let col = 0; col < matrix[row].length; col += 1) {
            values.push(matrix[row][col]);
        }
    }
    return values;
};

const normalizeVector = (vector) => {
    const lengthSquared = vector.reduce((sum, value) => sum + value * value, 0);
    if (lengthSquared <= 0) {
        return null;
    }
    const length = Math.sqrt(lengthSquared);
    return vector.map((value) => value / length);
};

const planeRotationMatrix = (axisA, axisB, angle) => {
    const cos = Math.cos(angle);
    const sin = Math.sin(angle);
    const matrix = createIdentityMatrix(4);
    matrix[axisA][axisA] = cos;
    matrix[axisA][axisB] = sin;
    matrix[axisB][axisA] = -sin;
    matrix[axisB][axisB] = cos;
    return matrix;
};

const composeMatrices = (matrices) => {
    return matrices.reduce((accumulator, matrix) => multiplyMatrices(matrix, accumulator), createIdentityMatrix(4));
};

const LEFT_BASIS = [
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 1]
].map((components) => {
    const [w, x, y, z] = components;
    return [
        [w, -x, -y, -z],
        [x,  w, -z,  y],
        [y,  z,  w, -x],
        [z, -y,  x,  w]
    ];
});

const RIGHT_BASIS = [
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 1]
].map((components) => {
    const [w, x, y, z] = components;
    return [
        [w, -x, -y, -z],
        [x,  w,  z, -y],
        [y, -z,  w,  x],
        [z,  y, -x,  w]
    ];
});

const PERMUTE_TO_QUAT = [
    [0, 0, 0, 1],
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, 0]
];

const PERMUTE_FROM_QUAT = transpose(PERMUTE_TO_QUAT);

const buildDecompositionMatrix = () => {
    const columns = [];
    for (let leftIndex = 0; leftIndex < LEFT_BASIS.length; leftIndex += 1) {
        for (let rightIndex = 0; rightIndex < RIGHT_BASIS.length; rightIndex += 1) {
            const product = multiplyMatrices(LEFT_BASIS[leftIndex], RIGHT_BASIS[rightIndex]);
            columns.push(flattenMatrix(product));
        }
    }
    const size = columns.length;
    const matrix = Array.from({ length: size }, (_, row) => columns.map((column) => column[row]));
    return matrix;
};

const augmentWithIdentity = (matrix) => {
    const size = matrix.length;
    const augmented = matrix.map((row, rowIndex) => {
        const extended = row.slice();
        for (let col = 0; col < size; col += 1) {
            extended.push(rowIndex === col ? 1 : 0);
        }
        return extended;
    });
    return augmented;
};

const gaussianElimination = (matrix) => {
    const size = matrix.length;
    for (let pivot = 0; pivot < size; pivot += 1) {
        let pivotRow = pivot;
        let pivotValue = Math.abs(matrix[pivotRow][pivot]);
        for (let row = pivot + 1; row < size; row += 1) {
            const value = Math.abs(matrix[row][pivot]);
            if (value > pivotValue) {
                pivotValue = value;
                pivotRow = row;
            }
        }
        if (pivotValue < 1e-12) {
            return null;
        }
        if (pivotRow !== pivot) {
            const temp = matrix[pivot];
            matrix[pivot] = matrix[pivotRow];
            matrix[pivotRow] = temp;
        }
        const pivotElement = matrix[pivot][pivot];
        for (let col = pivot; col < matrix[pivot].length; col += 1) {
            matrix[pivot][col] /= pivotElement;
        }
        for (let row = 0; row < size; row += 1) {
            if (row === pivot) {
                continue;
            }
            const factor = matrix[row][pivot];
            if (factor === 0) {
                continue;
            }
            for (let col = pivot; col < matrix[row].length; col += 1) {
                matrix[row][col] -= factor * matrix[pivot][col];
            }
        }
    }
    return matrix;
};

const invertMatrix = (matrix) => {
    const augmented = augmentWithIdentity(matrix);
    const reduced = gaussianElimination(augmented);
    if (!reduced) {
        return null;
    }
    const size = matrix.length;
    return reduced.map((row) => row.slice(size));
};

const DECOMPOSITION_MATRIX = buildDecompositionMatrix();
const DECOMPOSITION_INVERSE = invertMatrix(DECOMPOSITION_MATRIX);

const multiplyMatrixInverse = (inverse, vector) => {
    return inverse.map((row) => row.reduce((sum, value, index) => sum + value * vector[index], 0));
};

const toQuaternionOrder = (matrix) => multiplyMatrices(PERMUTE_TO_QUAT, multiplyMatrices(matrix, PERMUTE_FROM_QUAT));

const reshapeVectorToMatrix = (vector) => {
    const matrix = [];
    for (let row = 0; row < 4; row += 1) {
        const offset = row * 4;
        matrix.push(vector.slice(offset, offset + 4));
    }
    return matrix;
};

const powerIteration = (matrix, iterations = 12) => {
    let vector = [1, 0, 0, 0];
    for (let step = 0; step < iterations; step += 1) {
        vector = normalizeVector(multiplyMatrixVector(matrix, vector));
        if (!vector) {
            return null;
        }
    }
    return vector;
};

const clampQuaternionComponent = (value) => clamp(value, -1, 1);

export const computeRotationMatrixFromUniforms = (uniforms = {}) => {
    const angles = {
        xy: Number(uniforms.u_rotXY) || 0,
        xz: Number(uniforms.u_rotXZ) || 0,
        xw: Number(uniforms.u_rotXW) || 0,
        yz: Number(uniforms.u_rotYZ) || 0,
        yw: Number(uniforms.u_rotYW) || 0,
        zw: Number(uniforms.u_rotZW) || 0
    };
    const matrices = [
        planeRotationMatrix(0, 1, angles.xy),
        planeRotationMatrix(0, 2, angles.xz),
        planeRotationMatrix(0, 3, angles.xw),
        planeRotationMatrix(1, 2, angles.yz),
        planeRotationMatrix(1, 3, angles.yw),
        planeRotationMatrix(2, 3, angles.zw)
    ];
    return composeMatrices(matrices);
};

export const decomposeRotationToDoubleQuaternion = (matrix) => {
    if (!DECOMPOSITION_INVERSE) {
        return null;
    }
    const quaternionOrder = toQuaternionOrder(matrix);
    const flat = flattenMatrix(quaternionOrder);
    const coefficients = multiplyMatrixInverse(DECOMPOSITION_INVERSE, flat);
    const outer = reshapeVectorToMatrix(coefficients);
    const leftMetric = multiplyMatrices(outer, transpose(outer));
    const leftVector = normalizeVector(powerIteration(leftMetric));
    if (!leftVector) {
        return null;
    }
    const rawRight = multiplyMatrixVector(transpose(outer), leftVector);
    const rightVector = normalizeVector(rawRight);
    if (!rightVector) {
        return null;
    }
    const left = leftVector.map(clampQuaternionComponent);
    const right = rightVector.map(clampQuaternionComponent);
    const leftNorm = Math.sqrt(left.reduce((sum, value) => sum + value * value, 0));
    const rightNorm = Math.sqrt(right.reduce((sum, value) => sum + value * value, 0));
    if (leftNorm <= 0 || rightNorm <= 0) {
        return null;
    }
    const normalizedLeft = left.map((value) => value / leftNorm);
    const normalizedRight = right.map((value) => value / rightNorm);
    if (normalizedLeft[0] < 0) {
        for (let index = 0; index < normalizedLeft.length; index += 1) {
            normalizedLeft[index] *= -1;
            normalizedRight[index] *= -1;
        }
    }
    return {
        left: normalizedLeft,
        right: normalizedRight,
        outer,
        quaternionOrder
    };
};

export const computeQuaternionBridge = (uniforms) => {
    const rotationMatrix = computeRotationMatrixFromUniforms(uniforms);
    const decomposition = decomposeRotationToDoubleQuaternion(rotationMatrix);
    if (!decomposition) {
        return null;
    }
    const { left, right } = decomposition;
    const dot = left.reduce((sum, value, index) => sum + value * right[index], 0);
    const leftAngle = 2 * Math.acos(clampQuaternionComponent(left[0]));
    const rightAngle = 2 * Math.acos(clampQuaternionComponent(right[0]));
    const bridgeVector = left.map((value, index) => value * right[index]);
    const bridgeMagnitude = Math.sqrt(bridgeVector.reduce((sum, value) => sum + value * value, 0));
    const normalizedBridge = bridgeMagnitude > 0
        ? bridgeVector.map((value) => value / bridgeMagnitude)
        : bridgeVector.map(() => 0);
    const hopf = [
        left[0] * right[0] + left[3] * right[3] - left[1] * right[1] - left[2] * right[2],
        left[0] * right[1] + left[1] * right[0] + left[2] * right[3] - left[3] * right[2],
        left[0] * right[2] - left[1] * right[3] + left[2] * right[0] + left[3] * right[1],
        left[0] * right[3] + left[1] * right[2] - left[2] * right[1] + left[3] * right[0]
    ];
    const hopfNorm = Math.sqrt(hopf.reduce((sum, value) => sum + value * value, 0));
    const hopfNormalized = hopfNorm > 0 ? hopf.map((value) => value / hopfNorm) : hopf.map(() => 0);
    return {
        rotationMatrix,
        leftQuaternion: left,
        rightQuaternion: right,
        leftAngle,
        rightAngle,
        bridgeVector,
        bridgeMagnitude,
        normalizedBridge,
        dot,
        hopfFiber: hopfNormalized
    };
};
