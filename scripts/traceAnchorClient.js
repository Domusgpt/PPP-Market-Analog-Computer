const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

export const createTraceAnchorClient = ({ endpoint, fetcher, retries = 3, backoffMs = 400 } = {}) => {
    const request = typeof fetcher === 'function' ? fetcher : (typeof fetch === 'function' ? fetch : null);

    const anchorBatchRoot = async ({ root, metadata = {} }) => {
        if (!request) {
            throw new Error('No fetch implementation available for TRACE anchoring.');
        }
        let attempt = 0;
        while (attempt <= retries) {
            try {
                const response = await request(endpoint, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ root, metadata })
                });
                if (!response.ok) {
                    throw new Error(`TRACE anchor failed (${response.status})`);
                }
                const payload = await response.json();
                return {
                    hash: payload.hash || payload.anchor || payload.root || root,
                    anchoredAt: payload.anchoredAt || Date.now()
                };
            } catch (error) {
                attempt += 1;
                if (attempt > retries) {
                    throw error;
                }
                await sleep(backoffMs * attempt);
            }
        }
        throw new Error('TRACE anchor failed after retries.');
    };

    return { anchorBatchRoot };
};
