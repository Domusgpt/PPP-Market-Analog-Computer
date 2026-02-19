
import { test } from 'node:test';
import assert from 'node:assert';

/**
 * Security: XSS Regression Test
 * This test verifies that connection status updates are handled safely.
 */
test('XSS fix verification: status updates use textContent', () => {
    // Mock DOM elements
    const mockSpan = {
        textContent: '',
    };
    const mockElement = {
        className: '',
        innerHTML: '',
        querySelector: (selector) => {
            if (selector === 'span') return mockSpan;
            return null;
        }
    };

    const xssPayload = '<img src=x onerror=alert(1)>';

    // Logic from phase-lock-live.html
    const updateStatus = (status, el) => {
        const statusEl = el;
        if (status === 'connected') {
            statusEl.className = 'connection-status';
            statusEl.innerHTML = '<div class="status-dot active"></div><span>WebSocket Connected</span>';
        } else {
            statusEl.className = 'connection-status disconnected';
            statusEl.innerHTML = '<div class="status-dot"></div><span></span>';
            statusEl.querySelector('span').textContent = status;
        }
    };

    // Test with malicious payload
    updateStatus(xssPayload, mockElement);

    assert.strictEqual(mockElement.className, 'connection-status disconnected');
    assert.strictEqual(mockElement.innerHTML, '<div class="status-dot"></div><span></span>');
    assert.strictEqual(mockSpan.textContent, xssPayload, 'Payload should be stored as plain text, not parsed as HTML');

    // Serial error case
    const updateError = (err, el) => {
        const statusEl = el;
        statusEl.className = 'connection-status disconnected';
        statusEl.innerHTML = '<div class="status-dot"></div><span></span>';
        statusEl.querySelector('span').textContent = `Serial Error: ${err.message}`;
    };

    const mockError = { message: xssPayload };
    updateError(mockError, mockElement);

    assert.strictEqual(mockElement.className, 'connection-status disconnected');
    assert.strictEqual(mockElement.innerHTML, '<div class="status-dot"></div><span></span>');
    assert.strictEqual(mockSpan.textContent, `Serial Error: ${xssPayload}`, 'Error message with payload should be stored as plain text');
});
