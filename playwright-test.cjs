const { chromium } = require('playwright');

(async () => {
    const browser = await chromium.launch({ headless: true });
    const context = await browser.newContext();
    const page = await context.newPage();

    // Capture all console messages
    page.on('console', msg => {
        console.log('[' + msg.type() + '] ' + msg.text());
    });

    // Capture any page errors
    page.on('pageerror', err => {
        console.log('[PAGE ERROR] ' + err.message);
    });

    // Capture failed requests
    page.on('requestfailed', request => {
        console.log('[REQUEST FAILED] ' + request.url() + ': ' + request.failure().errorText);
    });

    console.log('Navigating to page...');
    try {
        await page.goto('https://domusgpt.github.io/ppp-info-site/', {
            waitUntil: 'networkidle',
            timeout: 30000
        });
        console.log('Page loaded, waiting for JS execution...');

        // Wait a bit for async operations
        await page.waitForTimeout(5000);

        // Check canvas state
        const canvasInfo = await page.evaluate(() => {
            const canvas = document.getElementById('gce-canvas');
            if (!canvas) return { error: 'Canvas not found' };
            const ctx = canvas.getContext('webgl') || canvas.getContext('webgl2');
            return {
                width: canvas.width,
                height: canvas.height,
                hasContext: !!ctx,
                contextType: ctx ? (ctx instanceof WebGL2RenderingContext ? 'webgl2' : 'webgl') : null
            };
        });
        console.log('Canvas info: ' + JSON.stringify(canvasInfo));

        // Check loading element
        const loadingState = await page.evaluate(() => {
            const loading = document.getElementById('loading');
            return {
                display: loading ? getComputedStyle(loading).display : 'not found',
                text: loading ? loading.innerText : ''
            };
        });
        console.log('Loading state: ' + JSON.stringify(loadingState));

        // Take a screenshot
        await page.screenshot({ path: 'playwright-screenshot.png' });
        console.log('Screenshot saved to playwright-screenshot.png');

    } catch (err) {
        console.log('Error: ' + err.message);
    }

    await browser.close();
})();
