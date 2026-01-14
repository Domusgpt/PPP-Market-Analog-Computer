const { chromium } = require('playwright');

(async () => {
    console.log('Starting browser...');

    const browser = await chromium.launch({
        headless: true,
        args: ['--no-sandbox', '--disable-setuid-sandbox']
    });

    const context = await browser.newContext({
        ignoreHTTPSErrors: true,
        bypassCSP: true
    });

    const page = await context.newPage();

    // Capture console messages
    page.on('console', msg => {
        const type = msg.type();
        const text = msg.text();
        if (type === 'error' || type === 'warning' || text.includes('===') || text.includes('Vertex') || text.includes('NDC') || text.includes('error') || text.includes('Error')) {
            console.log('[' + type.toUpperCase() + '] ' + text);
        }
    });

    page.on('pageerror', err => {
        console.log('[PAGE ERROR] ' + err.message);
    });

    console.log('Navigating to page...');

    try {
        await page.goto('http://localhost:8080/', {
            waitUntil: 'networkidle',
            timeout: 60000
        });

        console.log('Page loaded, waiting for rendering...');
        await page.waitForTimeout(3000);

        // Get canvas info
        const info = await page.evaluate(() => {
            const canvas = document.getElementById('gce-canvas');
            const loading = document.getElementById('loading');

            return {
                canvasExists: !!canvas,
                canvasWidth: canvas ? canvas.width : 0,
                canvasHeight: canvas ? canvas.height : 0,
                loadingVisible: loading ? getComputedStyle(loading).display !== 'none' : false,
                loadingText: loading ? loading.innerText : '',
                hasWebGL: !!canvas && !!(canvas.getContext('webgl') || canvas.getContext('experimental-webgl'))
            };
        });

        console.log('Page info:', JSON.stringify(info, null, 2));

        // Take screenshot
        await page.screenshot({ path: '/home/user/ppp-info-site/screenshot.png', fullPage: true });
        console.log('Screenshot saved to screenshot.png');

    } catch (err) {
        console.log('Error: ' + err.message);
    }

    await browser.close();
    console.log('Done');
})();
