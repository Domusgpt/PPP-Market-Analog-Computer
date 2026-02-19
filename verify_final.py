import asyncio
from playwright.async_api import async_playwright

async def run():
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page()

        page.on("console", lambda msg: print(f"CONSOLE: {msg.text}"))
        page.on("pageerror", lambda exc: print(f"PAGE ERROR: {exc}"))

        await page.goto("http://localhost:8000/hypercube-core-webgl-framework.html")

        # Give it some time to initialize
        await asyncio.sleep(5)

        # Check status message
        try:
            status = await page.inner_text("#statusMessage")
            print(f"Status Message: {status}")
        except Exception as e:
            print(f"Could not get status message: {e}")

        # Take a screenshot
        await page.screenshot(path="hypercube_final_optimized.png", full_page=True)

        await browser.close()

asyncio.run(run())
