const overlayId = 'ppp-console-protect-overlay';
const bannerId = 'ppp-console-protect-banner';

const createOverlay = () => {
  if (document.getElementById(overlayId)) {
    return document.getElementById(overlayId);
  }
  const overlay = document.createElement('div');
  overlay.id = overlayId;
  overlay.style.position = 'fixed';
  overlay.style.inset = '0';
  overlay.style.background = 'rgba(10, 25, 47, 0.92)';
  overlay.style.display = 'none';
  overlay.style.zIndex = '9999';
  overlay.style.alignItems = 'center';
  overlay.style.justifyContent = 'center';
  overlay.style.flexDirection = 'column';
  overlay.style.backdropFilter = 'blur(6px)';
  overlay.style.color = '#E0FFFF';
  overlay.style.fontFamily = "Inter, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif";

  const title = document.createElement('h2');
  title.textContent = 'Protected Console Context';
  title.style.fontSize = '1.8rem';
  title.style.marginBottom = '0.5rem';

  const message = document.createElement('p');
  message.textContent = 'Developer tooling is restricted. Contact Paul Phillips or Clear Seas Solutions for authorized sandbox keys.';
  message.style.fontSize = '1rem';
  message.style.maxWidth = '28rem';
  message.style.textAlign = 'center';
  message.style.lineHeight = '1.5';

  const closeButton = document.createElement('button');
  closeButton.textContent = 'Return to Experience';
  closeButton.style.marginTop = '1.5rem';
  closeButton.style.padding = '0.75rem 1.5rem';
  closeButton.style.borderRadius = '999px';
  closeButton.style.border = '1px solid rgba(137, 207, 240, 0.5)';
  closeButton.style.background = 'rgba(8, 47, 73, 0.6)';
  closeButton.style.color = '#E0FFFF';
  closeButton.style.cursor = 'pointer';
  closeButton.addEventListener('click', () => {
    hideOverlayUI();
  });

  overlay.appendChild(title);
  overlay.appendChild(message);
  overlay.appendChild(closeButton);
  document.body.appendChild(overlay);
  return overlay;
};

const createBanner = () => {
  if (document.getElementById(bannerId)) {
    return document.getElementById(bannerId);
  }
  const banner = document.createElement('div');
  banner.id = bannerId;
  banner.textContent = 'Console interactions are recorded under PPP security policy.';
  banner.style.position = 'fixed';
  banner.style.bottom = '1.5rem';
  banner.style.left = '50%';
  banner.style.transform = 'translateX(-50%)';
  banner.style.padding = '0.65rem 1.5rem';
  banner.style.borderRadius = '999px';
  banner.style.background = 'rgba(8, 25, 47, 0.86)';
  banner.style.border = '1px solid rgba(137, 207, 240, 0.4)';
  banner.style.color = '#E0FFFF';
  banner.style.fontSize = '0.85rem';
  banner.style.letterSpacing = '0.08em';
  banner.style.textTransform = 'uppercase';
  banner.style.zIndex = '9998';
  banner.style.display = 'none';
  document.body.appendChild(banner);
  return banner;
};

const hideOverlayUI = () => {
  const overlay = document.getElementById(overlayId);
  const banner = document.getElementById(bannerId);
  if (overlay) overlay.style.display = 'none';
  if (banner) banner.style.display = 'none';
};

let consoleLocked = false;

const lockConsole = () => {
  consoleLocked = true;
  const overlay = createOverlay();
  const banner = createBanner();
  overlay.style.display = 'flex';
  banner.style.display = 'block';
};

const unlockConsole = () => {
  consoleLocked = false;
  hideOverlayUI();
};

const devtoolsDetector = {
  threshold: 160,
  check() {
    const widthGap = window.outerWidth - window.innerWidth;
    const heightGap = window.outerHeight - window.innerHeight;
    return widthGap > this.threshold || heightGap > this.threshold;
  }
};

const protectConsole = () => {
  const lockedMethods = ['log', 'warn', 'error', 'info'];
  lockedMethods.forEach((method) => {
    const original = console[method];
    Object.defineProperty(console, method, {
      configurable: true,
      enumerable: true,
      writable: true,
      value: (...args) => {
        if (consoleLocked) {
          lockConsole();
          setTimeout(() => original.apply(console, args), 10);
          return;
        }
        original.apply(console, args);
      }
    });
  });
};

const startMonitoring = () => {
  let devtoolsOpen = false;
  const runCheck = () => {
    const openNow = devtoolsDetector.check();
    if (openNow && !devtoolsOpen) {
      devtoolsOpen = true;
      lockConsole();
    } else if (!openNow && devtoolsOpen) {
      devtoolsOpen = false;
      unlockConsole();
    }
  };
  runCheck();
  setInterval(runCheck, 800);
};

if (typeof window !== 'undefined') {
  window.addEventListener('DOMContentLoaded', () => {
    protectConsole();
    startMonitoring();
  });
}
