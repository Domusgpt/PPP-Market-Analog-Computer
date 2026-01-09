import { defineConfig } from 'vitest/config';

export default defineConfig({
  test: {
    globals: true,
    environment: 'happy-dom',
    include: ['src/**/*.test.ts', 'src/**/*.test.tsx'],
    exclude: ['node_modules', 'dist'],
    setupFiles: ['./src/test/setup.ts'],
    // Mock Web Crypto API for Node environment
    deps: {
      inline: ['@testing-library/react'],
    },
  },
});
