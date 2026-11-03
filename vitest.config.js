import { defineConfig } from 'vitest/config';
import TestDriver from 'testdriverai/vitest';
import { config } from 'dotenv';

config();

export default defineConfig({
  test: {
    testTimeout: 900000,
    hookTimeout: 900000,
    reporters: [
      'default',
      TestDriver(),
    ],
    setupFiles: ['testdriverai/vitest/setup'],
  },
});
