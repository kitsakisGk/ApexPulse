// ESLint flat config.
//
// `next lint` is deprecated and, with no config present, stops to ask which
// preset to use. In CI there is nobody to answer, so the job hangs until it
// times out. Running the ESLint CLI against an explicit config is the
// replacement Next recommends, and it never prompts.

import { dirname } from "node:path";
import { fileURLToPath } from "node:url";

import { FlatCompat } from "@eslint/eslintrc";

const compat = new FlatCompat({
  baseDirectory: dirname(fileURLToPath(import.meta.url)),
});

const config = [
  {
    ignores: [".next/**", "node_modules/**", "out/**", "next-env.d.ts"],
  },
  ...compat.extends("next/core-web-vitals", "next/typescript"),
];

export default config;
