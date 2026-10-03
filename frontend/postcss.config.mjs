// Tailwind 4 ships its own PostCSS plugin and handles vendor prefixing
// internally, so autoprefixer is no longer a separate step.
const config = {
  plugins: {
    "@tailwindcss/postcss": {},
  },
};

export default config;
