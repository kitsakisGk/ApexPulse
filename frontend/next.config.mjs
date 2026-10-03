/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  // The API runs as a separate service, so its base URL is configuration rather
  // than something baked into the bundle at build time.
  env: {
    NEXT_PUBLIC_API_URL: process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000",
    NEXT_PUBLIC_WS_URL: process.env.NEXT_PUBLIC_WS_URL ?? "ws://localhost:8000",
  },
};

export default nextConfig;
