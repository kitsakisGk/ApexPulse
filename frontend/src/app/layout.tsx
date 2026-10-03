import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "ApexPulse — Live CS2 Win Probability",
  description:
    "Real-time esports telemetry and live win-probability for Counter-Strike 2.",
};

export default function RootLayout({
  children,
}: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en" className="dark">
      <body className="min-h-screen bg-base-950 font-sans antialiased">
        {children}
      </body>
    </html>
  );
}
