import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "SignVoice — Communication without barriers",
  description: "An accessible ISL communication project. Recognition model integration is under development."
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
