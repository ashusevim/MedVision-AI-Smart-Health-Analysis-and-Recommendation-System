import "./globals.css"
import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
// import "@"
// import "";
// import { Toaster } from "@/components/ui/toaster";

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: "MedVision – AI Smart Health Analysis and Recommendation System",
  description: "Advanced AI-powered health analysis system with symptom checker, vital signs monitoring, risk assessment, and personalized health recommendations.",
  keywords: ["MedVision", "Health AI", "Symptom Checker", "Health Analysis", "AI Healthcare", "Medical AI", "Health Monitoring"],
  authors: [{ name: "MedVision AI Team" }],
  icons: {
    icon: "/logo.png",
  },
  openGraph: {
    title: "MedVision – AI Smart Health Analysis",
    description: "Your personal AI health assistant with advanced symptom analysis and recommendations",
    type: "website",
  },
  twitter: {
    card: "summary_large_image",
    title: "MedVision – AI Smart Health Analysis",
    description: "Your personal AI health assistant with advanced symptom analysis and recommendations",
  },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" suppressHydrationWarning>
      <body
        className={`${geistSans.variable} ${geistMono.variable} antialiased bg-background text-foreground`}
      >
        {children}
        {/* <Toaster /> */}
      </body>
    </html>
  );
}
