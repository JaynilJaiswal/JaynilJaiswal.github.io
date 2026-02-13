import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  output: "export",  // <--- Tells Next.js to create an 'index.html'
  images: {
    unoptimized: true, // <--- Required: GitHub Pages cannot optimize images on the fly
  },
};

export default nextConfig;
