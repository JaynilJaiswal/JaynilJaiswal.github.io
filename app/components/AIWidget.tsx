'use client';

import Script from 'next/script';

export default function AIWidget() {
  return (
    <Script 
      src="/widget.js" 
      strategy="lazyOnload" 
      onLoad={() => {
        // This safely initializes your class once the script finishes loading in the browser
        if (typeof window !== "undefined" && window.PortfolioAIWidget) {
          const apiUrl = process.env.NEXT_PUBLIC_AI_GATEWAY_URL || "http://localhost:8080";
          const widget = new window.PortfolioAIWidget(apiUrl);
          widget.init();
        }
      }}
    />
  );
}