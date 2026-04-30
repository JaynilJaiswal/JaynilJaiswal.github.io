'use client';

import { useState, useRef, useEffect } from 'react';
import ReactMarkdown from 'react-markdown';

type Message = { role: 'user' | 'bot'; content: string };

export default function InlineAIChat() {
  const [messages, setMessages] = useState<Message[]>([
    { 
      role: 'bot', 
      content: "Hi! I'm the AI Gateway for Jaynil's portfolio. Ask me anything about his experience engineering AI Infrastructure and Distributed Systems." 
    }
  ]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const chatContainerRef = useRef<HTMLDivElement>(null);

  // Auto-scroll inside the chat container
  useEffect(() => {
    if (chatContainerRef.current) {
      chatContainerRef.current.scrollTo({
        top: chatContainerRef.current.scrollHeight,
        behavior: 'smooth'
      });
    }
  }, [messages]);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim() || isLoading) return;

    const userQuestion = input.trim();
    setMessages((prev) => [...prev, { role: 'user', content: userQuestion }]);
    setInput('');
    setIsLoading(true);

    try {
      const apiUrl = process.env.NEXT_PUBLIC_AI_GATEWAY_URL || "http://localhost:8080";
      const response = await fetch(`${apiUrl}/ask`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question: userQuestion }),
      });

      if (!response.ok) {
        if (response.status === 429) throw new Error("Rate limit exceeded. Please wait a moment.");
        throw new Error(`Server Error: ${response.status}`);
      }

      const data = await response.json();
      setMessages((prev) => [...prev, { role: 'bot', content: data.answer }]);
    } catch (error: any) {
      console.error("Chat Error:", error);
      setMessages((prev) => [...prev, { role: 'bot', content: `Oops: ${error.message}` }]);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="flex flex-col h-full w-full bg-gray-50 dark:bg-dark-900 rounded-lg overflow-hidden border border-gray-200 dark:border-gray-700">
      
      {/* Messages Area */}
      <div ref={chatContainerRef} className="flex-1 p-6 overflow-y-auto space-y-6">
        {messages.map((msg, idx) => (
          <div key={idx} className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}>
            <div className={`max-w-[85%] rounded-2xl px-5 py-4 text-sm md:text-base shadow-sm ${
              msg.role === 'user' 
                ? 'bg-blue-600 text-white rounded-br-sm' 
                : 'bg-white dark:bg-dark-800 text-gray-800 dark:text-gray-100 border border-gray-100 dark:border-gray-700 rounded-bl-sm'
            }`}>
              
              {/* --- MARKDOWN RENDERER --- */}
              {msg.role === 'user' ? (
                // Render plain text for user messages
                <p>{msg.content}</p> 
              ) : (
                // Render Markdown for the AI responses with Tailwind styling rules
                <ReactMarkdown 
                  components={{
                    p: ({node, ...props}) => <p className="mb-3 last:mb-0 leading-relaxed" {...props} />,
                    ul: ({node, ...props}) => <ul className="list-disc pl-6 mb-4 space-y-1" {...props} />,
                    ol: ({node, ...props}) => <ol className="list-decimal pl-6 mb-4 space-y-1" {...props} />,
                    li: ({node, ...props}) => <li className="leading-relaxed" {...props} />,
                    h1: ({node, ...props}) => <h1 className="text-xl font-bold mb-2 mt-4" {...props} />,
                    h2: ({node, ...props}) => <h2 className="text-lg font-bold mb-2 mt-4" {...props} />,
                    h3: ({node, ...props}) => <h3 className="text-md font-bold mb-2 mt-3" {...props} />,
                    strong: ({node, ...props}) => <strong className="font-semibold text-gray-900 dark:text-white" {...props} />,
                    a: ({node, ...props}) => <a className="text-blue-500 hover:underline" target="_blank" rel="noopener noreferrer" {...props} />,
                    code: ({node, inline, className, children, ...props}: any) => {
                      return inline ? (
                        <code className="bg-gray-100 dark:bg-gray-700 text-pink-600 dark:text-pink-400 px-1.5 py-0.5 rounded font-mono text-sm" {...props}>
                          {children}
                        </code>
                      ) : (
                        <div className="bg-gray-800 text-gray-100 p-3 rounded-lg overflow-x-auto mb-4 text-sm font-mono my-2">
                          <code {...props}>{children}</code>
                        </div>
                      )
                    }
                  }}
                >
                  {msg.content}
                </ReactMarkdown>
              )}
            </div>
          </div>
        ))}
        
        {/* Loading Indicator */}
        {isLoading && (
          <div className="flex justify-start">
            <div className="max-w-[80%] bg-white dark:bg-dark-800 border border-gray-100 dark:border-gray-700 rounded-2xl rounded-bl-sm px-4 py-3 flex space-x-2 items-center shadow-sm">
              <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '0ms' }}></div>
              <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '150ms' }}></div>
              <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '300ms' }}></div>
            </div>
          </div>
        )}
      </div>

      {/* Input Form */}
      <form onSubmit={handleSubmit} className="p-4 bg-white dark:bg-dark-800 border-t border-gray-200 dark:border-gray-700 flex gap-3">
        <input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="e.g., Tell me about the SLAM system..."
          className="flex-1 bg-gray-100 dark:bg-dark-900 text-gray-800 dark:text-gray-100 px-4 py-3 rounded-full focus:outline-none focus:ring-2 focus:ring-blue-500 border border-transparent dark:border-gray-700 transition-all"
          disabled={isLoading}
        />
        <button
          type="submit"
          disabled={isLoading || !input.trim()}
          className="bg-blue-600 hover:bg-blue-700 disabled:bg-blue-400 disabled:cursor-not-allowed text-white px-6 py-3 rounded-full font-medium transition-colors flex items-center justify-center"
        >
          <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 19l9 2-9-18-9 18 9-2zm0 0v-8" />
          </svg>
        </button>
      </form>
    </div>
  );
}