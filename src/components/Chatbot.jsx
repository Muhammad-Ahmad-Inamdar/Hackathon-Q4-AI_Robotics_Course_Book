import React, { useState, useRef, useEffect } from 'react';
import './Chatbot.css';

const Chatbot = () => {
  const [isOpen, setIsOpen] = useState(false);
  const [messages, setMessages] = useState([]);
  const [inputValue, setInputValue] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);
  const messagesEndRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const sendMessage = async () => {
    if (!inputValue.trim() || isLoading) return;

    const userMessage = {
      id: Date.now(),
      role: 'user',
      content: inputValue,
      timestamp: new Date()
    };

    // Add user message to chat
    setMessages(prev => [...prev, userMessage]);
    setInputValue('');
    setIsLoading(true);
    setError(null);

    try {
      // Prepare messages for API (only send last few messages to avoid token limits)
      const apiMessages = [{ role: 'user', content: inputValue }];

      const response = await fetch('https://99-muhammad-rag-chatbot-backend.hf.space/chat', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          messages: apiMessages,
          chapter: null // Optional chapter filter, can be added based on user selection
        })
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();

      const botMessage = {
        id: Date.now() + 1,
        role: 'assistant',
        content: data.response,
        sources: data.sources,
        sources_used: data.sources_used,
        filter_applied: data.filter_applied,
        context_depth: data.context_depth,
        timestamp: new Date()
      };

      setMessages(prev => [...prev, botMessage]);
    } catch (err) {
      setError('Failed to send message. Please try again.');
      console.error('Error sending message:', err);
    } finally {
      setIsLoading(false);
    }
  };

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  const toggleChat = () => {
    setIsOpen(!isOpen);
  };

  const closeChat = () => {
    setIsOpen(false);
  };

  return (
    <>
      {/* Floating Chat Button */}
      {!isOpen && (
        <button
          onClick={toggleChat}
          className="chatbot-floating-btn"
          aria-label="Open chat"
        >
          <svg xmlns="http://www.w3.org/2000/svg" className="w-6 h-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 12h.01M12 12h.01M16 12h.01M21 12c0 4.418-4.03 8-9 8a9.863 9.863 0 01-4.255-.949L3 20l1.395-3.72C3.512 15.042 3 13.574 3 12c0-4.418 4.03-8 9-8s9 3.582 9 8z" />
          </svg>
        </button>
      )}

      {/* Chat Window */}
      {isOpen && (
        <div className="chatbot-window">
          {/* Chat Header */}
          <div className="chatbot-header">
            <h3>Chat with RAG Assistant</h3>
            <button
              onClick={closeChat}
              className="chatbot-close-btn"
              aria-label="Close chat"
            >
              <svg xmlns="http://www.w3.org/2000/svg" className="w-5 h-5" viewBox="0 0 20 20" fill="currentColor">
                <path fillRule="evenodd" d="M4.293 4.293a1 1 0 011.414 0L10 8.586l4.293-4.293a1 1 0 111.414 1.414L11.414 10l4.293 4.293a1 1 0 01-1.414 1.414L10 11.414l-4.293 4.293a1 1 0 01-1.414-1.414L8.586 10 4.293 5.707a1 1 0 010-1.414z" clipRule="evenodd" />
              </svg>
            </button>
          </div>

          {/* Messages Container */}
          <div className="chatbot-messages">
            {messages.length === 0 ? (
              <div className="empty-state">
                <p>Hello! I'm your RAG-powered assistant.</p>
                <p>Ask me anything about your documents.</p>
              </div>
            ) : (
              messages.map((message) => (
                <div
                  key={message.id}
                  className={`chatbot-message ${message.role === 'user' ? 'user' : 'assistant'}`}
                >
                  <div className="chatbot-avatar"></div>
                  <div className="chatbot-bubble-container">
                    <div
                      className={`chatbot-bubble ${message.role === 'user' ? 'user' : 'assistant'}`}
                    >
                      <p className="whitespace-pre-wrap">{message.content}</p>

                      {/* Show additional RAG info for bot responses */}
                      {message.role === 'assistant' && message.sources && (
                        <div className="chatbot-sources">
                          <details>
                            <summary>Show sources</summary>
                            <div>
                              {message.sources.slice(0, 3).map((source, idx) => (
                                <div key={idx} className="chatbot-source-item">
                                  {source}
                                </div>
                              ))}
                              {message.sources.length > 3 && (
                                <p>
                                  + {message.sources.length - 3} more sources
                                </p>
                              )}
                            </div>
                          </details>
                          {message.filter_applied && (
                            <p>Filter: {message.filter_applied}</p>
                          )}
                          {message.context_depth && (
                            <p>Context: {message.context_depth}</p>
                          )}
                        </div>
                      )}
                    </div>
                  </div>
                </div>
              ))
            )}

            {isLoading && (
              <div className="chatbot-message assistant chatbot-loading">
                <div className="chatbot-avatar assistant"></div>
                <div className="chatbot-bubble-container">
                  <div className="chatbot-typing-indicator">
                    <span className="chatbot-typing-text">Bot is typing...</span>
                    <div className="chatbot-typing-dot"></div>
                    <div className="chatbot-typing-dot"></div>
                    <div className="chatbot-typing-dot"></div>
                  </div>
                </div>
              </div>
            )}

            <div ref={messagesEndRef} />
          </div>

          {/* Error Message */}
          {error && (
            <div className="chatbot-error">
              {error}
              <button
                onClick={() => setError(null)}
                className="close-btn"
              >
                ×
              </button>
            </div>
          )}

          {/* Input Area */}
          <div className="chatbot-input-area">
            <div className="chatbot-input-container">
              <textarea
                value={inputValue}
                onChange={(e) => setInputValue(e.target.value)}
                onKeyPress={handleKeyPress}
                placeholder="Type your message..."
                className="chatbot-input"
                rows="1"
                disabled={isLoading}
              />
              <button
                onClick={sendMessage}
                disabled={!inputValue.trim() || isLoading}
                className="chatbot-send-btn"
              >
                <svg xmlns="http://www.w3.org/2000/svg" className="w-5 h-5" viewBox="0 0 20 20" fill="currentColor">
                  <path d="M10.894 2.553a1 1 0 00-1.788 0l-7 14a1 1 0 001.169 1.409l5-1.429A1 1 0 009 15.571V11a1 1 0 112 0v4.571a1 1 0 00.725.962l5 1.428a1 1 0 001.17-1.408l-7-14z" />
                </svg>
              </button>
            </div>
            <p className="chatbot-powered-text">
              Powered by RAG Vector Database
            </p>
          </div>
        </div>
      )}
    </>
  );
};

export default Chatbot;