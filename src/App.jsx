import React from 'react';
import Chatbot from './components/Chatbot';

function App() {
  return (
    <div className="min-h-screen bg-gray-100">
      {/* Your website content goes here */}
      <header className="bg-white shadow">
        <div className="max-w-7xl mx-auto px-4 py-6 sm:px-6 lg:px-8">
          <h1 className="text-3xl font-bold text-gray-900">My Website</h1>
        </div>
      </header>

      <main className="max-w-7xl mx-auto px-4 py-6 sm:px-6 lg:px-8">
        <div className="bg-white overflow-hidden shadow rounded-lg">
          <div className="px-4 py-5 sm:p-6">
            <h2 className="text-xl font-semibold text-gray-800 mb-4">Welcome to My Site</h2>
            <p className="text-gray-600">
              This is a sample website with a RAG-powered chatbot. Click the floating button
              on the bottom right to start chatting with the AI assistant.
            </p>
            <div className="mt-6 p-4 bg-blue-50 rounded-lg">
              <h3 className="font-medium text-blue-800">Try asking:</h3>
              <ul className="list-disc list-inside text-blue-700 mt-2">
                <li>What documents do you have access to?</li>
                <li>Explain the key concepts from the documents</li>
                <li>Summarize the main points about [topic]</li>
              </ul>
            </div>
          </div>
        </div>
      </main>

      {/* Chatbot component - this will render the floating button and chat window */}
      <Chatbot />
    </div>
  );
}

export default App;