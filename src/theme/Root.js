import React from 'react';
import Chatbot from '../components/Chatbot';

function Root({ children }) {
  return (
    <>
      {children}
      <Chatbot />
    </>
  );
}

export default Root;