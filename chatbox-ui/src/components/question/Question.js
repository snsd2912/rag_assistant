import React, { useState } from 'react';
import axios from 'axios';
import './question.css';

function AskQuestion() {
  const [question, setQuestion] = useState('');
  const [answer, setAnswer] = useState('');

  const handleSubmit = async (e) => {
    e.preventDefault();
    try {
      const response = await axios.post('http://localhost:8000/api/ask', { question });
      setAnswer(response.data.answer);
    } catch (error) {
      setAnswer('Error fetching answer.');
    }
  };

  return (
    <div>
      <h1>Ask a Question</h1>
      <form onSubmit={handleSubmit}>
        <input
          type="text"
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          placeholder="Please type your question here..."
        />
        <button type="submit">Submit</button>
      </form>
      <div>
        <h2>Answer:</h2>
        <p>{answer}</p>
      </div>
    </div>
  );
}

export default AskQuestion;
