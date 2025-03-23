import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import axios from 'axios';
import './home.css';
import config from '../../config';
import imgLoading from '../../assets/loading_2.gif';

function Home() {
  const navigate = useNavigate();
  // Using normal HTTP requests
  const username = sessionStorage.getItem('username');
  const [question, setQuestion] = useState('');
  const [file, setFile] = useState(null);
  const role = sessionStorage.getItem('role');
  const [loading, setLoading] = useState(false);
  const [upLoading, setupLoading] = useState(false);
  const [isDisabled, setIsDisabled] = useState(false);
  const [messages, setMessages] = useState([]);


  // const [answer, setAnswer] = useState([]);
  // const [previousQuestion, setPreviousQuestion] = useState({ question: '', answer: [] });

  const handleLogout = () => {
    sessionStorage.clear();
    navigate('/');
  };

  useEffect(() => {
    if (!loading || messages.length <= 0) {
      return;
    }
    let intervalId;
    let n = 1;
    intervalId = setInterval(() => {
      setMessages((messages) => messages.map((item, index) => {
        if (index === messages.length - 1) {
          item.answer = ".".repeat(n);
        }
        return item;
      }));
      n++;
      if (n > 4) {
        n = 1;
      }
    }, 500);
    return () => {
      if (intervalId) clearInterval(intervalId);
    };
  }, [loading]);

  const handleAsk = async () => {
    if (!question.trim() || true === loading) return;
    setLoading(true)
    const message = {
      question,
      answer: ''
    }
    setMessages([
      ...messages,
      message
    ])

    try {
      const response = await axios.post(`${config.backendUrl}/api/ask`, { question });
      if (response) {
        message.answer = response.data.summary;
      }
    } catch (error) {
      message.answer = 'Error fetching answer.';
    } finally {
      setQuestion('');
      setIsDisabled(false);
      setLoading(false);
      // clearInterval(interval);
    }
  };

  const handleKeyDown = async (evt) => {
    if (evt.key === 'Enter') {
      handleAsk();
    }
  }

  const handleFileUpload = async (e) => {
    e.preventDefault();
    if (!file) return;
    setupLoading(true)
    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await axios.post(`${config.backendUrl}/api/upload`, formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });
      if (response.data.message) {
        alert(`${response.data.message}`);
      }
      if (response.data.error) {
        alert(`${response.data.error}`);
      }
    } catch (error) {
      alert('Error uploading file.');
    } finally {
      setupLoading(false)
    }
  };

  return (
    <div className="home-container">
      <div className="header" style={{ textAlign: 'right', marginBottom: '20px' }}>
        <span className="username">Hello, {username}</span>
        <button className="logout-button" onClick={handleLogout} disabled={isDisabled}>Logout</button>
      </div>
      <div style={{ display: 'flex', flexDirection: 'row', alignItems: 'center', justifyContent: 'center' }} className='chat-container'>
        <div className="question-container flex-2" >
          <div className="previous-question-container flex-1" >
            {
              messages.map((item) => (
                <div className='message-item'>
                  <div className='question-item'>
                    {item.question}
                  </div>
                  <div className='answer-item'>
                    {item.answer}
                  </div>
                </div>
              ))
            }
          </div>
          <div className='question-content'>
            <input
              type="text"
              placeholder="Type your question here..."
              className="question-input"
              value={question}
              onClick={() => { setQuestion(''); setIsDisabled(true); }}
              onChange={(e) => setQuestion(e.target.value)}
              onKeyDown={handleKeyDown}
              disabled={loading}
            />

            <button
              className="submit-button"
              disabled={loading}
              onClick={handleAsk}
            >
              {loading ? 'Loading...' : 'Submit'}
            </button>
          </div>
        </div>

        {role === 'admin' && (
          <div className="upload-container flex-1" style={{ marginBottom: '20px', textAlign: 'center' }}>
            <h2>Upload Document</h2>
            {
              true === upLoading ?
                <div className='upload-loading' >
                  <img src={imgLoading}></img>
                </div>
                : <form onSubmit={handleFileUpload}>
                  <input
                    type="file"
                    accept=".pdf,.docx"
                    onChange={(e) => setFile(e.target.files[0])}
                    style={{ marginBottom: '10px' }}
                  />
                  <button
                    type="submit"
                    className="upload-button"
                    disabled={upLoading}
                  >
                    Upload
                  </button>
                </form>
            }
          </div>
        )}
      </div>
    </div>
  );
}

export default Home;
