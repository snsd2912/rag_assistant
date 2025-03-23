import React, { useEffect } from 'react';
import ReactDOM from 'react-dom';
import { BrowserRouter as Router, Routes, Route, useNavigate } from 'react-router-dom';
import Login from './components/login/Login';
import Home from './components/home/Home';
import './index.css';

function App() {
  const navigate = useNavigate();

  useEffect(() => {
    const username = sessionStorage.getItem('username');
    if (username) {
      navigate('/home');
    } else {
      navigate('/');
    }
  }, [navigate]);

  return (
    <Routes>
      <Route path="/" element={<Login />} />
      <Route path="/home" element={<Home />} />
    </Routes>
  );
}

function Root() {
  return (
    <Router>
      <App />
    </Router>
  );
}

ReactDOM.render(<Root />, document.getElementById('root'))
