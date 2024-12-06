// src/index.js
import React from "react";
import ReactDOM from "react-dom/client";
import App from "./App";  // Import your main component (App.js)
import "./index.css";  // Global CSS file

const root = ReactDOM.createRoot(document.getElementById("root"));
root.render(
  <React.StrictMode>
    <App /> {/* Render the App component */}
  </React.StrictMode>
);
