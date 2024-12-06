import React, { useRef, useEffect, useState } from "react";
import "./App.css";

const words = [
  { hindi: "नमस्ते", english: "Hello" },
  { hindi: "पानी", english: "Water" },
  { hindi: "खुश", english: "Happy" },
  { hindi: "किताब", english: "Book" },
  { hindi: "घर", english: "House" },
];

function App() {
  const canvasRef = useRef(null);
  const videoRef = useRef(null);
  const [wordIndex, setWordIndex] = useState(0);
  const [showWebcam, setShowWebcam] = useState(false);
  const [message, setMessage] = useState("");
  const [isRunning, setIsRunning] = useState(false);

  // Resize canvas dynamically
  const resizeCanvas = () => {
    const canvas = canvasRef.current;
    if (canvas) {
      canvas.width = window.innerWidth;
      canvas.height = window.innerHeight;
    }
  };

  // Draw a dot and display a word
  const drawDot = () => {
    const canvas = canvasRef.current;
    const ctx = canvas.getContext("2d");

    ctx.clearRect(0, 0, canvas.width, canvas.height);

    const x = Math.random() * canvas.width;
    const y = Math.random() * canvas.height;

    ctx.beginPath();
    ctx.arc(x, y, 10, 0, Math.PI * 2);
    ctx.fillStyle = "red";
    ctx.fill();

    const currentWord = words[wordIndex];
    ctx.font = "20px Arial";
    ctx.fillStyle = "white";
    ctx.fillText(`${currentWord.hindi} / ${currentWord.english}`, x + 15, y);
  };

  // Start the webcam for authentication
  const startWebcam = async () => {
    const video = videoRef.current;

    try {
      // Start webcam
      const stream = await navigator.mediaDevices.getUserMedia({
        video: true,
      });
      video.srcObject = stream;
      video.play();
      setMessage("Look at the camera for authentication.");

      // Authentication process
      setTimeout(() => {
        setMessage("Authentication successful!");
        stopWebcam(); // Stop the webcam after 10 seconds
      }, 10000); // 10 seconds for authentication
    } catch (error) {
      console.error("Error starting webcam:", error);

      // Handle different webcam access errors
      if (error.name === "NotAllowedError") {
        setMessage("Webcam access denied. Please allow camera permissions.");
      } else if (error.name === "NotFoundError") {
        setMessage("No webcam found. Please connect a webcam.");
      } else {
        setMessage("Error accessing webcam! Ensure it's connected.");
      }
    }
  };

  // Stop the webcam
  const stopWebcam = () => {
    const video = videoRef.current;
    if (video.srcObject) {
      const tracks = video.srcObject.getTracks();
      tracks.forEach((track) => track.stop());
      video.srcObject = null;
    }
    setShowWebcam(false);
    setIsRunning(false);
  };

  // Animation loop for dots
  useEffect(() => {
    if (isRunning && wordIndex < words.length) {
      const timer = setTimeout(() => {
        drawDot();
        setWordIndex(wordIndex + 1);
      }, 3000); // 3 seconds delay per dot
      return () => clearTimeout(timer);
    } else if (wordIndex >= words.length) {
      setShowWebcam(true);
      setMessage("Preparing for authentication...");
      startWebcam(); // Start webcam authentication after words are done
    }
  }, [wordIndex, isRunning]);

  // Resize canvas on window resize
  useEffect(() => {
    resizeCanvas();
    window.addEventListener("resize", resizeCanvas);
    return () => window.removeEventListener("resize", resizeCanvas);
  }, []);

  const handleStart = () => {
    setIsRunning(true);
    setWordIndex(0);
    setMessage("Follow the instructions.");
  };

  return (
    <div className="app">
      <canvas ref={canvasRef} className="canvas"></canvas>

      {showWebcam && (
        <div className="webcam-container">
          {/* Webcam Video */}
          <video
            ref={videoRef}
            autoPlay
            playsInline
            style={{
              width: "100%",
              height: "auto",
              border: "2px solid #4CAF50",
              borderRadius: "8px",
            }}
          ></video>

          {/* Message below webcam */}
          <div
            className="webcam-message"
            style={{
              marginTop: "10px",
              fontSize: "18px",
              color: "#FFFFFF",
              textAlign: "center",
              backgroundColor: "#4CAF50",
              padding: "10px",
              borderRadius: "5px",
            }}
          >
            {message || "Please look at the camera for authentication."}
          </div>
        </div>
      )}

      {message && <div className="message">{message}</div>}

      <div className="controls">
        <button id="start-btn" onClick={handleStart} disabled={isRunning}>
          Start
        </button>
      </div>
    </div>
  );
}

export default App;
