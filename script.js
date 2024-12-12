let currentStream = null; // To store the webcam stream

function sendOTP() {
    const aadhaar = document.getElementById('aadhaar').value;
    const otpSection = document.getElementById('otp-section');
    const result = document.getElementById('result');

    // Validate Aadhaar Number format (12 digits)
    if (aadhaar.length !== 12 || !/^\d{12}$/.test(aadhaar)) {
        result.innerHTML = "Please enter a valid 12-digit Aadhaar number.";
        result.style.color = "red";
        return;
    }

    // Simulate OTP sending process
    result.innerHTML = "OTP sent successfully to your registered mobile number.";
    result.style.color = "green";

    // Show OTP input field after "sending" the OTP
    otpSection.style.display = "block";
    
    // Show webcam section for face verification
    startCamera();
}

function verifyOTP() {
    const otp = document.getElementById('otp').value;
    const result = document.getElementById('result');

    // Validate OTP format (6 digits)
    if (otp.length !== 6 || !/^\d{6}$/.test(otp)) {
        result.innerHTML = "Please enter a valid 6-digit OTP.";
        result.style.color = "red";
        return;
    }

    // Simulate OTP verification process
    if (otp === "123456") {  // Assuming OTP 123456 is valid for the demo
        result.innerHTML = "OTP verified successfully. Login complete!";
        result.style.color = "green";
    } else {
        result.innerHTML = "Invalid OTP. Please try again.";
        result.style.color = "red";
    }
}

function startCamera() {
    // Get user media (camera) and show video stream
    if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
        navigator.mediaDevices.getUserMedia({ video: true })
            .then((stream) => {
                currentStream = stream; // Store the stream for later stopping
                const video = document.getElementById('video');
                video.srcObject = stream;
                document.getElementById('webcam-section').style.display = 'block';
            })
            .catch((error) => {
                console.error('Error accessing webcam: ', error);
                alert('Could not access your webcam. Please allow camera access.');
            });
    } else {
        alert('Webcam not supported by your browser.');
    }
}

function captureImage() {
    const video = document.getElementById('video');
    const canvas = document.getElementById('canvas');
    const ctx = canvas.getContext('2d');
    
    // Set the canvas size to match the video element
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    
    // Draw the current frame from the video onto the canvas
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    
    // Optionally, you can save the captured image as a base64 string
    const imageData = canvas.toDataURL('image/png');
    
    // Display a success message (for now we simulate the facial recognition result)
    const imageResult = document.getElementById('image-result');
    imageResult.innerHTML = "Face captured successfully! (Simulated facial recognition)";
    imageResult.style.color = "green";
    
    // You can use the captured image for facial recognition in the real implementation
    console.log("Captured Image Data URL:", imageData);
}
