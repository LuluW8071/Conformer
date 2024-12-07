let isRecording = false;
let audioVisualizer;
let audioContext = new (window.AudioContext || window.webkitAudioContext)();
let mediaRecorder;
let audioData = [];
let recordBtn = document.getElementById("recordBtn");
let resultDisplay = document.getElementById("result");


recordBtn.addEventListener("click", toggleRecording);

class AudioVisualizer {
    constructor(audioContext, processFrame, processError) {
        this.audioContext = audioContext;
        this.processFrame = processFrame;
        this.connectStream = this.connectStream.bind(this);
        navigator.mediaDevices.getUserMedia({ audio: true, video: false })
            .then(this.connectStream)
            .catch((error) => {
                if (processError) {
                    processError(error);
                }
            });
    }

    connectStream(stream) {
        this.analyser = this.audioContext.createAnalyser();
        const source = this.audioContext.createMediaStreamSource(stream);
        source.connect(this.analyser);
        this.analyser.smoothingTimeConstant = 0.5;
        this.analyser.fftSize = 256;  // Increased fftSize for better frequency resolution

        this.initRenderLoop(this.analyser);
    }

    initRenderLoop() {
        const frequencyData = new Uint8Array(this.analyser.frequencyBinCount);
        const processFrame = this.processFrame || (() => { });

        const renderFrame = () => {
            this.analyser.getByteFrequencyData(frequencyData);
            processFrame(frequencyData);

            requestAnimationFrame(renderFrame);
        };
        requestAnimationFrame(renderFrame);
    }
}

const visualMainElement = document.querySelector('main');
const visualValueCount = 16;
let visualElements;

const colors = ['#ff6347', '#4682b4', '#32cd32', '#ff1493', '#ffd700', '#8a2be2', '#ff4500', '#00fa9a', '#a52a2a', '#5f9ea0', '#f0e68c', '#dda0dd', '#0000ff', '#ff00ff', '#adff2f', '#c71585'];

const createDOMElements = () => {
    for (let i = 0; i < visualValueCount; ++i) {
        const elm = document.createElement('div');
        visualMainElement.appendChild(elm);

        // Assign a color from the predefined array
        elm.style.background = colors[i % colors.length];
    }
    visualElements = document.querySelectorAll('main div');
};

const init = () => {
    // Creating initial DOM elements
    const audioContext = new AudioContext();
    const initDOM = () => {
      visualMainElement.innerHTML = '';
      createDOMElements();
    };
    initDOM();

    // Swapping values around for a better visual effect
    const dataMap = { 0: 15, 1: 10, 2: 8, 3: 9, 4: 6, 5: 5, 6: 2, 7: 1, 8: 0, 9: 4, 10: 3, 11: 7, 12: 11, 13: 12, 14: 13, 15: 14 };
    const processFrame = (data) => {
      const values = Object.values(data);
      let i;
      for (i = 0; i < visualValueCount; ++i) {
        const value = values[dataMap[i]] / 255;
        const elmStyles = visualElements[i].style;
        elmStyles.transform = `scaleY( ${value} )`;
        elmStyles.opacity = Math.max(.25, value);
      }
    };

    const processError = () => {
      visualMainElement.classList.add('error');
      visualMainElement.innerText = 'Please allow access to your microphone';
    }

    const audioVisualizer = new AudioVisualizer(audioContext, processFrame, processError);

};

// Initialize the visualizer and the DOM once on page load
document.addEventListener('DOMContentLoaded', init);

// Toggle the recording state
function toggleRecording() {
    if (isRecording) {
        stopRecording();
    } else {
        startRecording();
    }
}

function startRecording() {
    recordBtn.textContent = "Record"; // Change button text to stop recording
    isRecording = true;
    audioData = []; // Clear previous audio data

    // Access microphone and initialize both MediaRecorder and Visualizer
    navigator.mediaDevices.getUserMedia({ audio: true })
        .then((stream) => {
            // Initialize MediaRecorder
            mediaRecorder = new MediaRecorder(stream);
            mediaRecorder.ondataavailable = (event) => {
                audioData.push(event.data); // Collect audio chunks
            };

            mediaRecorder.onstop = () => {
                // When recording stops, create the audio blob
                const audioBlob = new Blob(audioData, { type: 'audio/wav' });
                transcribeAudio(audioBlob); // Send for transcription
            };

            mediaRecorder.start();
            console.log("Recording started...");

        })
        .catch((err) => {
            console.error("Error accessing microphone: ", err);
            alert("Could not access the microphone.");
            isRecording = false;
            recordBtn.textContent = "Start Recording"; // Reset button text
        });
}

function stopRecording() {
    if (!isRecording || !mediaRecorder) return;

    isRecording = false;
    mediaRecorder.stop(); // Stop recording
    console.log("Recording stopped.");

    recordBtn.textContent = "Clear"; // Reset button text
}

function transcribeAudio(audioBlob) {
    const formData = new FormData();
    formData.append("file", audioBlob, "audio_temp.wav"); // Append the audio blob

    fetch("/transcribe/", {
        method: "POST",
        body: formData,
    })
        .then(response => response.json())
        .then(data => {
            resultDisplay.textContent = data.transcription;
        })
        .catch((err) => {
            console.error("Error during transcription:", err);
            resultDisplay.textContent = "Error during transcription.";
        });
}
