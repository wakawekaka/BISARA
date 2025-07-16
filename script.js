// Global variables (no changes here)
let bisaraNetModel;
let hands;
const SEQUENCE_LENGTH = 30;
const NUM_LANDMARKS = 21;
const sequence = [];
let currentPrediction = '';
const confidenceThreshold = 0.95;
let lastAppendedWord = '';
let isWaiting = false;
let selectedVoice = null;
let periodTimer;
let isAutoTranslateEnabled = true;

const actions = {
    0: 'blank',
    1: 'halo',
    2: 'pagi',
    3: 'selamat',
    4: 'semua',
};

// --- MODIFIED SECTION ---

// Function to load content dynamically from fragments
function loadContent(url) {
    // Stop the camera if it's running to prevent resource leaks
    if (window.currentStream) {
        window.currentStream.getTracks().forEach((track) => track.stop());
        window.currentStream = null;
    }

    fetch(url)
        .then((response) => response.text())
        .then((data) => {
            // Inject the HTML fragment into the main-content div
            document.getElementById("main-content").innerHTML = data;

            // After loading content, run initializers for that specific page
            if (url.includes("page-speech.html")) {
                initializeMainContent(); // Set up buttons and checkboxes
                initializeCamera();      // Start the camera
                loadBisaraNetModel();    // Load the ML model
            }
            // Add an 'else if' here for list.html if it ever needs JS
        })
        .catch((error) => console.error("Error loading content:", error));
}

// Function to set up sidebar event listeners
function setupSidebar() {
    const sidebar = document.getElementById("sidebar");
    const hamburger = document.getElementById("hamburger");
    const mainContent = document.getElementById("main-content");

    // Toggle sidebar visibility
    hamburger.addEventListener("click", (e) => {
        e.stopPropagation();
        if (sidebar.style.left === "0px") {
            closeSidebar();
        } else {
            openSidebar();
        }
    });

    // Close sidebar when clicking outside of it
    mainContent.addEventListener("click", () => {
        if (sidebar.style.left === "0px") {
            closeSidebar();
        }
    });

    // Navigate to Sign Language To Speech section (using the new file)
    document.getElementById("to-speech").addEventListener("click", (e) => {
        e.preventDefault();
        closeSidebar(() => loadContent("page-speech.html"));
    });

    // Navigate to Sign Language List section
    document.getElementById("to-list").addEventListener("click", (e) => {
        e.preventDefault();
        closeSidebar(() => loadContent("list.html"));
    });

    setDefaultVoice();
}

// --- END MODIFIED SECTION ---


function initializeMainContent() {
    const translateButton = document.getElementById('translate-button');
    const autoTranslateCheckbox = document.getElementById('auto-translate-checkbox');

    if (translateButton && autoTranslateCheckbox) {
        translateButton.disabled = isAutoTranslateEnabled;

        autoTranslateCheckbox.addEventListener('change', (event) => {
            isAutoTranslateEnabled = event.target.checked;
            translateButton.disabled = isAutoTranslateEnabled;
        });
    }

    if (translateButton) {
        translateButton.addEventListener('click', () => {
            const outputTextbox = document.getElementById('output-textbox');
            if (outputTextbox && outputTextbox.value) {
                speakText(outputTextbox.value);
                clearTimeout(periodTimer); // Also clear the period timer
            }
        });
    }
}

async function loadBisaraNetModel() {
    try {
        const loadingIndicator = document.getElementById('loading-indicator');
        if (loadingIndicator) {
            loadingIndicator.style.display = 'block';
        }
        // Using './' makes the path relative and more reliable for GitHub Pages
        bisaraNetModel = await tf.loadLayersModel('./model/model.json');
        console.log('BisaraNet.tfjs model loaded successfully!', bisaraNetModel);
        if (loadingIndicator) {
            loadingIndicator.style.display = 'none';
        }
    } catch (error) {
        console.error('Failed to load BisaraNet.tfjs model:', error);
    }
}

function initializeCamera() {
    const videoElement = document.getElementById("video");
    if (!videoElement) {
        console.error("Camera video element not found.");
        return;
    }

    navigator.mediaDevices
        .getUserMedia({
            video: true
        })
        .then((stream) => {
            videoElement.srcObject = stream;
            videoElement.style.transform = "scaleX(-1)";
            videoElement.play();
            window.currentStream = stream; // Save stream for cleanup

            videoElement.addEventListener('loadeddata', () => {
                console.log('Camera video loaded, initializing MediaPipe and starting prediction loop...');
                initializeMediaPipeHands();
                startPredictionLoop(videoElement);
            });
        })
        .catch((error) => console.error("Error accessing the camera:", error));
}

function initializeMediaPipeHands() {
    if (typeof Hands === 'undefined') {
        console.error("MediaPipe 'Hands' object is not defined.");
        return;
    }

    try {
        hands = new Hands({
            locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`
        });

        hands.setOptions({
            maxNumHands: 2,
            modelComplexity: 1,
            minDetectionConfidence: 0.5,
            minTrackingConfidence: 0.5
        });

        hands.onResults(onHandsResults);
        console.log('MediaPipe Hands initialized.');
    } catch (error) {
        console.error('Error initializing MediaPipe Hands:', error);
    }
}

async function onHandsResults(results) {
    let keypoints = new Array(NUM_LANDMARKS * 3 * 2).fill(0);

    if (results.multiHandLandmarks && results.multiHandLandmarks.length > 0) {
        results.multiHandLandmarks.slice(0, 2).forEach((handLandmarks, i) => {
            const flippedHandLandmarks = handLandmarks.map(landmark => ({ ...landmark, x: 1 - landmark.x }));
            const startIdx = i * (NUM_LANDMARKS * 3);
            const x_coords = flippedHandLandmarks.map(lm => lm.x);
            const y_coords = flippedHandLandmarks.map(lm => lm.y);
            const z_coords = flippedHandLandmarks.map(lm => lm.z);
            const handKeypoints = [...x_coords, ...y_coords, ...z_coords];
            keypoints.splice(startIdx, handKeypoints.length, ...handKeypoints);
        });
    }

    sequence.push(keypoints);
    if (sequence.length > SEQUENCE_LENGTH) {
        sequence.shift();
    }

    if (sequence.length === SEQUENCE_LENGTH && bisaraNetModel) {
        try {
            const inputTensor = tf.tensor(sequence, [SEQUENCE_LENGTH, NUM_LANDMARKS * 3 * 2], 'float32').expandDims(0);
            const prediction = bisaraNetModel.predict(inputTensor);
            const predictionArray = prediction.arraySync()[0];
            const predictedIndex = tf.argMax(predictionArray).dataSync()[0];
            const outputTextbox = document.getElementById('output-textbox');

            if (predictionArray[predictedIndex] > confidenceThreshold) {
                currentPrediction = actions[predictedIndex];

                if (outputTextbox && currentPrediction !== 'blank' && currentPrediction !== lastAppendedWord) {
                    clearTimeout(periodTimer);
                    let wordToAppend = currentPrediction;
                    if (outputTextbox.value.trim() === '' || outputTextbox.value.endsWith('. ')) {
                        wordToAppend = wordToAppend.charAt(0).toUpperCase() + wordToAppend.slice(1);
                    }
                    outputTextbox.value += wordToAppend + ' ';
                    lastAppendedWord = currentPrediction;

                    periodTimer = setTimeout(() => {
                        if (outputTextbox.value.trim().length > 0) {
                            if (outputTextbox.value.endsWith(', ')) {
                                outputTextbox.value = outputTextbox.value.slice(0, -2) + '. ';
                            } else if (!outputTextbox.value.endsWith('. ')) {
                                outputTextbox.value = outputTextbox.value.trim() + '. ';
                            }
                            if (isAutoTranslateEnabled) {
                                speakText(outputTextbox.value);
                            }
                            lastAppendedWord = '';
                        }
                    }, 3000);
                }
            } else {
                lastAppendedWord = '';
            }

            inputTensor.dispose();
            prediction.dispose();

        } catch (error) {
            console.error('Error during model prediction:', error);
        }
    }
}

async function startPredictionLoop(videoElement) {
    let isProcessing = false;

    async function detectHands() {
        if (!hands) {
            setTimeout(detectHands, 100);
            return;
        }
        if (videoElement.readyState === videoElement.HAVE_ENOUGH_DATA && !isProcessing) {
            isProcessing = true;
            await hands.send({ image: videoElement });
            isProcessing = false;
        }
        requestAnimationFrame(detectHands);
    }
    requestAnimationFrame(detectHands);
}

function openSidebar() {
    const sidebar = document.getElementById("sidebar");
    sidebar.style.left = "0px";
    const hamburger = document.getElementById("hamburger");
    hamburger.classList.add("open");
}

function closeSidebar(callback) {
    const sidebar = document.getElementById("sidebar");
    sidebar.style.left = "-300px";
    const hamburger = document.getElementById("hamburger");
    hamburger.classList.remove("open");
    setTimeout(() => {
        if (callback) callback();
    }, 300);
}

function setDefaultVoice() {
    if (typeof speechSynthesis === 'undefined') return;
    const setVoice = () => {
        const voices = speechSynthesis.getVoices();
        if (voices.length === 0) return;
        selectedVoice = voices.find(voice => voice.lang.startsWith('id')) || voices[0];
    };
    setVoice();
    if (speechSynthesis.onvoiceschanged !== undefined) {
        speechSynthesis.onvoiceschanged = setVoice;
    }
}

function speakText(text) {
    if (speechSynthesis.speaking) return;
    if (text !== "") {
        const utterThis = new SpeechSynthesisUtterance(text);
        utterThis.onend = function(event) {
            const outputTextbox = document.getElementById('output-textbox');
            if (outputTextbox) {
                outputTextbox.value = '';
                lastAppendedWord = '';
            }
        };
        utterThis.onerror = function(event) {
            console.error("SpeechSynthesisUtterance.onerror");
        };
        if (selectedVoice) {
            utterThis.voice = selectedVoice;
        }
        speechSynthesis.speak(utterThis);
    }
}

// --- FINAL INITIALIZATION ---
// This runs when the page first loads
document.addEventListener('DOMContentLoaded', () => {
    // Load the sidebar first
    fetch("sidebar.html")
        .then((response) => response.text())
        .then((data) => {
            document.getElementById("sidebar-container").innerHTML = data;
            // After the sidebar is loaded, set up its buttons
            setupSidebar();
            // Finally, load the default page content, which will start the camera
            loadContent('page-speech.html');
        })
        .catch((error) => console.error("Error loading sidebar:", error));
});
