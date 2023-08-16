import * as faceapi from 'face-api.js';

const video = document.getElementById('webcam') as HTMLVideoElement;

let capturedImages: HTMLImageElement[] = [];

async function setupWebcam() {
  // Access the webcam and request permission to use it
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ video: {} });
    video.srcObject = stream;

    // Return a promise that resolves once the video element can play the video stream
    return new Promise<HTMLVideoElement>((resolve) => {
      video.onloadedmetadata = () => {
        video.play();
        resolve(video);
      };
    });
  } catch (err) {
    console.error("Error accessing the webcam:", err);
    throw err;
    }
  }
  
  async function loadModels() {
    try {
      // Load the SsdMobilenetv1 model for face detection
      await faceapi.nets.ssdMobilenetv1.loadFromUri('/models');
  
      // Load the face landmark model
      await faceapi.nets.faceLandmark68Net.loadFromUri('/models');
  
      // Load the face expression recognition model
      await faceapi.nets.faceExpressionNet.loadFromUri('/models');
    } catch (err) {
      console.error("Error loading models:", err);
      throw err;
    }
  }
  
  async function detectFaceDirection() {
    const threshold = 40; // Adjust this value based on your needs
    let leftLookDetected = false;
    let rightLookDetected = false;
    
    async function checkDirection() {
      const detection = await faceapi.detectSingleFace(video).withFaceLandmarks();
      if (!detection) {
        console.error("No face detected");
        return;
      }
  
      const landmarks = detection.landmarks;
      const leftEye = landmarks.getLeftEye();
      const rightEye = landmarks.getRightEye();
      const leftEar = landmarks.getJawOutline()[0];
      const rightEar = landmarks.getJawOutline()[16];
  
      const leftEyeToEarDistance = Math.hypot(leftEye[0].x - leftEar.x, leftEye[0].y - leftEar.y);
      const rightEyeToEarDistance = Math.hypot(rightEye[0].x - rightEar.x, rightEye[0].y - rightEar.y);
  
      if (leftEyeToEarDistance < threshold && !leftLookDetected) {
        console.log("Looking to the left");
        leftLookDetected = true;
      } else if (rightEyeToEarDistance < threshold && !rightLookDetected) {
        console.log("Looking to the right");
        rightLookDetected = true;
      }
  
      // Check if both left and right looks have been detected
      if (leftLookDetected && rightLookDetected) {
        console.log("Both directions detected");
        return;
      }
  
      // Continue checking direction if not both directions have been detected
      setTimeout(checkDirection, 100);
    }
  
    try {
      // Prompt the user to look left
      console.log("Please look to the left");
      await checkDirection();
  
      // Prompt the user to look right
      console.log("Please look to the right");
      await checkDirection();
  
      console.log("Face direction detection completed");
    } catch (err) {
      console.error("Error detecting face direction:", err);
      throw err;
    }
  }
  
  
  async function detectExpressions() {
    const expressionPrompt = async (expression: string) => {
      console.log(`Please show a ${expression} expression.`);
      while (true) {
        const detection = await faceapi.detectSingleFace(video).withFaceExpressions();
        if (!detection) {
          console.error("No face detected");
          continue;
        }
  
        const expressions = detection.expressions;
        const maxExpression = Object.keys(expressions).reduce((a, b) => (expressions[a] > expressions[b] ? a : b));
  
        if (maxExpression === expression) {
          console.log(`${expression} expression detected.`);
          await captureImages(expression);
          break;
        }
      }
    };
  
    await expressionPrompt('neutral');
    await expressionPrompt('happy');
  }
  
  async function captureImages(expression: string) {
    const video = document.getElementById('webcam') as HTMLVideoElement;
    const canvas = document.getElementById('captureCanvas') as HTMLCanvasElement;
    const context = canvas.getContext('2d');
  
    for (let i = 0; i < 3; i++) {
      context.drawImage(video, 0, 0, canvas.width, canvas.height);
      const image = new Image();
      image.src = canvas.toDataURL('image/jpeg');
      image.id = `${expression}_${i}`;
      capturedImages.push(image);
      await new Promise(resolve => setTimeout(resolve, 1000));
    }
  }
  
  async function selectBestImages() {
    const imageSelection = document.createElement('div');
    imageSelection.id = 'imageSelection';
    document.body.appendChild(imageSelection);
  
    capturedImages.forEach(image => {
      imageSelection.appendChild(image);
      image.onclick = (event: Event) => {
        const target = event.target as HTMLImageElement;
        if (target.classList.contains('selected')) {
          target.classList.remove('selected');
        } else {
          const selectedImages = document.getElementsByClassName('selected');
          if (selectedImages.length < 2) {
            target.classList.add('selected');
          }
        }
      };
    });
  
    const confirmButton = document.createElement('button');
    confirmButton.textContent = 'Confirm Selection';
    confirmButton.onclick = async () => {
      const selectedImages = document.getElementsByClassName('selected');
      if (selectedImages.length === 2) {
        // Do something with the selected images
        console.log('Images selected:', selectedImages);
        await computeEmbeddings(); // Call computeEmbeddings after the user has selected the images
      } else {
        alert('Please select 2 images.');
      }
    };
    imageSelection.appendChild(confirmButton);
  }

async function computeEmbeddings() {
  const selectedImages = document.getElementsByClassName('selected');
  if (selectedImages.length !== 2) {
    console.error('Please select 2 images before computing embeddings.');
    return;
  }

  const embeddings: Array<Float32Array> = [];

  for (let i = 0; i < selectedImages.length; i++) {
    const image = selectedImages[i] as HTMLImageElement;
    const detection = await faceapi.detectSingleFace(image).withFaceLandmarks().withFaceDescriptor();

    if (!detection) {
      console.error('No face detected in the selected image.');
      return;
    }

    embeddings.push(detection.descriptor);
  }

  console.log(JSON.stringify({ embeddings }))
}

  
async function main() {
  try {
    await setupWebcam();
    await loadModels();
    await detectFaceDirection();
    await detectExpressions();
    await selectBestImages();
  } catch (err) {
    console.error(err);
  }
}

main().catch((err) => {
  console.error(err);
});