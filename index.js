

import * as tf from '@tensorflow/tfjs';


const IMAGE_SIZE = 256;

const TOPK_PREDICTIONS = 10;

const MODEL_PATH = 'https://raw.githubusercontent.com/miranthajayatilake/pix2pix/master/model.json';


let pixmodel;

const pix2pix = async () => {

status('Please wait. Loading model. This might take sometime...But this happens only once.');


pixmodel = await tf.loadModel(MODEL_PATH);

// document.write('function done')
// status('Done loading pix');

pixmodel.predict(tf.zeros([1, IMAGE_SIZE, IMAGE_SIZE, 3])).dispose();

  status('');

  // Make a prediction through the locally hosted puppy.jpg.
  const sampleElement = document.getElementById('sample');
  if (sampleElement.complete && sampleElement.naturalHeight !== 0) {
    predict(sampleElement);
    sampleElement.style.display = '';
  } else {
    sampleElement.onload = () => {
      predict(sampleElement);
      sampleElement.style.display = '';
    }
  }  

// status('Image upload done');
document.getElementById('file-container').style.display = '';

}




async function predict(imgElement) {
  // status('Predicting...');

  const startTime = performance.now();
  const output_img = tf.tidy(() => {
    // tf.fromPixels() returns a Tensor from an image element.
    const img = tf.fromPixels(imgElement).toFloat();

    const offset = tf.scalar(127.5);
    // Normalize the image from [0, 255] to [-1, 1].
    const normalized = img.sub(offset).div(offset);

    // Reshape to a single-element batch so we can pass it to predict.
    const batched = normalized.reshape([1, IMAGE_SIZE, IMAGE_SIZE, 3]);

    // Make a prediction through mobilenet.
    return pixmodel.predict(batched);
  });

  console.log(output_img)

  showResults(imgElement, output_img);
}



//
// UI
//

function showResults(imgElement, output_img) {
  const predictionContainer = document.createElement('div');
  predictionContainer.className = 'pred-container';

  const imgContainer = document.createElement('div');
  imgContainer.appendChild(imgElement);
  predictionContainer.appendChild(imgContainer);


  output_img = tf.squeeze(output_img,[0])



  output_img = tf.div(tf.sub(output_img, tf.min(output_img)),tf.sub(tf.max(output_img),tf.min(output_img)));

  // let canvas = document.getElementById('myCanvas');
  var canvas = document.createElement("CANVAS");

  const output = tf.toPixels(output_img, canvas);
  // canvas.appendChild(output);
  predictionContainer.appendChild(canvas);  

  // status('Results successfully generated');


  predictionsElement.insertBefore(
      predictionContainer, predictionsElement.firstChild);
}




const filesElement = document.getElementById('files');
filesElement.addEventListener('change', evt => {
  let files = evt.target.files;
  // Display thumbnails & issue call to predict each image.
  for (let i = 0, f; f = files[i]; i++) {
    // Only process image files (skip non image files)
    if (!f.type.match('image.*')) {
      continue;
    }
    let reader = new FileReader();
    const idx = i;
    // Closure to capture the file information.
    reader.onload = e => {
      // Fill the image & call predict.
      let img = document.createElement('img');
      img.src = e.target.result;
      img.width = IMAGE_SIZE;
      img.height = IMAGE_SIZE;
      img.onload = () => predict(img);
    };

    // Read in the image file as a data URL.
    reader.readAsDataURL(f);
  }
});

const demoStatusElement = document.getElementById('status');
const status = msg => demoStatusElement.innerText = msg;

const predictionsElement = document.getElementById('predictions');

pix2pix();