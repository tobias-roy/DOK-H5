import {Component, OnInit} from '@angular/core';
import * as tf from '@tensorflow/tfjs';
import {loadGraphModel} from '@tensorflow/tfjs-converter';
import { async } from 'rxjs';
import {resolve} from '@angular/compiler-cli';
tf.setBackend('webgl');
const threshold = 0.80;

@Component({
  selector: 'app-tfjsdetection',
  imports: [],
  templateUrl: './tfjsdetection.component.html',
  standalone: true,
  styleUrl: './tfjsdetection.component.css'
})
export class TfjsdetectionComponent implements OnInit {
  videoStream: any;
  videoRef: any;
  canvasRef: any;
  ngOnInit(): void {
    console.log('ngOnInit called');
  }

  async ngAfterViewInit() {
    console.log('ngAfterViewInit called');
    this.videoRef = document.getElementById('videoFeed');
    this.canvasRef = document.getElementById('canvasOverlay');
    console.log('videoRef:', this.videoRef);
    console.log('canvasRef:', this.canvasRef);
    await this.startCamera();
  }

  // Define the classes for detection
  classesDir = {
    1: { name: 'Aircraft Carrier', id: 1 },
    2: { name: 'Bulkers', id: 2 },
    3: { name: 'Car Carrier', id: 3 },
    4: { name: 'Container Ship', id: 4 },
    5: { name: 'Cruise', id: 5 },
    6: { name: 'DDG', id: 6 },
    7: { name: 'Recreational', id: 7 },
    8: { name: 'Sailboat', id: 8 },
    9: { name: 'Submarine', id: 9 },
    10: { name: 'Tug', id: 10 }
  }

  // In the function below using promises to do a Promise.all is the best approach
  async startCamera() {
    console.log('startCamera called');
    if(navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
      const webcamPromise = navigator.mediaDevices
        .getUserMedia({
          audio: false,
          video: {
            width: 640,
            height: 480
          }
        })
        .then(stream => {
          console.log('getUserMedia stream:', stream);
          this.videoStream = stream;
          this.videoRef.srcObject = this.videoStream;
          return new Promise<void>((resolve, reject) => {
            this.videoRef.onloadedmetadata = () => {
              console.log('videoRef onloadedmetadata');
              this.videoRef.play();
              resolve();
            };
          });
        });

      await tf.setBackend('webgl');
      console.log('TensorFlow backend set to webgl');
      await tf.ready();
      console.log('TensorFlow ready');

      const modelPromise = loadModel();

      Promise.all([modelPromise, webcamPromise])
        .then(values => {
          console.log('Promises resolved:', values);
          this.detectFrame(this.videoRef, values[0]);
        }).catch(error => {
        console.error('Error in Promise.all:', error);
      })
    }
  }

  detectFrame = (video: any, model: any) => {
    console.log('detectFrame called');
    tf.engine().startScope();
    model.executeAsync(process_input(video)).then((predictions: any) => {
      console.log('Predictions:', predictions);
      this.renderPredictions(predictions);
      requestAnimationFrame(() => {
        this.detectFrame(video, model);
      });
      tf.engine().endScope();
    }).catch(function (error: any) {
      console.error('Error in detectFrame:', error);
    });
  }

  renderPredictions = (predictions: any) => {
    console.log('renderPredictions called');
    const ctx = this.canvasRef.getContext("2d");
    ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);

    // Font options.
    const font = "16px sans-serif";
    ctx.font = font;
    ctx.textBaseline = "top";

    const boxes = predictions[4].arraySync();
    const scores = predictions[5].arraySync();
    const classes = predictions[6].dataSync();
    const detections = buildDetectedObjects(scores, threshold, boxes, classes, this.classesDir);

    detections.forEach(item => {
      console.log('Detection item:', item);
      const x = item['bbox'][0];
      const y = item['bbox'][1];
      const width = item['bbox'][2];
      const height = item['bbox'][3];

      // Draw the bounding box.
      ctx.strokeStyle = "#00FFFF";
      ctx.lineWidth = 4;
      ctx.strokeRect(x, y, width, height);

      // Draw the label background.
      ctx.fillStyle = "#00FFFF";
      const textWidth = ctx.measureText(item["label"] + " " + (100 * item["score"]).toFixed(2) + "%").width;
      const textHeight = parseInt(font, 10); // base 10
      ctx.fillRect(x, y, textWidth + 4, textHeight + 4);
    });
  };
}

function process_input(video_frame: any){
  console.log('process_input called');
  const tfImg = tf.browser.fromPixels(video_frame).toInt();
  return tfImg.transpose([0,1,2]).expandDims();
}

function buildDetectedObjects(scores: any, threshold: any, boxes: any, classes: any, classesDir: any) {
  console.log('buildDetectedObjects called');
  const detectionObjects: any[] = [];
  const video_frame = document.getElementById('videoFeed');
  if (scores != null && Array.isArray(scores[0])) {
    scores[0].forEach((score: any, i: number) => {
      if (score > threshold && video_frame != null) {
        console.log('Score above threshold:', score);
        const classId = classes[i];
        console.log('Class ID:', classId);
        if (classesDir[classId]) {
          const bbox = [];
          const minY = boxes[0][i][0] * video_frame.offsetHeight;
          const minX = boxes[0][i][1] * video_frame.offsetWidth;
          const maxY = boxes[0][i][2] * video_frame.offsetHeight;
          const maxX = boxes[0][i][3] * video_frame.offsetWidth;
          bbox[0] = minX;
          bbox[1] = minY;
          bbox[2] = maxX - minX;
          bbox[3] = maxY - minY;
          detectionObjects.push({
            class: classId,
            label: classesDir[classId].name,
            score: score.toFixed(4),
            bbox: bbox
          });
        }
      }
    });
  }
  console.log('Detection objects:', detectionObjects);
  return detectionObjects;
}

async function loadModel() {
  console.log('loadModel called');
  let model: any;
  try {
    model = await loadGraphModel('https://raw.githubusercontent.com/tobias-roy/DOK-H5/refs/heads/MachineLearning/tf2/models/research/object_detection/inference_graph/mobilenet_v2_lite/tfjsconvert/model.json', {onProgress: (number) => console.log('Model loading progress:', number)});
    console.log('Model loaded successfully');
  } catch (error) {
    console.log('Error loading model:', error);
  }
  return model;
}
