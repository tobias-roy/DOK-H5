import { Component, OnInit, ViewChild, ElementRef, AfterViewInit } from '@angular/core';
import * as tf from '@tensorflow/tfjs';
import { loadGraphModel } from '@tensorflow/tfjs-converter';

interface ClassInfo {
  name: string;
  id: number;
}

@Component({
  selector: 'app-tfjsdetectionv2',
  template: `
    <div>
      <h1>Real-Time Object Detection: Ships</h1>
      <h3>Mobilenet V2</h3>
      <video
        #videoFrame
        style="height: 360px; width: 480px; position: absolute;"
        class="size"
        autoplay
        playsinline
        muted
        id="frame"
        width="480"
        height="360"
      ></video>
      <canvas
        #canvasOverlay
        class="size"
        width="480"
        height="360"
        style="position: absolute;"
      ></canvas>
    </div>
  `,
  standalone: true,
  styles: []
})
export class TfjsdetectionComponentV2 implements OnInit, AfterViewInit {
  @ViewChild('videoFrame') videoRef!: ElementRef;
  @ViewChild('canvasOverlay') canvasRef!: ElementRef;

  videoStream: any;

  // Define the classes for detection
  classesDir: { [key: number]: ClassInfo } = {
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

  private threshold = 0.99;

  ngOnInit(): void {
    tf.setBackend('webgl');
  }

  ngAfterViewInit(): void {
    this.startCamera();
  }

  async startCamera(): Promise<void> {
    if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({
          audio: false,
          video: {
            width: 480,
            height: 360
          }
        });

        this.videoRef.nativeElement.srcObject = stream;
        this.videoRef.nativeElement.onloadedmetadata = async () => {
          this.videoRef.nativeElement.play();

          try {
            const model = await this.loadModel();
            this.detectFrame(this.videoRef.nativeElement, model);
          } catch (error) {
            console.error('Error loading model:', error);
          }
        };
      } catch (error) {
        console.error('Error accessing webcam:', error);
      }
    }
  }

  async loadModel(): Promise<any> {
    try {
      const model = await loadGraphModel('https://raw.githubusercontent.com/tobias-roy/DOK-H5/refs/heads/MachineLearning/tf2/models/research/object_detection/inference_graph/resnet50/tfjsconvert/model.json');
      return model;
    } catch (error) {
      console.error('Model loading failed:', error);
      throw error;
    }
  }

  detectFrame = (video: any, model: any): void => {
    tf.engine().startScope();
    model.executeAsync(this.processInput(video)).then((predictions: any) => {
      this.renderPredictions(predictions);
      requestAnimationFrame(() => {
        this.detectFrame(video, model);
      });
      tf.engine().endScope();
    });
  }

  processInput(video_frame: any): tf.Tensor {
    const tfImg = tf.browser.fromPixels(video_frame).toInt();
    return tfImg.transpose([0,1,2]).expandDims();
  }

  renderPredictions = (predictions: any[]): void => {
    const ctx = this.canvasRef.nativeElement.getContext("2d");
    ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);

    // Font options
    const font = "16px sans-serif";
    ctx.font = font;
    ctx.textBaseline = "top";

    const detections = this.buildDetectedObjects(predictions, this.threshold);
    console.log('Detections:', detections);

    detections.forEach((item: any) => {
      const [x, y, width, height] = item.bbox;

      // Draw the bounding box
      ctx.strokeStyle = "#00FFFF";
      ctx.lineWidth = 4;
      ctx.strokeRect(x, y, width, height);

      // Draw the label background
      ctx.fillStyle = "#00FFFF";
      const labelText = `${item.label} ${(item.score * 100).toFixed(2)}%`;
      const textWidth = ctx.measureText(labelText).width;
      const textHeight = parseInt(font, 10);
      ctx.fillRect(x, y, textWidth + 4, textHeight + 4);

      // Draw the text
      ctx.fillStyle = "#000000";
      ctx.fillText(labelText, x, y);
    });
  }

  buildDetectedObjects(predictions: any[], threshold: number): any[] {
    const detections: any[] = [];
    const videoFrame = this.videoRef.nativeElement;
    const scaleFactor = 0.95; // Adjust this factor to scale down the bounding boxes

    // Find the tensors with specific shapes
    const classesTensor = predictions.find(p => p.shape.length === 3 && p.shape[2] === 11);
    const scoresTensor = predictions.find(p => p.shape.length === 2 && p.shape[1] === 300);
    const boxesTensor = predictions.find(p => p.shape.length === 3 && p.shape[2] === 4);

    if (!classesTensor || !scoresTensor || !boxesTensor) {
      console.error('Could not find required tensors');
      return detections;
    }

    const classes = classesTensor.arraySync()[0];
    const scores = scoresTensor.arraySync()[0];
    const boxes = boxesTensor.arraySync()[0];

    for (let i = 0; i < scores.length; i++) {
      if (scores[i] > threshold) {
        // Find the class with the highest probability
        const classProbs = classes[i];
        const classId = classProbs.indexOf(Math.max(...classProbs));

        const [minY, minX, maxY, maxX] = boxes[i];

        // Calculate the bounding box with scaling
        const bboxWidth = (maxX - minX) * videoFrame.videoWidth * scaleFactor;
        const bboxHeight = (maxY - minY) * videoFrame.videoHeight * scaleFactor;
        const bboxX = minX * videoFrame.videoWidth + (1 - scaleFactor) * bboxWidth / 2;
        const bboxY = minY * videoFrame.videoHeight + (1 - scaleFactor) * bboxHeight / 2;

        const bbox = [bboxX, bboxY, bboxWidth, bboxHeight];

        detections.push({
          class: classId,
          label: this.classesDir[classId]?.name || 'Unknown',
          score: scores[i],
          bbox: bbox
        });
      }
    }

    return detections;
  }
}
