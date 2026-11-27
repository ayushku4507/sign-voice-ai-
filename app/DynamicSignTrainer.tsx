"use client";

import { useEffect, useRef, useState, type RefObject } from "react";

type Point = { x: number; y: number; z: number };
type Example = { label: string; frames: number[][] };
type Detector = {
  detectForVideo: (video: HTMLVideoElement, timestamp: number) => { landmarks?: Point[][] };
  close: () => void;
};
type TfModel = {
  predict: (input: unknown) => { data: () => Promise<Float32Array>; dispose: () => void };
  fit: (input: unknown, labels: unknown, options: Record<string, unknown>) => Promise<unknown>;
  save: (url: string) => Promise<unknown>;
  dispose: () => void;
};

const KEY = "signvoice-dynamic-asl-examples-v1";
const MODEL_KEY = "signvoice-dynamic-asl-model-v1";
const LABEL_KEY = "signvoice-dynamic-asl-labels-v1";
const FRAME_COUNT = 20;
const FEATURES = 63;
const TASK_MODEL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task";
const WASM_URL = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@1.2.3/wasm";

function normalize(points: Point[]) {
  if (points.length !== 21) return null;
  const wrist = points[0];
  const scale = Math.max(0.001, Math.hypot(points[9].x - wrist.x, points[9].y - wrist.y, points[9].z - wrist.z));
  const result: number[] = [];
  for (const point of points) {
    result.push((point.x - wrist.x) / scale, (point.y - wrist.y) / scale, (point.z - wrist.z) / scale);
  }
  return result;
}

function resample(frames: number[][], count = FRAME_COUNT) {
  if (frames.length === count) return frames;
  return Array.from({ length: count }, (_, index) => {
    const position = (index * (frames.length - 1)) / (count - 1);
    const low = Math.floor(position);
    const high = Math.min(frames.length - 1, low + 1);
    const part = position - low;
    return frames[low].map((value, feature) => value * (1 - part) + frames[high][feature] * part);
  });
}

export default function DynamicSignTrainer({
  cameraOn,
  videoRef,
}: {
  cameraOn: boolean;
  videoRef: RefObject<HTMLVideoElement | null>;
}) {
  const [label, setLabel] = useState("");
  const [examples, setExamples] = useState<Example[]>([]);
  const [status, setStatus] = useState("Turn on the camera to record motion examples.");
  const [trained, setTrained] = useState(false);
  const [recording, setRecording] = useState(false);
  const [training, setTraining] = useState(false);
  const [transcript, setTranscript] = useState("");
  const [prediction, setPrediction] = useState("");
  const [progress, setProgress] = useState(0);
  const detectorRef = useRef<Detector | null>(null);
  const rafRef = useRef<number | null>(null);
  const sequenceRef = useRef<number[][]>([]);
  const captureRef = useRef<number[][] | null>(null);
  const lastSampleRef = useRef(0);
  const lastVideoRef = useRef(-1);
  const lastInferenceRef = useRef(0);
  const inferenceBusyRef = useRef(false);
  const labelRef = useRef(label);
  labelRef.current = label;
  const modelRef = useRef<TfModel | null>(null);
  const labelsRef = useRef<string[]>([]);
  const examplesRef = useRef<Example[]>([]);
  const lastPredictionRef = useRef<{ label: string; since: number; emitted: boolean } | null>(null);
  const trainingRef = useRef(false);

  useEffect(() => {
    examplesRef.current = examples;
  }, [examples]);

  useEffect(() => {
    try {
      const raw = localStorage.getItem(KEY);
      if (raw) setExamples(JSON.parse(raw) as Example[]);
      const labels = localStorage.getItem(LABEL_KEY);
      if (labels) labelsRef.current = JSON.parse(labels) as string[];
      void import("@tensorflow/tfjs").then(async (tf) => {
        await tf.ready();
        try {
          modelRef.current = (await tf.loadLayersModel("indexeddb://" + MODEL_KEY)) as unknown as TfModel;
          setTrained(true);
          setStatus("Your saved motion model is ready on this device.");
        } catch {
          setStatus("Record examples for at least two signs, then train your model.");
        }
      }).catch(() => setStatus("TensorFlow.js could not start. Refresh and try again."));
    } catch {
      setStatus("Saved training examples could not be loaded.");
    }
  }, []);

  useEffect(() => {
    try {
      localStorage.setItem(KEY, JSON.stringify(examples));
    } catch {
      setStatus("Browser storage is full. Clear some examples and record again.");
    }
  }, [examples]);

  useEffect(() => {
    let cancelled = false;
    let detector: Detector | null = null;
    const stop = () => {
      if (rafRef.current !== null) cancelAnimationFrame(rafRef.current);
      rafRef.current = null;
      detector?.close();
      detector = null;
      detectorRef.current = null;
      captureRef.current = null;
      sequenceRef.current = [];
      lastPredictionRef.current = null;
      setRecording(false);
      setPrediction("");
    };
    if (!cameraOn) {
      stop();
      return () => { cancelled = true; stop(); };
    }
    async function start() {
      try {
        const { FilesetResolver, HandLandmarker } = await import("@mediapipe/tasks-vision");
        const vision = await FilesetResolver.forVisionTasks(WASM_URL);
        const instance = await HandLandmarker.createFromOptions(vision, {
          baseOptions: { modelAssetPath: TASK_MODEL },
          runningMode: "VIDEO",
          numHands: 1,
          minHandDetectionConfidence: 0.55,
          minHandPresenceConfidence: 0.55,
          minTrackingConfidence: 0.5,
        });
        if (cancelled) { instance.close(); return; }
        detector = instance as unknown as Detector;
        detectorRef.current = detector;
        setStatus("Hand tracking ready. Record the same sign several times.");
        const loop = () => {
          if (cancelled || !detector) return;
          const video = videoRef.current;
          if (video && video.readyState >= 2 && video.currentTime !== lastVideoRef.current) {
            lastVideoRef.current = video.currentTime;
            const result = detector.detectForVideo(video, performance.now());
            const points = result.landmarks?.[0];
            const frame = points ? normalize(points) : null;
            const now = performance.now();
            if (frame) {
              if (captureRef.current && now - lastSampleRef.current >= 65) {
                captureRef.current.push(frame);
                lastSampleRef.current = now;
                setProgress(Math.min(100, Math.round((captureRef.current.length / FRAME_COUNT) * 100)));
                if (captureRef.current.length >= FRAME_COUNT) {
                  const name = labelRef.current.trim().toLocaleLowerCase();
                  setExamples((current) => [...current, { label: name, frames: resample(captureRef.current ?? []) }]);
                  captureRef.current = null;
                  setRecording(false);
                  setProgress(100);
                  setStatus("Saved one motion example for “" + name + "”. Record a few more.");
                }
              }
              sequenceRef.current.push(frame);
              if (sequenceRef.current.length > FRAME_COUNT) sequenceRef.current.shift();
              if (modelRef.current && sequenceRef.current.length === FRAME_COUNT && !trainingRef.current && now - lastInferenceRef.current > 850 && !inferenceBusyRef.current) {
                lastInferenceRef.current = now;
                inferenceBusyRef.current = true;
                const currentModel = modelRef.current;
                const labels = labelsRef.current;
                void import("@tensorflow/tfjs").then((tf) => {
                  const input = tf.tensor2d([sequenceRef.current.flat()], [1, FRAME_COUNT * FEATURES]);
                  const output = currentModel.predict(input) as { data: () => Promise<Float32Array>; dispose: () => void };
                  void output.data().then((scores) => {
                    input.dispose(); output.dispose();
                    let best = 0;
                    for (let i = 1; i < scores.length; i += 1) if (scores[i] > scores[best]) best = i;
                    const guessed = scores[best] >= 0.72 ? labels[best] : "";
                    setPrediction(guessed);
                    const prior = lastPredictionRef.current;
                    if (!guessed) lastPredictionRef.current = null;
                    else if (prior?.label !== guessed) lastPredictionRef.current = { label: guessed, since: now, emitted: false };
                    else if (!prior.emitted && now - prior.since > 1200) {
                      lastPredictionRef.current = { ...prior, emitted: true };
                      setTranscript((current) => current ? current + " " + guessed : guessed);
                    }
                  }).catch(() => { inferenceBusyRef.current = false; input.dispose(); output.dispose(); });
                });
              }
            } else {
              sequenceRef.current = [];
              setPrediction("");
              lastPredictionRef.current = null;
            }
          }
          rafRef.current = requestAnimationFrame(loop);
        };
        loop();
      } catch {
        setStatus("Could not load hand tracking. Check your connection, then turn the camera off and on.");
      }
    }
    void start();
    return () => { cancelled = true; stop(); };
  }, [cameraOn, videoRef]);

  const counts = examples.reduce<Record<string, number>>((counts, example) => {
    counts[example.label] = (counts[example.label] ?? 0) + 1;
    return counts;
  }, {});
  const labels = Object.keys(counts).sort();

  function record() {
    if (!cameraOn || !detectorRef.current) { setStatus("Turn on the camera first."); return; }
    if (!label.trim()) { setStatus("Enter the word or phrase you are signing."); return; }
    captureRef.current = [];
    lastSampleRef.current = performance.now();
    setRecording(true);
    setProgress(0);
    setStatus("Perform the sign once, clearly, while the camera records the movement.");
  }

  async function train() {
    const usable = labels.filter((name) => (counts[name] ?? 0) >= 3);
    if (usable.length < 2) {
      setStatus("Record at least 3 examples each for two or more different signs before training.");
      return;
    }
    trainingRef.current = true;
    setTraining(true);
    setStatus("Training your local TensorFlow model…");
    try {
      const tf = await import("@tensorflow/tfjs");
      await tf.ready();
      modelRef.current?.dispose();
      const selected = examples.filter((example) => usable.includes(example.label));
      const xs = tf.tensor2d(selected.map((item) => item.frames.flat()), [selected.length, FRAME_COUNT * FEATURES]);
      const ys = tf.tensor2d(selected.map((item) => usable.map((name) => name === item.label ? 1 : 0)), [selected.length, usable.length]);
      const model = tf.sequential();
      model.add(tf.layers.dense({ inputShape: [FRAME_COUNT * FEATURES], units: 128, activation: "relu" }));
      model.add(tf.layers.dropout({ rate: 0.25 }));
      model.add(tf.layers.dense({ units: 64, activation: "relu" }));
      model.add(tf.layers.dense({ units: usable.length, activation: "softmax" }));
      model.compile({ optimizer: tf.train.adam(0.001), loss: "categoricalCrossentropy", metrics: ["accuracy"] });
      await model.fit(xs, ys, { epochs: 45, batchSize: Math.min(8, selected.length), shuffle: true, verbose: 0 });
      xs.dispose(); ys.dispose();
      await model.save("indexeddb://" + MODEL_KEY);
      modelRef.current = model as unknown as TfModel;
      labelsRef.current = usable;
      localStorage.setItem(LABEL_KEY, JSON.stringify(usable));
      setTrained(true);
      setStatus("Trained and saved on this device for: " + usable.join(", ") + ".");
    } catch {
      setStatus("Training failed. Clear a few examples or reload the page and try again.");
    } finally {
      trainingRef.current = false;
      setTraining(false);
    }
  }

  function clearData() {
    modelRef.current?.dispose();
    modelRef.current = null;
    labelsRef.current = [];
    localStorage.removeItem(KEY);
    localStorage.removeItem(LABEL_KEY);
    void import("@tensorflow/tfjs").then((tf) => tf.io.removeModel("indexeddb://" + MODEL_KEY)).catch(() => {});
    setExamples([]);
    setTrained(false);
    setTranscript("");
    setPrediction("");
    setStatus("Saved sign examples and model cleared from this browser.");
  }

  return (
    <details className="asl-trainer" style={{ marginTop: 14 }}>
      <summary>Train a TensorFlow motion-sign model ({examples.length} examples)</summary>
      <p className="small-note" style={{ marginTop: 12 }}>
        Add one label for a sign, record it at least 3 times, then repeat for another sign and press Train.
        The camera is processed on this device; only hand landmark numbers are saved in this browser.
        This personal model learns only the signs you record. It does not yet understand every ASL sign or continuous ASL sentences.
      </p>
      <label className="asl-transcript-label" htmlFor="dynamicSignLabel">Sign label (for example: HELP)</label>
      <input id="dynamicSignLabel" value={label} onChange={(event) => setLabel(event.target.value)} placeholder="Type a word or phrase" maxLength={40} />
      <div className="asl-capture-row" style={{ marginTop: 12 }}>
        <button className="secondary-btn" onClick={record} disabled={recording || !cameraOn}>{recording ? "Recording…" : "Record sign example"}</button>
        <button className="primary-btn" onClick={() => void train()} disabled={recording || training}>Train TensorFlow model</button>
        <span className="small-note">{progress}%</span>
      </div>
      {recording && <progress className="asl-progress" value={progress} max={100} aria-label="Motion capture progress" />}
      <p className="asl-model-status" role="status">{status}</p>
      <p className="small-note">Saved labels: {labels.length ? labels.map((name) => name + " (" + counts[name] + ")").join(" · ") : "none"}{trained ? " · trained model ready" : ""}</p>
      <label className="asl-transcript-label" htmlFor="dynamicSignTranscript">Recognized sign text</label>
      <textarea id="dynamicSignTranscript" className="asl-transcript" value={transcript} onChange={(event) => setTranscript(event.target.value)} placeholder={prediction ? "Recognizing: " + prediction : "Train a model to recognize recorded signs…"} />
      <div className="asl-transcript-actions">
        <button className="small-action" onClick={() => setTranscript((text) => text + " ")}>Space</button>
        <button className="small-action" onClick={() => setTranscript((text) => text.trimEnd().split(" ").slice(0, -1).join(" "))}>Undo word</button>
        <button className="small-action" onClick={() => setTranscript("")}>Clear text</button>
        <button className="small-action" onClick={clearData}>Clear training data</button>
      </div>
    </details>
  );
}
