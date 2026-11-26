"use client";

import { useEffect, useMemo, useRef, useState, type RefObject } from "react";

type Point = { x: number; y: number; z: number };
type Sample = { label: string; pose?: number[]; sequence?: number[][] };
type Detector = {
  detectForVideo: (video: HTMLVideoElement, timestamp: number) => {
    landmarks?: Point[][];
    handedness?: Array<Array<{ categoryName?: string }>>;
    handednesses?: Array<Array<{ categoryName?: string }>>;
  };
  close: () => void;
};
type Capture = { label: string; startedAt: number; lastSampleAt: number; frames: number[][] };
type MlpLayer = { w: number[][]; b: number[]; act: "relu" | "softmax" | "linear" };
type MlpModel = { labels: string[]; featureLength: number; layers: MlpLayer[] };
type Match = { label: string; match: number; source: "trained" | "sample" };

const LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ".split("");
const STORAGE_KEY = "signvoice-asl-alphabet-v1";
const DYNAMIC_LETTERS = new Set(["J", "Z"]);
const SEQUENCE_LENGTH = 12;
const STATIC_DISTANCE_LIMIT = 0.34;
const DYNAMIC_DISTANCE_LIMIT = 0.30;
const TASK_MODEL =
  "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task";
const WASM_URL = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@1.2.3/wasm";

function poseFeatures(points: Point[], handedness: string) {
  if (points.length !== 21) return null;
  const mirror = handedness.toLowerCase() === "left" ? -1 : 1;
  const wrist = points[0];
  const middle = points[9];
  const dx = (middle.x - wrist.x) * mirror;
  const dy = middle.y - wrist.y;
  const angle = -Math.PI / 2 - Math.atan2(dy, dx);
  const cos = Math.cos(angle);
  const sin = Math.sin(angle);
  const scale = Math.max(0.001, Math.hypot(dx, dy));
  const output: number[] = [];

  for (const point of points) {
    const x = (point.x - wrist.x) * mirror;
    const y = point.y - wrist.y;
    const z = point.z - wrist.z;
    output.push((x * cos - y * sin) / scale);
    output.push((x * sin + y * cos) / scale);
    output.push(z / scale);
  }
  return output;
}

function distance(a: number[], b: number[]) {
  if (a.length !== b.length) return Number.POSITIVE_INFINITY;
  let sum = 0;
  for (let i = 0; i < a.length; i += 1) {
    const delta = a[i] - b[i];
    sum += delta * delta;
  }
  return Math.sqrt(sum / a.length);
}

function resample(frames: number[][], count: number) {
  if (frames.length === count) return frames;
  return Array.from({ length: count }, (_, index) => {
    const sourceIndex = Math.round((index * (frames.length - 1)) / (count - 1));
    return frames[sourceIndex];
  });
}

function average(frames: number[][]) {
  if (frames.length === 1) return frames[0];
  return frames[0].map((_, index) =>
    frames.reduce((sum, frame) => sum + frame[index], 0) / frames.length,
  );
}

function modelFeatures(points: Point[]) {
  if (points.length !== 21) return null;
  const wrist = points[0];
  const middle = points[9];
  const scale = Math.max(
    0.000001,
    Math.hypot(middle.x - wrist.x, middle.y - wrist.y, middle.z - wrist.z),
  );
  const output = new Float32Array(63);
  points.forEach((point, index) => {
    output[index * 3] = (point.x - wrist.x) / scale;
    output[index * 3 + 1] = (point.y - wrist.y) / scale;
    output[index * 3 + 2] = (point.z - wrist.z) / scale;
  });
  return output;
}

function predictStatic(points: Point[], model: MlpModel) {
  const features = modelFeatures(points);
  if (!features || model.featureLength !== features.length) return null;
  let activations: number[] = Array.from(features);
  for (const layer of model.layers) {
    const output = layer.b.slice();
    for (let inputIndex = 0; inputIndex < activations.length; inputIndex += 1) {
      const value = activations[inputIndex];
      if (value === 0) continue;
      const row = layer.w[inputIndex];
      for (let outputIndex = 0; outputIndex < output.length; outputIndex += 1) {
        output[outputIndex] += value * row[outputIndex];
      }
    }
    if (layer.act === "relu") {
      activations = output.map((value) => Math.max(0, value));
    } else if (layer.act === "softmax") {
      const max = Math.max(...output);
      const exponentials = output.map((value) => Math.exp(value - max));
      const total = exponentials.reduce((sum, value) => sum + value, 0);
      activations = exponentials.map((value) => value / total);
    } else {
      activations = output;
    }
  }
  let bestIndex = 0;
  for (let index = 1; index < activations.length; index += 1) {
    if (activations[index] > activations[bestIndex]) bestIndex = index;
  }
  return { label: model.labels[bestIndex], confidence: activations[bestIndex] };
}

function findMatch(
  pose: number[] | null,
  points: Point[] | undefined,
  sequence: number[][],
  samples: Sample[],
  model: MlpModel | null,
): Match | null {
  if (!pose) return null;
  const staticSamples = samples.filter((sample) => sample.pose && !DYNAMIC_LETTERS.has(sample.label));
  const staticRanked = staticSamples
    .map((sample) => ({ label: sample.label, distance: distance(pose, sample.pose!) }))
    .sort((a, b) => a.distance - b.distance);
  const personalStaticBest = staticRanked[0] ?? null;

  let dynamicBest: { label: string; distance: number } | null = null;
  if (sequence.length >= SEQUENCE_LENGTH) {
    const current = resample(sequence.slice(-SEQUENCE_LENGTH), SEQUENCE_LENGTH);
    const dynamicSamples = samples.filter((sample) => sample.sequence && DYNAMIC_LETTERS.has(sample.label));
    const ranked = dynamicSamples
      .map((sample) => {
        const reference = resample(sample.sequence!, SEQUENCE_LENGTH);
        const averageDistance =
          reference.reduce((sum, frame, index) => sum + distance(frame, current[index]), 0) /
          SEQUENCE_LENGTH;
        return { label: sample.label, distance: averageDistance };
      })
      .sort((a, b) => a.distance - b.distance);
    dynamicBest = ranked[0] ?? null;
  }

  if (dynamicBest && dynamicBest.distance <= DYNAMIC_DISTANCE_LIMIT) {
    return {
      label: dynamicBest.label,
      match: Math.max(0, Math.min(100, Math.round((1 - dynamicBest.distance / DYNAMIC_DISTANCE_LIMIT) * 100))),
      source: "sample",
    };
  }

  const trainedMatch = points && model ? predictStatic(points, model) : null;
  if (trainedMatch && trainedMatch.confidence >= 0.58) {
    return { label: trainedMatch.label, match: Math.round(trainedMatch.confidence * 100), source: "trained" };
  }

  if (personalStaticBest && personalStaticBest.distance <= STATIC_DISTANCE_LIMIT) {
    return {
      label: personalStaticBest.label,
      match: Math.max(0, Math.min(100, Math.round((1 - personalStaticBest.distance / STATIC_DISTANCE_LIMIT) * 100))),
      source: "sample",
    };
  }
  return null;
}

export default function ASLTrainer({
  cameraOn,
  videoRef,
}: {
  cameraOn: boolean;
  videoRef: RefObject<HTMLVideoElement | null>;
}) {
  const [samples, setSamples] = useState<Sample[]>([]);
  const modelRef = useRef<MlpModel | null>(null);
  const [storageReady, setStorageReady] = useState(false);
  const [selectedLetter, setSelectedLetter] = useState("A");
  const [modelStatus, setModelStatus] = useState("Turn on the camera to load the trained ASL model and hand tracking.");
  const [captureStatus, setCaptureStatus] = useState("");
  const [captureProgress, setCaptureProgress] = useState(0);
  const [capturing, setCapturing] = useState(false);
  const [handDetected, setHandDetected] = useState(false);
  const [prediction, setPrediction] = useState<Match | null>(null);
  const [transcript, setTranscript] = useState("");
  const [autoAdd, setAutoAdd] = useState(true);
  const stableMatchRef = useRef<{ label: string; since: number; added: boolean } | null>(null);
  const autoAddRef = useRef(autoAdd);
  autoAddRef.current = autoAdd;
  const detectorRef = useRef<Detector | null>(null);
  const samplesRef = useRef<Sample[]>([]);
  const rafRef = useRef<number | null>(null);
  const captureRef = useRef<Capture | null>(null);
  const sequenceRef = useRef<number[][]>([]);
  const lastVideoTimeRef = useRef(-1);
  const frameNumberRef = useRef(0);

  useEffect(() => {
    samplesRef.current = samples;
  }, [samples]);

  useEffect(() => {
    try {
      const saved = window.localStorage.getItem(STORAGE_KEY);
      if (saved) setSamples(JSON.parse(saved) as Sample[]);
    } catch {
      setCaptureStatus("Saved training examples could not be loaded.");
    } finally {
      setStorageReady(true);
    }
  }, []);

  useEffect(() => {
    if (!storageReady) return;
    try {
      window.localStorage.setItem(STORAGE_KEY, JSON.stringify(samples));
    } catch {
      setCaptureStatus("Browser storage is full. Clear samples and train fewer examples.");
    }
  }, [samples, storageReady]);

  useEffect(() => {
    let cancelled = false;
    let detector: Detector | null = null;

    const stop = () => {
      if (rafRef.current !== null) cancelAnimationFrame(rafRef.current);
      rafRef.current = null;
      if (detector) detector.close();
      detector = null;
      detectorRef.current = null;
      captureRef.current = null;
      sequenceRef.current = [];
      lastVideoTimeRef.current = -1;
      setCapturing(false);
      setPrediction(null);
      setHandDetected(false);
    };

    if (!cameraOn) {
      stop();
      setModelStatus("Turn on the camera to load the trained ASL model and hand tracking.");
      return () => {
        cancelled = true;
        stop();
      };
    }

    async function start() {
      try {
        setModelStatus("Loading the trained ASL alphabet model and hand tracking…");
        const response = await fetch("/models/asl-alphabet-mlp.json");
        if (response.ok) {
          const model = (await response.json()) as MlpModel;
          if (model.featureLength === 63 && Array.isArray(model.labels) && Array.isArray(model.layers)) {
            modelRef.current = model;
          }
        }
        const { FilesetResolver, HandLandmarker } = await import("@mediapipe/tasks-vision");
        const vision = await FilesetResolver.forVisionTasks(WASM_URL);
        const created = await HandLandmarker.createFromOptions(vision, {
          baseOptions: { modelAssetPath: TASK_MODEL },
          runningMode: "VIDEO",
          numHands: 1,
          minHandDetectionConfidence: 0.55,
          minHandPresenceConfidence: 0.55,
          minTrackingConfidence: 0.5,
        });
        if (cancelled) {
          created.close();
          return;
        }
        detector = created as unknown as Detector;
        detectorRef.current = detector;
        setModelStatus(
          modelRef.current
            ? "Trained ASL model ready for 24 static letters. J and Z need motion examples."
            : "Hand tracking ready, but the trained model did not load. Capture examples for letters.",
        );
        const loop = () => {
          if (cancelled || !detector) return;
          const video = videoRef.current;
          if (video && video.readyState >= 2 && video.currentTime !== lastVideoTimeRef.current) {
            lastVideoTimeRef.current = video.currentTime;
            try {
              const result = detector.detectForVideo(video, performance.now());
              const points = result.landmarks?.[0];
              const handLabel =
                result.handednesses?.[0]?.[0]?.categoryName ??
                result.handedness?.[0]?.[0]?.categoryName ??
                "Right";
              const pose = points ? poseFeatures(points, handLabel) : null;
              setHandDetected(Boolean(pose));
              if (pose) {
                frameNumberRef.current += 1;
                const now = performance.now();
                const capture = captureRef.current;
                if (capture && now - capture.lastSampleAt >= 90) {
                  capture.frames.push(pose);
                  capture.lastSampleAt = now;
                  const dynamic = DYNAMIC_LETTERS.has(capture.label);
                  const targetFrames = dynamic ? SEQUENCE_LENGTH : 7;
                  const duration = dynamic ? 1350 : 700;
                  setCaptureProgress(Math.min(100, Math.round((capture.frames.length / targetFrames) * 100)));
                  if (capture.frames.length >= targetFrames && now - capture.startedAt >= duration) {
                    const sample: Sample = dynamic
                      ? { label: capture.label, sequence: resample(capture.frames, SEQUENCE_LENGTH) }
                      : { label: capture.label, pose: average(capture.frames.slice(-3)) };
                    setSamples((current) => [...current, sample]);
                    setCaptureStatus("Saved one " + capture.label + " example in this browser.");
                    setCapturing(false);
                    setCaptureProgress(100);
                    captureRef.current = null;
                  }
                }

                sequenceRef.current.push(pose);
                if (sequenceRef.current.length > SEQUENCE_LENGTH) sequenceRef.current.shift();
                if (frameNumberRef.current % 4 === 0) {
                  const match = findMatch(pose, points, sequenceRef.current, samplesRef.current, modelRef.current);
                  setPrediction(match);
                  const now = performance.now();
                  if (!match) {
                    stableMatchRef.current = null;
                  } else if (stableMatchRef.current?.label !== match.label) {
                    stableMatchRef.current = { label: match.label, since: now, added: false };
                  } else if (autoAddRef.current && !stableMatchRef.current.added && now - stableMatchRef.current.since >= 900) {
                    stableMatchRef.current.added = true;
                    setTranscript((current) => current + match.label);
                  }
                }
              } else {
                sequenceRef.current = [];
                if (frameNumberRef.current % 4 === 0) {
                  setPrediction(null);
                  stableMatchRef.current = null;
                }
              }
            } catch {
              setModelStatus("Hand tracking paused. Turn the camera off and on to retry.");
            }
          }
          rafRef.current = requestAnimationFrame(loop);
        };
        loop();
      } catch {
        setModelStatus("Could not load the ASL model or hand tracking. Check your internet connection and try again.");
      }
    }

    void start();
    return () => {
      cancelled = true;
      stop();
    };
  }, [cameraOn, videoRef]);

  const counts = useMemo(() => {
    const result: Record<string, number> = {};
    for (const letter of LETTERS) result[letter] = 0;
    for (const sample of samples) result[sample.label] = (result[sample.label] ?? 0) + 1;
    return result;
  }, [samples]);

  function beginCapture() {
    if (!cameraOn || !detectorRef.current) {
      setCaptureStatus("Turn on the camera first.");
      return;
    }
    if (!handDetected) {
      setCaptureStatus("Show one hand in the camera before capturing.");
      return;
    }
    captureRef.current = {
      label: selectedLetter,
      startedAt: performance.now(),
      lastSampleAt: 0,
      frames: [],
    };
    setCapturing(true);
    setCaptureProgress(0);
    setCaptureStatus(
      DYNAMIC_LETTERS.has(selectedLetter)
        ? "Draw the " + selectedLetter + " sign once, clearly, while the camera records."
        : "Hold the " + selectedLetter + " handshape steady while the camera records.",
    );
  }

  function clearTraining() {
    setSamples([]);
    setPrediction(null);
    setCaptureStatus("Training examples cleared from this browser.");
  }

  function addPrediction() {
    if (!prediction) return;
    setTranscript((current) => current + prediction.label);
  }

  return (
    <div className="asl-recognizer">
      <div className="asl-live">
        <div className="asl-readout">
          <span className="asl-readout-label">POSSIBLE LETTER</span>
          <strong>{prediction?.label ?? "—"}</strong>
          <span className="small-note">
            {prediction ? (prediction.source === "trained" ? "Trained model confidence: " : "Personal sample match: ") + prediction.match + "%" : handDetected ? "No confident letter match" : "Show one hand in the camera"}
          </span>
        </div>
        <div className="asl-live-controls">
          <span className={handDetected ? "asl-indicator on" : "asl-indicator"}>{handDetected ? "HAND FOUND" : "NO HAND"}</span>
          <button className={autoAdd ? "secondary-btn active" : "secondary-btn"} onClick={() => setAutoAdd((enabled) => !enabled)} aria-pressed={autoAdd}>{autoAdd ? "Auto text: on" : "Auto text: off"}</button>
          <button className="secondary-btn" onClick={addPrediction} disabled={!prediction}>Add letter</button>
        </div>
      </div>

      <label className="asl-transcript-label" htmlFor="aslTranscript">Recognized text</label>
      <p className="small-note">{autoAdd ? "A steady recognized letter is added automatically. Hold each letter briefly; use Space between words." : "Turn on Auto text to add steady recognized letters, or press Add letter."}</p>
      <textarea
        id="aslTranscript"
        className="asl-transcript"
        value={transcript}
        onChange={(event) => setTranscript(event.target.value)}
        placeholder="Add recognized letters here…"
      />
      <div className="asl-transcript-actions">
        <button className="small-action" onClick={() => setTranscript((text) => text + " ")}>Space</button>
        <button className="small-action" onClick={() => setTranscript((text) => text.slice(0, -1))}>Undo</button>
        <button className="small-action" onClick={() => setTranscript("")}>Clear text</button>
      </div>

      <details className="asl-trainer" open={samples.length === 0}>
        <summary>Train your ASL alphabet model ({samples.length} examples)</summary>
        {samples.length === 0 && <p className="asl-first-run"><strong>A trained model is included for 24 static ASL letters.</strong> Turn on the camera to recognize A–I and K–Y. J and Z use movement, so capture examples for those two letters.</p>}
        <p className="small-note">
          The built-in neural model recognizes A–I and K–Y from hand landmarks. Capture J and Z motion examples with your own hand; those samples stay in this browser. Camera video is not uploaded. This recognizes fingerspelled letters, not full ASL vocabulary signs. Model weights: <a href="https://github.com/JmBagares/SignLanguageDetector" target="_blank" rel="noreferrer">SignSense by Jan Manuel Bagares (MIT)</a>.
        </p>
        <p className="asl-model-status">{modelStatus}</p>
        <div className="asl-letter-grid" aria-label="Choose an ASL letter">
          {LETTERS.map((letter) => (
            <button
              key={letter}
              className={selectedLetter === letter ? "asl-letter selected" : "asl-letter"}
              onClick={() => setSelectedLetter(letter)}
              aria-pressed={selectedLetter === letter}
              title={DYNAMIC_LETTERS.has(letter) ? "Motion sign; capture the full movement" : "Static handshape"}
            >
              <strong>{letter}</strong>
              <span>{counts[letter] ?? 0}</span>
            </button>
          ))}
        </div>
        <div className="asl-capture-row">
          <button className="primary-btn" onClick={beginCapture} disabled={capturing || !cameraOn}>
            {capturing ? "Capturing…" : "Capture " + selectedLetter + " example"}
          </button>
          <span className="small-note">
            {DYNAMIC_LETTERS.has(selectedLetter) ? "J and Z use movement; capture several clear examples." : "The built-in model handles static letters; examples here are optional."}
          </span>
        </div>
        {capturing && <progress className="asl-progress" value={captureProgress} max={100} aria-label="Capture progress" />}
        {captureStatus && <p className="asl-capture-status" role="status">{captureStatus}</p>}
        <div className="asl-training-footer">
          <span className="small-note">{samples.length} examples saved in this browser</span>
          <button className="small-action" onClick={clearTraining} disabled={!samples.length}>Clear training examples</button>
        </div>
      </details>
    </div>
  );
}
