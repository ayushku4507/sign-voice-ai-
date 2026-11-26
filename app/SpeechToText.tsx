"use client";

import { useEffect, useRef, useState } from "react";
import { Mic, MicOff } from "lucide-react";

type SpeechWorkerMessage =
  | { type: "progress"; status: string; progress?: number }
  | { type: "ready" }
  | { type: "result"; text: string }
  | { type: "error"; message: string };

type AudioCapture = {
  stream: MediaStream;
  context: AudioContext;
  source: MediaStreamAudioSourceNode;
  processor: ScriptProcessorNode;
  silentGain: GainNode;
};

const TARGET_SAMPLE_RATE = 16000;
const CHUNK_SECONDS = 5;

function resample(chunks: Float32Array[], sourceRate: number) {
  const inputLength = chunks.reduce((total, chunk) => total + chunk.length, 0);
  const input = new Float32Array(inputLength);
  let offset = 0;
  for (const chunk of chunks) {
    input.set(chunk, offset);
    offset += chunk.length;
  }
  if (sourceRate === TARGET_SAMPLE_RATE) return input;

  const outputLength = Math.floor((input.length * TARGET_SAMPLE_RATE) / sourceRate);
  const output = new Float32Array(outputLength);
  const ratio = sourceRate / TARGET_SAMPLE_RATE;
  for (let index = 0; index < outputLength; index += 1) {
    const position = index * ratio;
    const left = Math.floor(position);
    const right = Math.min(left + 1, input.length - 1);
    const blend = position - left;
    output[index] = input[left] * (1 - blend) + input[right] * blend;
  }
  return output;
}

export default function SpeechToText({ language }: { language: string }) {
  const [heardText, setHeardText] = useState("");
  const [listening, setListening] = useState(false);
  const [status, setStatus] = useState("First use downloads the speech model; transcription then runs on this device.");
  const workerRef = useRef<Worker | null>(null);
  const captureRef = useRef<AudioCapture | null>(null);
  const listeningRef = useRef(false);
  const languageRef = useRef(language);
  const audioChunksRef = useRef<Float32Array[]>([]);
  const sampleCountRef = useRef(0);
  const pendingAudioRef = useRef<Float32Array[]>([]);
  const transcribingRef = useRef(false);

  languageRef.current = language;

  function releaseMicrophone() {
    const capture = captureRef.current;
    captureRef.current = null;
    if (!capture) return;
    capture.processor.onaudioprocess = null;
    capture.processor.disconnect();
    capture.source.disconnect();
    capture.silentGain.disconnect();
    capture.stream.getTracks().forEach((track) => track.stop());
    void capture.context.close();
  }

  function processNextAudioChunk() {
    const worker = workerRef.current;
    if (!worker || transcribingRef.current || pendingAudioRef.current.length === 0) return;
    const audio = pendingAudioRef.current.shift();
    if (!audio) return;
    transcribingRef.current = true;
    const asrLanguage = languageRef.current === "hi-IN" ? "hindi" : "english";
    worker.postMessage(
      { type: "transcribe", audio, language: asrLanguage },
      [audio.buffer],
    );
  }

  useEffect(() => {
    const worker = new Worker(new URL("./speech-worker.ts", import.meta.url), { type: "module" });
    workerRef.current = worker;
    worker.onmessage = (event: MessageEvent<SpeechWorkerMessage>) => {
      const message = event.data;
      if (message.type === "progress") {
        const percent = Number.isFinite(message.progress) ? Math.round(message.progress!) : null;
        setStatus(percent === null
          ? "Loading speech model on this device…"
          : "Downloading speech model… " + percent + "%");
      } else if (message.type === "ready") {
        setStatus(listeningRef.current ? "Listening. Audio stays on this device." : "Speech model ready. Start listening when you are ready.");
      } else if (message.type === "result") {
        const text = message.text.trim();
        if (text) {
          setHeardText((current) => current.trim()
            ? current.trimEnd() + " " + text
            : text);
        }
        transcribingRef.current = false;
        processNextAudioChunk();
        if (listeningRef.current) setStatus("Listening. Words are transcribed every few seconds.");
      } else {
        transcribingRef.current = false;
        listeningRef.current = false;
        releaseMicrophone();
        setListening(false);
        setStatus("Speech model could not load. Check the internet connection, then press Start listening to retry.");
      }
    };
    worker.onerror = () => {
      listeningRef.current = false;
      releaseMicrophone();
      setListening(false);
      setStatus("The speech model stopped unexpectedly. Press Start listening to retry.");
    };

    return () => {
      listeningRef.current = false;
      releaseMicrophone();
      worker.terminate();
      workerRef.current = null;
    };
  }, []);

  async function startListening() {
    const worker = workerRef.current;
    if (!worker || listeningRef.current) return;
    setStatus("Loading speech model and microphone…");
    worker.postMessage({ type: "load" });
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: { echoCancellation: true, noiseSuppression: true, autoGainControl: true },
      });
      const context = new AudioContext();
      await context.resume();
      const source = context.createMediaStreamSource(stream);
      const processor = context.createScriptProcessor(4096, 1, 1);
      const silentGain = context.createGain();
      silentGain.gain.value = 0;
      audioChunksRef.current = [];
      sampleCountRef.current = 0;
      source.connect(processor);
      processor.connect(silentGain);
      silentGain.connect(context.destination);

      processor.onaudioprocess = (event) => {
        const samples = event.inputBuffer.getChannelData(0).slice();
        audioChunksRef.current.push(samples);
        sampleCountRef.current += samples.length;
        if (sampleCountRef.current >= context.sampleRate * CHUNK_SECONDS) {
          const audio = resample(audioChunksRef.current, context.sampleRate);
          audioChunksRef.current = [];
          sampleCountRef.current = 0;
          pendingAudioRef.current.push(audio);
          processNextAudioChunk();
        }
      };

      captureRef.current = { stream, context, source, processor, silentGain };
      listeningRef.current = true;
      setListening(true);
      setStatus("Listening. Words are transcribed every few seconds.");
    } catch (error) {
      listeningRef.current = false;
      releaseMicrophone();
      setListening(false);
      setStatus(
        error instanceof DOMException && error.name === "NotAllowedError"
          ? "Microphone access is blocked. Allow microphone access for this site, then try again."
          : "Could not start the microphone. Check that it is connected and available.",
      );
    }
  }

  function stopListening() {
    listeningRef.current = false;
    const capture = captureRef.current;
    if (capture && sampleCountRef.current >= capture.context.sampleRate) {
      const audio = resample(audioChunksRef.current, capture.context.sampleRate);
      pendingAudioRef.current.push(audio);
    }
    audioChunksRef.current = [];
    sampleCountRef.current = 0;
    releaseMicrophone();
    setListening(false);
    setStatus(pendingAudioRef.current.length
      ? "Microphone off. Finishing the last transcription…"
      : "Stopped listening.");
    processNextAudioChunk();
  }

  return (
    <div className="panel voice-card">
      <div className="card-icon"><Mic size={19} /></div>
      <h3>Speech to text</h3>
      <p>Live transcription runs in your browser. The first time, it downloads a speech model; your microphone audio stays on this device.</p>
      <label htmlFor="heardText">Recognized speech</label>
      <textarea
        id="heardText"
        value={heardText}
        onChange={(event) => setHeardText(event.target.value)}
        placeholder="Speech transcript will appear here, or type/edit it…"
      />
      <div className="card-actions">
        <button className="secondary-btn" onClick={listening ? stopListening : () => void startListening()}>
          {listening ? <><MicOff size={16} /> Stop listening</> : <><Mic size={16} /> Start listening</>}
        </button>
        <span className="small-note" role="status" aria-live="polite">{status}</span>
      </div>
    </div>
  );
}
