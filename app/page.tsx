"use client";

import { useEffect, useRef, useState } from "react";
import ASLTrainer from "./ASLTrainer";
import { AudioLines, Camera, CircleHelp, Languages, Mic, MicOff, ShieldCheck, Volume2 } from "lucide-react";

type SpeechRecognitionAlternative = { transcript: string };
type SpeechRecognitionResult = { 0: SpeechRecognitionAlternative; isFinal: boolean };
type SpeechRecognitionEventLike = { results: ArrayLike<SpeechRecognitionResult> };
type SpeechRecognitionErrorLike = { error?: string };
type SpeechRecognitionLike = {
  lang: string;
  continuous: boolean;
  interimResults: boolean;
  onresult: ((event: SpeechRecognitionEventLike) => void) | null;
  onerror: ((event: SpeechRecognitionErrorLike) => void) | null;
  onend: (() => void) | null;
  start: () => void;
  stop: () => void;
};
type SpeechWindow = Window & {
  SpeechRecognition?: new () => SpeechRecognitionLike;
  webkitSpeechRecognition?: new () => SpeechRecognitionLike;
};

const phrases = [
  "I need help.",
  "Please call my family.",
  "I need medical assistance.",
  "I cannot hear you. Please type.",
  "Please speak slowly."
];

export default function Home() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const recognitionRef = useRef<SpeechRecognitionLike | null>(null);
  const shouldListenRef = useRef(false);
  const restartTimerRef = useRef<number | null>(null);
  const activeSpeechRef = useRef<{ text: string; startedAt: number } | null>(null);
  const [cameraOn, setCameraOn] = useState(false);
  const [cameraError, setCameraError] = useState("");
  const [spokenText, setSpokenText] = useState("");
  const [heardText, setHeardText] = useState("");
  const [language, setLanguage] = useState("en-IN");
  const [listening, setListening] = useState(false);

  useEffect(() => () => {
    streamRef.current?.getTracks().forEach((track) => track.stop());
    shouldListenRef.current = false;
    if (restartTimerRef.current !== null) window.clearTimeout(restartTimerRef.current);
    recognitionRef.current?.stop();
    window.speechSynthesis?.cancel();
  }, []);

  async function toggleCamera() {
    if (cameraOn) {
      streamRef.current?.getTracks().forEach((track) => track.stop());
      streamRef.current = null;
      if (videoRef.current) videoRef.current.srcObject = null;
      setCameraOn(false);
      return;
    }
    try {
      setCameraError("");
      const stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: "user" }, audio: false });
      streamRef.current = stream;
      if (videoRef.current) videoRef.current.srcObject = stream;
      setCameraOn(true);
    } catch {
      setCameraError("Camera permission was denied or no camera is available. Check your browser permissions.");
    }
  }

  function speak(text: string) {
    const cleanText = text.trim();
    if (!("speechSynthesis" in window) || !cleanText) return;
    const active = activeSpeechRef.current;
    const now = Date.now();
    // Ignore duplicate activations for the same phrase while its first utterance is starting.
    if (active?.text === cleanText && now - active.startedAt < 1200) return;
    activeSpeechRef.current = { text: cleanText, startedAt: now };
    window.speechSynthesis.cancel();
    const utterance = new SpeechSynthesisUtterance(cleanText);
    utterance.lang = language;
    utterance.rate = 1;
    utterance.onend = () => {
      if (activeSpeechRef.current?.text === cleanText) activeSpeechRef.current = null;
    };
    utterance.onerror = () => {
      if (activeSpeechRef.current?.text === cleanText) activeSpeechRef.current = null;
    };
    window.speechSynthesis.speak(utterance);
  }

  function toggleListening() {
    if (shouldListenRef.current) {
      shouldListenRef.current = false;
      if (restartTimerRef.current !== null) window.clearTimeout(restartTimerRef.current);
      restartTimerRef.current = null;
      recognitionRef.current?.stop();
      setListening(false);
      return;
    }
    const speechWindow = window as SpeechWindow;
    const Recognition = speechWindow.SpeechRecognition ?? speechWindow.webkitSpeechRecognition;
    if (!Recognition) {
      setHeardText("Speech recognition is not available in this browser. Try Chrome or Edge, allow microphone access, or type the message.");
      return;
    }
    const recognition = new Recognition();
    recognition.lang = language;
    recognition.continuous = true;
    recognition.interimResults = true;
    shouldListenRef.current = true;
    recognition.onresult = (event) => {
      const transcript = Array.from(event.results)
        .map((result) => result[0]?.transcript ?? "")
        .join("");
      setHeardText(transcript);
    };
    recognition.onerror = (event) => {
      const fatal = event.error === "not-allowed" || event.error === "service-not-allowed" || event.error === "audio-capture";
      if (fatal) {
        shouldListenRef.current = false;
        setListening(false);
        setHeardText(
          event.error === "audio-capture"
            ? "No microphone was found. Connect or enable a microphone, then try again."
            : "Microphone access was blocked. Allow microphone access for this site, then try again.",
        );
      }
    };
    recognition.onend = () => {
      if (!shouldListenRef.current) {
        setListening(false);
        return;
      }
      // Speech services may stop after pauses. Restart while the user still has listening enabled.
      restartTimerRef.current = window.setTimeout(() => {
        if (!shouldListenRef.current) return;
        try {
          recognition.start();
        } catch {
          restartTimerRef.current = window.setTimeout(() => {
            if (!shouldListenRef.current) return;
            try {
              recognition.start();
            } catch {
              shouldListenRef.current = false;
              setListening(false);
              setHeardText("Speech recognition stopped. Press Start listening to try again.");
            }
          }, 500);
        }
      }, 250);
    };
    recognitionRef.current = recognition;
    try {
      recognition.start();
      setListening(true);
    } catch {
      shouldListenRef.current = false;
      setListening(false);
      setHeardText("Could not start microphone recognition. Allow microphone access and try again.");
    }
  }

  return (
    <main className="shell">
      <header className="topbar">
        <a className="brand" href="#"><span className="brand-mark"><AudioLines size={21} /></span>SignVoice</a>
        <nav aria-label="Main navigation"><a href="#translator">Translator</a><a href="#voice">Voice tools</a><a href="#emergency">Quick phrases</a></nav>
        <span className="status"><span className="dot" /> Prototype</span>
      </header>

      <section className="hero">
        <div className="eyebrow"><span className="eyebrow-line" /> ACCESSIBLE COMMUNICATION, BUILT TOGETHER</div>
        <h1>Make every<br /><span>conversation</span> count.</h1>
        <p className="hero-copy">A work-in-progress toolkit for ASL fingerspelling and voice. The camera model learns from examples recorded by you in this browser.</p>
        <div className="hero-actions"><a className="primary-btn" href="#translator">Open translator <span>↗</span></a><a className="text-btn" href="#how-it-works">How it works</a></div>
        <div className="trust-row"><span><ShieldCheck size={15} /> Camera stays in your browser</span><span><CircleHelp size={15} /> ASL examples stay on this device</span></div>
      </section>

      <section className="workspace" id="translator">
        <div className="section-heading"><div><div className="eyebrow">01 / LIVE WORKSPACE</div><h2>Sign to text</h2></div><span className="pill muted">Personal ASL alphabet model</span></div>
        <div className="workspace-grid">
          <div className="panel camera-panel">
            <div className="panel-head"><span><Camera size={17} /> Camera input</span><span className="live-label"><i /> {cameraOn ? "CAMERA ON" : "STANDBY"}</span></div>
            <div className="camera-view">
              <video ref={videoRef} autoPlay playsInline muted aria-label="Camera preview" />
              {!cameraOn && <div className="camera-placeholder"><div className="camera-icon"><Camera size={28} /></div><strong>Your camera preview</strong><span>Enable your camera to begin.</span></div>}
              <div className="camera-corner top-left" /><div className="camera-corner top-right" /><div className="camera-corner bottom-left" /><div className="camera-corner bottom-right" />
            </div>
            <button className="primary-btn full" onClick={toggleCamera}>{cameraOn ? "Turn camera off" : "Enable camera"}</button>
            {cameraError && <p className="error" role="alert">{cameraError}</p>}
            <p className="small-note">Video is not uploaded. Training saves hand landmarks in this browser only.</p>
          </div>
          <div className="panel output-panel">
            <div className="panel-head"><span><Languages size={17} /> ASL fingerspelling</span><span className="pill warning">{cameraOn ? "Hand tracking on" : "Camera off"}</span></div>
            <ASLTrainer cameraOn={cameraOn} videoRef={videoRef} />
          </div>
        </div>
      </section>

      <section className="section" id="voice">
        <div className="section-heading"><div><div className="eyebrow">02 / VOICE TOOLS</div><h2>Keep both sides in the conversation.</h2></div></div>
        <div className="voice-grid">
          <div className="panel voice-card">
            <div className="card-icon"><Mic size={19} /></div><h3>Speech to text</h3><p>Turn spoken words into readable text using your browser&apos;s speech recognition, where supported.</p>
            <label htmlFor="heardText">Recognized speech</label><textarea id="heardText" value={heardText} onChange={(e) => setHeardText(e.target.value)} placeholder="Speech transcript will appear here, or type/edit it…" />
            <div className="card-actions"><button className="secondary-btn" onClick={toggleListening}>{listening ? <><MicOff size={16} /> Stop listening</> : <><Mic size={16} /> Start listening</>}</button><span className="small-note">Browser support varies</span></div>
          </div>
          <div className="panel voice-card">
            <div className="card-icon"><Volume2 size={19} /></div><h3>Text to speech</h3><p>Speak typed text aloud. Available voices depend on your device and browser.</p>
            <label htmlFor="spokenText">Text to speak</label><textarea id="spokenText" value={spokenText} onChange={(e) => setSpokenText(e.target.value)} placeholder="Type a message to speak…" />
            <div className="card-actions"><button className="secondary-btn" onClick={() => speak(spokenText)}><Volume2 size={16} /> Speak message</button><span className="small-note">On-device browser voice</span></div>
          </div>
        </div>
        <div className="language-row"><label htmlFor="language">Speech language</label><select id="language" value={language} onChange={(e) => setLanguage(e.target.value)}><option value="en-IN">English (India)</option><option value="hi-IN">Hindi</option><option value="en-US">English (US)</option></select><span className="small-note">This changes browser speech settings, not automatic translation.</span></div>
      </section>

      <section className="section" id="emergency">
        <div className="section-heading"><div><div className="eyebrow">03 / QUICK COMMUNICATION</div><h2>Useful phrases, one tap away.</h2></div></div>
        <div className="phrase-grid">{phrases.map((phrase) => <button className="phrase" key={phrase} onClick={() => { setSpokenText(phrase); speak(phrase); }}><span>{phrase}</span><span className="phrase-arrow">↗</span></button>)}</div>
        <p className="small-note emergency-note">These buttons only display and speak a phrase. They do not call emergency services or contact another person.</p>
      </section>

      <section className="section" id="how-it-works">
        <div className="section-heading"><div><div className="eyebrow">04 / TRANSPARENT BY DESIGN</div><h2>Built in stages. Tested with people.</h2></div></div>
        <div className="steps"><div><span>01</span><h3>Capture</h3><p>Get permission and capture hand movement locally.</p></div><div><span>02</span><h3>Recognize</h3><p>Learn ASL alphabet handshapes; J and Z use motion samples.</p></div><div><span>03</span><h3>Communicate</h3><p>Show uncertainty and let users correct the output.</p></div></div>
      </section>

      <footer><a className="brand" href="#"><span className="brand-mark"><AudioLines size={18} /></span>SignVoice</a><span>Open-source project • Recognition model under development</span><a href="https://github.com/" target="_blank" rel="noreferrer">GitHub ↗</a></footer>
    </main>
  );
}
