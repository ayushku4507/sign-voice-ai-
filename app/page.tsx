"use client";

import { useEffect, useRef, useState } from "react";
import { AudioLines, Camera, CircleHelp, Languages, Mic, MicOff, ShieldCheck, Volume2 } from "lucide-react";

type SpeechRecognitionResult = { transcript: string };
type SpeechRecognitionEventLike = { results: ArrayLike<ArrayLike<SpeechRecognitionResult>> };
type SpeechRecognitionLike = {
  lang: string;
  interimResults: boolean;
  onresult: ((event: SpeechRecognitionEventLike) => void) | null;
  onerror: (() => void) | null;
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
  const [cameraOn, setCameraOn] = useState(false);
  const [cameraError, setCameraError] = useState("");
  const [spokenText, setSpokenText] = useState("");
  const [heardText, setHeardText] = useState("");
  const [language, setLanguage] = useState("en-IN");
  const [listening, setListening] = useState(false);

  useEffect(() => () => {
    streamRef.current?.getTracks().forEach((track) => track.stop());
    recognitionRef.current?.stop();
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
    if (!("speechSynthesis" in window) || !text.trim()) return;
    window.speechSynthesis.cancel();
    const utterance = new SpeechSynthesisUtterance(text);
    utterance.lang = language;
    window.speechSynthesis.speak(utterance);
  }

  function toggleListening() {
    if (listening) {
      recognitionRef.current?.stop();
      setListening(false);
      return;
    }
    const speechWindow = window as SpeechWindow;
    const Recognition = speechWindow.SpeechRecognition ?? speechWindow.webkitSpeechRecognition;
    if (!Recognition) {
      setHeardText("Speech recognition is not supported in this browser. Try a supported browser or type the message.");
      return;
    }
    const recognition = new Recognition();
    recognition.lang = language;
    recognition.interimResults = true;
    recognition.onresult = (event) => {
      const transcript = Array.from(event.results).map((result) => result[0]?.transcript ?? "").join("");
      setHeardText(transcript);
    };
    recognition.onerror = () => setListening(false);
    recognition.onend = () => setListening(false);
    recognitionRef.current = recognition;
    try {
      recognition.start();
      setListening(true);
    } catch {
      setListening(false);
      setHeardText("Could not start microphone recognition. Check browser permissions.");
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
        <p className="hero-copy">A work-in-progress communication toolkit for Indian Sign Language and voice. Designed with clarity, dignity and privacy in mind.</p>
        <div className="hero-actions"><a className="primary-btn" href="#translator">Open translator <span>↗</span></a><a className="text-btn" href="#how-it-works">How it works</a></div>
        <div className="trust-row"><span><ShieldCheck size={15} /> Camera stays in your browser</span><span><CircleHelp size={15} /> Recognition model not connected yet</span></div>
      </section>

      <section className="workspace" id="translator">
        <div className="section-heading"><div><div className="eyebrow">01 / LIVE WORKSPACE</div><h2>Sign to text</h2></div><span className="pill muted">Model integration pending</span></div>
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
            <p className="small-note">No video is uploaded or recorded by this prototype.</p>
          </div>
          <div className="panel output-panel">
            <div className="panel-head"><span><Languages size={17} /> Recognition output</span><span className="pill warning">Not connected</span></div>
            <div className="empty-output"><div className="wave-symbol">〰</div><strong>Waiting for a validated ISL model</strong><p>Hand tracking alone does not understand sign language. A trained, tested ISL model is required before predictions can appear here.</p></div>
            <div className="confidence"><span>MODEL CONFIDENCE</span><strong>—</strong></div>
            <div className="model-foot"><span className="dot amber" /> No predictions are being generated</div>
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
        <div className="steps"><div><span>01</span><h3>Capture</h3><p>Get permission and capture hand movement locally.</p></div><div><span>02</span><h3>Recognize</h3><p>Use a validated ISL model—not hand landmarks alone.</p></div><div><span>03</span><h3>Communicate</h3><p>Show uncertainty and let users correct the output.</p></div></div>
      </section>

      <footer><a className="brand" href="#"><span className="brand-mark"><AudioLines size={18} /></span>SignVoice</a><span>Open-source project • Recognition model under development</span><a href="https://github.com/" target="_blank" rel="noreferrer">GitHub ↗</a></footer>
    </main>
  );
}
