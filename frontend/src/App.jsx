import { BrowserRouter, Routes, Route, Link, useNavigate, useLocation } from "react-router-dom";
import { useEffect, useRef, useState } from "react";
import "./App.css";

const BACKEND = "https://emotion-music-backend-4lln.onrender.com";
const MOODS = ["Auto", "Happy", "Energetic", "Calm", "Sad"];

function Navbar() {
  return (
    <nav className="navbar">
      <div className="nav-brand">🎭🎵 EmotionMusic AI</div>
      <div className="nav-links">
        <Link to="/">Home</Link>
        <Link to="/experience">Experience</Link>
        <Link to="/how-it-works">How It Works</Link>
        <Link to="/demo">Demo Mode</Link>
      </div>
    </nav>
  );
}

function Home() {
  return (
    <section className="hero page-screen">
      <div className="hero-content">
        <div className="badge">Your emotions, your soundtrack</div>
        <h1>
          Turn Your <span>Emotion</span>
          <br />Into Music
        </h1>
        <p>
          Capture your facial expression, detect your emotion, map it to a mood,
          and receive best song recommendations.
        </p>

        <div className="hero-actions">
          <Link to="/experience">
            <button className="primary big">Start Experience</button>
          </Link>
          <Link to="/how-it-works">
            <button className="ghost big">See Pipeline</button>
          </Link>
        </div>

        <div className="stats">
          <div><b>97.23%</b><span>CK+ Accuracy</span></div>
          <div><b>93.02%</b><span>Music Model</span></div>
          <div><b>Real-Time</b><span>Web Demo</span></div>
        </div>
      </div>

      <div className="hero-visual">
        <div className="album-card floating">
          <div className="album-art">🎧</div>
          <h3>Now Matching</h3>
          <p>Emotion → Mood → Music</p>
          <div className="mini-eq"><i></i><i></i><i></i><i></i><i></i></div>
        </div>
      </div>
    </section>
  );
}

function Experience() {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const location = useLocation();

  const initialMood = location.state?.selectedMood || "Auto";

  const [status, setStatus] = useState("Ready");
  const [emotion, setEmotion] = useState("—");
  const [mood, setMood] = useState("—");
  const [probs, setProbs] = useState({});
  const [selectedMood, setSelectedMood] = useState(initialMood);
  const [recommendations, setRecommendations] = useState([]);
  const [history, setHistory] = useState([]);
  const [favorites, setFavorites] = useState(() => JSON.parse(localStorage.getItem("emr_favs") || "[]"));
  const [nowPlaying, setNowPlaying] = useState(null);
  const [lastBlob, setLastBlob] = useState(null);

  useEffect(() => {
    navigator.mediaDevices
      .getUserMedia({ video: true, audio: false })
      .then((stream) => {
        if (videoRef.current) videoRef.current.srcObject = stream;
      })
      .catch(() => setStatus("Camera permission denied"));
  }, []);

  function saveFavorites(next) {
    setFavorites(next);
    localStorage.setItem("emr_favs", JSON.stringify(next));
  }

  function trackKey(t) {
    return `${t.track}||${t.artist}`.toLowerCase();
  }

  function isFavorite(t) {
    if (!t) return false;
    return favorites.some((x) => trackKey(x) === trackKey(t));
  }

  function toggleFavorite() {
    if (!nowPlaying) return;
    const exists = isFavorite(nowPlaying);

    const next = exists
      ? favorites.filter((x) => trackKey(x) !== trackKey(nowPlaying))
      : [nowPlaying, ...favorites].slice(0, 25);

    saveFavorites(next);
  }

  async function captureFrame() {
    const ctx = canvasRef.current.getContext("2d");
    ctx.drawImage(videoRef.current, 0, 0, canvasRef.current.width, canvasRef.current.height);
    return new Promise((resolve) => canvasRef.current.toBlob(resolve, "image/jpeg", 0.95));
  }

  async function callPredict(blob) {
    const form = new FormData();
    form.append("file", blob, "frame.jpg");
    form.append("mood_override", selectedMood !== "Auto" ? selectedMood : "");

    const res = await fetch(`${BACKEND}/predict`, {
      method: "POST",
      body: form,
    });

    return await res.json();
  }

  async function runPipeline(refresh = false) {
    try {
      setStatus(refresh ? "Refreshing..." : "Capturing...");

      let blob = lastBlob;

      if (!refresh) {
        blob = await captureFrame();
        setLastBlob(blob);
      }

      if (!blob) throw new Error("No captured frame yet.");

      setStatus("Running AI models...");
      const data = await callPredict(blob);

      setEmotion(data.emotion);
      setMood(data.target_mood);
      setProbs(data.emotion_probs);
      setRecommendations(data.recommendations || []);

      if (data.recommendations?.length > 0) {
        setNowPlaying({ ...data.recommendations[0], mood: data.target_mood });
      }

      setHistory((prev) => [
        {
          emotion: data.emotion,
          mood: data.target_mood,
          time: new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }),
        },
        ...prev,
      ].slice(0, 5));

      setStatus("Done ✅");
    } catch (err) {
      console.error(err);
      setStatus("Error. Check backend or browser console.");
    }
  }

  const topEmotions = Object.entries(probs).sort((a, b) => b[1] - a[1]).slice(0, 3);

  return (
    <main className="wrap page-screen">
      <div className="section-title">
        <span>Live Experience</span>
        <h2>Capture emotion and get music recommendations</h2>
        {selectedMood !== "Auto" && (
          <p className="muted">Demo mood selected: <b>{selectedMood}</b>. Capture your face, and songs will use this mood.</p>
        )}
      </div>

      <div className="layout">
        <section className="glass-card">
          <h3>Camera</h3>
          <video ref={videoRef} autoPlay playsInline />
          <canvas ref={canvasRef} width="224" height="224" style={{ display: "none" }} />

          <div className="controls">
            <button className="primary" onClick={() => runPipeline(false)}>Capture & Recommend</button>
            <button disabled={!lastBlob} onClick={() => runPipeline(true)}>Refresh</button>
          </div>

          <div className="status">Status: {status}</div>

          <div className="metric-grid">
            <div className="metric"><span>Emotion</span><b>{emotion}</b></div>
            <div className="metric"><span>Mood</span><b>{mood}</b></div>
          </div>

          <h3>Mood Override</h3>
          <div className="chips">
            {MOODS.map((m) => (
              <button
                key={m}
                className={selectedMood === m ? "chip active" : "chip"}
                onClick={() => setSelectedMood(m)}
              >
                {m}
              </button>
            ))}
          </div>

          <h3>Top Emotions</h3>
          {topEmotions.length === 0 && <p className="muted">Capture an image to see probabilities.</p>}
          {topEmotions.map(([name, value]) => (
            <div className="emo-row" key={name}>
              <span>{name}</span>
              <div className="bar-wrap">
                <div className="bar" style={{ width: `${(value * 100).toFixed(1)}%` }} />
              </div>
              <span>{(value * 100).toFixed(1)}%</span>
            </div>
          ))}

          <h3>History</h3>
          {history.length === 0 && <p className="muted">No captures yet.</p>}
          {history.map((h, i) => (
            <div className="history-item" key={i}>
              <span>🎭 {h.emotion} → 🎵 {h.mood}</span>
              <span>{h.time}</span>
            </div>
          ))}
        </section>

        <section>
          <div className="now-playing">
            <div>
              <span className="small-label">Now Playing</span>
              <h2>{nowPlaying?.track || "No track selected"}</h2>
              <p>{nowPlaying?.artist || "Click a recommendation to preview it"}</p>
              <small>{nowPlaying ? `Mood: ${nowPlaying.mood}` : "—"}</small>
            </div>

            <div className="now-actions">
              <div className={nowPlaying ? "equalizer playing" : "equalizer"}>
                <i></i><i></i><i></i><i></i><i></i>
              </div>
              <button onClick={toggleFavorite}>{isFavorite(nowPlaying) ? "♥" : "♡"}</button>
              {nowPlaying && <a href={nowPlaying.spotify_url} target="_blank" rel="noreferrer">Open</a>}
            </div>
          </div>

          <div className="glass-card">
            <h3>Recommendations</h3>
            <div className="songs">
              {recommendations.length === 0 && <p className="muted">Capture your expression to generate songs.</p>}
              {recommendations.map((r, i) => (
                <div className="song-card" key={i} onClick={() => setNowPlaying({ ...r, mood })}>
                  <div>
                    <b>{r.track}</b>
                    <p>{r.artist}</p>
                    <small>Score: {Number(r.score).toFixed(4)}</small>
                  </div>
                  <a href={r.spotify_url} target="_blank" rel="noreferrer">Spotify</a>
                </div>
              ))}
            </div>
          </div>

          <div className="glass-card">
            <h3>Favorites</h3>
            {favorites.length === 0 && <p className="muted">No favorites yet.</p>}
            {favorites.map((f, i) => (
              <div className="song-card" key={i}>
                <div><b>{f.track}</b><p>{f.artist}</p></div>
                <a href={f.spotify_url} target="_blank" rel="noreferrer">Open</a>
              </div>
            ))}
          </div>
        </section>
      </div>
    </main>
  );
}

function HowItWorks() {
  const steps = [
    ["📷", "Webcam Capture", "The browser captures one frame from the live camera feed."],
    ["🙂", "Face Detection", "OpenCV crops the face region before inference."],
    ["🧠", "ConvNeXt FER", "The model predicts the facial emotion."],
    ["🎚️", "Mood Mapping", "Emotion is mapped into Happy, Calm, Sad, or Energetic."],
    ["🌳", "XGBoost Model", "Audio features are scored for the target mood."],
    ["🎧", "Spotify Links", "Top songs are returned with Spotify search links."],
  ];

  return (
    <section className="wrap page-screen">
      <div className="section-title">
        <span>Pipeline</span>
        <h2>How the system works</h2>
      </div>

      <div className="pipeline">
        {steps.map((step, i) => (
          <div className="pipeline-card" key={i}>
            <div className="pipeline-icon">{step[0]}</div>
            <h3>{step[1]}</h3>
            <p>{step[2]}</p>
          </div>
        ))}
      </div>
    </section>
  );
}

function DemoMode() {
  const navigate = useNavigate();

  function chooseMood(mood) {
    navigate("/experience", { state: { selectedMood: mood } });
  }

  return (
    <section className="wrap page-screen">
      <div className="section-title">
        <span>Demo Mode</span>
        <h2>Select a mood and continue to the recommender</h2>
      </div>

      <div className="demo-card">
        <p>
          Choose a mood below. You will be redirected to the Experience page.
          There, capture your face once, and the selected mood will automatically
          override the detected emotion for song recommendations.
        </p>

        <div className="demo-buttons">
          {["Happy", "Energetic", "Calm", "Sad"].map((m) => (
            <button key={m} onClick={() => chooseMood(m)}>
              {m}
            </button>
          ))}
        </div>
      </div>
    </section>
  );
}

export default function App() {
  return (
    <BrowserRouter>
      <div className="app">
        <div className="animated-bg">
          <span>♪</span><span>♫</span><span>♬</span><span>♩</span><span>♪</span>
        </div>

        <Navbar />

        <Routes>
          <Route path="/" element={<Home />} />
          <Route path="/experience" element={<Experience />} />
          <Route path="/how-it-works" element={<HowItWorks />} />
          <Route path="/demo" element={<DemoMode />} />
        </Routes>
      </div>
    </BrowserRouter>
  );
}