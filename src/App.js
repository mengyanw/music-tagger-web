import { useState } from "react";
import { AUDIO_PATH, MODEL_PATH, Demo } from "./utils";

function App() {
  const [runningResult, setRunningResult] = useState([]);
  const handleDemoRequest = async () => {
    const TopNIndex = await Demo(AUDIO_PATH, MODEL_PATH);
    setRunningResult(TopNIndex)
  }
  return (
    <div className="App">
      <button onClick={handleDemoRequest}>Demo</button>
      {runningResult}
    </div>
  );
}

export default App;
