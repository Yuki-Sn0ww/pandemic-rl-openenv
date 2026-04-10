"""
Pandemic RL — OpenEnv API Server
=================================
FastAPI server exposing the Pandemic RL environment for evaluation.

Endpoints:
  POST /reset  → Initialize/reset environment, returns observation
  POST /step   → Take an action, returns (observation, reward, done, info)
  GET  /state  → Returns full environment state
"""

import sys
import os

# Ensure project root is importable
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from fastapi import FastAPI
from fastapi.responses import HTMLResponse

app = FastAPI(
    title="Pandemic RL",
    description="OpenEnv Pandemic RL Simulation API",
    version="1.0.0",
)

# Global environment instance
env = None


@app.get("/health")
def health():
    """Health check (JSON)."""
    return {"message": "Pandemic RL Server Running", "status": "ok"}


@app.get("/", response_class=HTMLResponse)
def root():
    """Dashboard UI for Hugging Face Space."""
    return """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Pandemic RL Simulation</title>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
<style>
  *{margin:0;padding:0;box-sizing:border-box}
  body{font-family:'Inter',sans-serif;background:#0a0e1a;color:#e2e8f0;min-height:100vh}
  .bg{position:fixed;inset:0;z-index:-1;background:radial-gradient(ellipse at 20% 50%,rgba(99,102,241,.12) 0%,transparent 50%),radial-gradient(ellipse at 80% 20%,rgba(236,72,153,.08) 0%,transparent 50%),#0a0e1a}
  .container{max-width:960px;margin:0 auto;padding:2rem 1.5rem}
  header{text-align:center;margin-bottom:2.5rem}
  header h1{font-size:2rem;font-weight:700;background:linear-gradient(135deg,#818cf8,#c084fc,#f472b6);-webkit-background-clip:text;-webkit-text-fill-color:transparent;margin-bottom:.4rem}
  header p{color:#94a3b8;font-size:.95rem}
  .badge{display:inline-block;padding:.25rem .7rem;border-radius:9999px;font-size:.75rem;font-weight:600;margin-top:.5rem}
  .badge-green{background:rgba(34,197,94,.15);color:#4ade80;border:1px solid rgba(34,197,94,.25)}
  .grid{display:grid;grid-template-columns:1fr 1fr;gap:1rem;margin-bottom:1.5rem}
  @media(max-width:640px){.grid{grid-template-columns:1fr}}
  .card{background:rgba(30,41,59,.55);backdrop-filter:blur(12px);border:1px solid rgba(148,163,184,.1);border-radius:12px;padding:1.25rem;transition:border-color .2s}
  .card:hover{border-color:rgba(129,140,248,.3)}
  .card h3{font-size:.8rem;text-transform:uppercase;letter-spacing:.06em;color:#94a3b8;margin-bottom:.6rem}
  .card-full{grid-column:1/-1}
  .actions-row{display:flex;gap:.6rem;flex-wrap:wrap;margin-bottom:1rem}
  .btn{padding:.55rem 1.2rem;border:none;border-radius:8px;font-family:inherit;font-size:.85rem;font-weight:600;cursor:pointer;transition:all .2s}
  .btn-primary{background:linear-gradient(135deg,#6366f1,#8b5cf6);color:#fff}
  .btn-primary:hover{opacity:.85;transform:translateY(-1px)}
  .btn-action{background:rgba(99,102,241,.12);color:#a5b4fc;border:1px solid rgba(99,102,241,.25)}
  .btn-action:hover{background:rgba(99,102,241,.22)}
  .btn-action.active{background:rgba(34,197,94,.15);color:#4ade80;border-color:rgba(34,197,94,.3)}
  #log{background:rgba(15,23,42,.7);border:1px solid rgba(148,163,184,.08);border-radius:10px;padding:1rem;font-family:'Courier New',monospace;font-size:.8rem;line-height:1.6;max-height:320px;overflow-y:auto;color:#cbd5e1;white-space:pre-wrap}
  .stat{font-size:1.6rem;font-weight:700;color:#f8fafc}
  .stat-label{font-size:.75rem;color:#64748b;margin-top:.15rem}
  .stat-row{display:flex;gap:1.5rem;flex-wrap:wrap}
  .stat-item{min-width:80px}
  table{width:100%;border-collapse:collapse;font-size:.82rem}
  th{text-align:left;color:#94a3b8;font-weight:500;padding:.4rem .6rem;border-bottom:1px solid rgba(148,163,184,.12)}
  td{padding:.4rem .6rem;border-bottom:1px solid rgba(148,163,184,.06);color:#e2e8f0}
  .city-bar{height:6px;border-radius:3px;margin-top:2px}
  .legend{display:flex;gap:1rem;font-size:.72rem;color:#94a3b8;margin-top:.5rem;flex-wrap:wrap}
  .legend span::before{content:'';display:inline-block;width:8px;height:8px;border-radius:2px;margin-right:4px;vertical-align:middle}
  .legend .s::before{background:#22d3ee} .legend .i::before{background:#f87171}
  .legend .r::before{background:#4ade80} .legend .d::before{background:#94a3b8}
</style>
</head>
<body>
<div class="bg"></div>
<div class="container">
  <header>
    <h1>🦠 Pandemic RL Simulation</h1>
    <p>Meta PyTorch OpenEnv Hackathon — 3-City SIR Model</p>
    <div class="badge badge-green">● Server Running</div>
  </header>

  <div class="actions-row">
    <button class="btn btn-primary" onclick="doReset()">Reset Environment</button>
    <button class="btn btn-action" onclick="doStep(0)">⏸ Do Nothing</button>
    <button class="btn btn-action" onclick="doStep(1)">🔒 Quar. C0</button>
    <button class="btn btn-action" onclick="doStep(2)">🔒 Quar. C1</button>
    <button class="btn btn-action" onclick="doStep(3)">🔒 Quar. C2</button>
    <button class="btn btn-action" onclick="doStep(4)">💉 Vacc. C0</button>
    <button class="btn btn-action" onclick="doStep(5)">💉 Vacc. C1</button>
    <button class="btn btn-action" onclick="doStep(6)">💉 Vacc. C2</button>
  </div>

  <div class="grid">
    <div class="card">
      <h3>Simulation Stats</h3>
      <div class="stat-row">
        <div class="stat-item"><div class="stat" id="st-step">—</div><div class="stat-label">Step</div></div>
        <div class="stat-item"><div class="stat" id="st-reward">—</div><div class="stat-label">Reward</div></div>
        <div class="stat-item"><div class="stat" id="st-done">—</div><div class="stat-label">Done</div></div>
      </div>
    </div>
    <div class="card">
      <h3>City Health Overview</h3>
      <table>
        <thead><tr><th>City</th><th>Suscept.</th><th>Infected</th><th>Recovered</th><th>Dead</th></tr></thead>
        <tbody id="city-table"><tr><td colspan="5" style="color:#64748b">Click "Reset Environment" to begin</td></tr></tbody>
      </table>
      <div class="legend"><span class="s">Susceptible</span><span class="i">Infected</span><span class="r">Recovered</span><span class="d">Dead</span></div>
    </div>
    <div class="card card-full">
      <h3>Event Log</h3>
      <div id="log">Pandemic RL Server ready.\\nClick "Reset Environment" to initialize the simulation.\\n</div>
    </div>
  </div>

  <div style="text-align:center;color:#475569;font-size:.75rem;margin-top:1rem">
    API endpoints: <code>POST /reset</code> · <code>POST /step?action=N</code> · <code>GET /state</code> · <code>GET /health</code>
  </div>
</div>

<script>
const log=document.getElementById('log');
let stepNum=0,totalReward=0;
function addLog(m){log.textContent+=m+'\\n';log.scrollTop=log.scrollHeight}
function fmt(v){return(v*1000).toFixed(0)}

function updateCities(obs){
  if(!obs||obs.length<12)return;
  let html='';
  for(let c=0;c<3;c++){
    const s=obs[c*4],i=obs[c*4+1],r=obs[c*4+2],d=obs[c*4+3];
    html+=`<tr><td><strong>City ${c}</strong></td><td>${fmt(s)}</td><td style="color:#f87171">${fmt(i)}</td><td style="color:#4ade80">${fmt(r)}</td><td>${fmt(d)}</td></tr>`;
  }
  document.getElementById('city-table').innerHTML=html;
}

async function doReset(){
  addLog('→ Resetting environment...');
  try{
    const r=await fetch('/reset',{method:'POST'});
    const d=await r.json();
    if(d.error){addLog('✗ '+d.error);return}
    stepNum=0;totalReward=0;
    document.getElementById('st-step').textContent='0';
    document.getElementById('st-reward').textContent='0.00';
    document.getElementById('st-done').textContent='No';
    updateCities(d.observation);
    addLog('✓ Environment reset. Ready for actions.');
  }catch(e){addLog('✗ Error: '+e)}
}

async function doStep(a){
  const labels=['Do Nothing','Quarantine City 0','Quarantine City 1','Quarantine City 2','Vaccinate City 0','Vaccinate City 1','Vaccinate City 2'];
  addLog('→ Step '+(stepNum+1)+': '+labels[a]);
  try{
    const r=await fetch('/step?action='+a,{method:'POST'});
    const d=await r.json();
    if(d.error){addLog('✗ '+d.error);return}
    stepNum++;totalReward+=d.reward||0;
    document.getElementById('st-step').textContent=stepNum;
    document.getElementById('st-reward').textContent=totalReward.toFixed(2);
    document.getElementById('st-done').textContent=d.done?'Yes':'No';
    updateCities(d.observation);
    addLog('  reward='+((d.reward||0).toFixed(3))+' | done='+d.done);
    if(d.done)addLog('\\n🏁 Simulation complete after '+stepNum+' steps. Total reward: '+totalReward.toFixed(2));
  }catch(e){addLog('✗ Error: '+e)}
}
</script>
</body>
</html>"""


@app.post("/reset")
def reset():
    """Initialize or reset the environment. Must be called before /step or /state."""
    global env
    try:
        from env.environment import PandemicEnv
        from env.tasks import TASK_EASY

        env = PandemicEnv(config=TASK_EASY["config"], seed=42)
        obs = env.reset()
        return {"observation": obs}
    except Exception as e:
        return {"error": f"Failed to reset environment: {e}"}


@app.post("/step")
def step(action: int = 0):
    """Take one step in the environment."""
    global env
    if env is None:
        return {"error": "Environment not initialized. Call /reset first."}
    try:
        obs, reward, done, info = env.step(action)
        return {
            "observation": obs,
            "reward": reward,
            "done": done,
            "info": info,
        }
    except Exception as e:
        return {"error": f"Step failed: {e}"}


@app.get("/state")
def state():
    """Return full environment state."""
    global env
    if env is None:
        return {"error": "Environment not initialized. Call /reset first."}
    try:
        return env.state()
    except Exception as e:
        return {"error": f"State retrieval failed: {e}"}


# ── Entry point ──────────────────────────────────────────────────────────────

def main():
    """Required by pyproject.toml scripts entry."""
    import uvicorn
    uvicorn.run("server.app:app", host="0.0.0.0", port=8000)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server.app:app", host="0.0.0.0", port=8000)