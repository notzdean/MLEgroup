"""Replaces the /mobile endpoint in realtime_inference.py with the new SSE-based passive receiver."""

NEW_MOBILE = '''
@app.get("/mobile", response_class=HTMLResponse)
def mobile_emulator():
    """
    CDA Officer mobile alert receiver.
    Passively listens for alerts via Server-Sent Events (SSE).
    Alerts push instantly when a hospital submits a case that crosses the threshold.
    Officer does not submit anything — pure receiver.
    """
    html = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>CDA Officer - Live Alert Monitor</title>
<link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&display=swap" rel="stylesheet">
<style>
* { margin:0; padding:0; box-sizing:border-box; }
body { background:linear-gradient(135deg,#1a1a2e,#16213e,#0f3460); min-height:100vh;
       display:flex; align-items:center; justify-content:center;
       font-family:"Space Grotesk",sans-serif; padding:20px; }
.scene { display:flex; gap:60px; align-items:flex-start; flex-wrap:wrap; justify-content:center; }
.phone { width:320px; background:#1c1c1e; border-radius:50px; padding:12px;
         box-shadow:0 0 0 2px #3a3a3c,0 40px 80px rgba(0,0,0,.6); }
.phone-screen { background:#000; border-radius:40px; overflow:hidden; min-height:620px; position:relative; }
.notch { width:120px; height:30px; background:#1c1c1e; border-radius:0 0 20px 20px; margin:0 auto; z-index:10; }
.lock-screen { background:linear-gradient(180deg,#1a1a2e,#0d1117); min-height:590px; padding:20px; display:flex; flex-direction:column; }
.status-bar { display:flex; justify-content:space-between; font-size:12px; color:#fff; padding:4px 8px 12px; }
.time-display { text-align:center; margin:20px 0 30px; }
.time-display .time { font-size:56px; font-weight:300; color:#fff; letter-spacing:-2px; line-height:1; }
.time-display .date { font-size:14px; color:rgba(255,255,255,.7); margin-top:4px; }
.notification { background:rgba(255,255,255,.12); backdrop-filter:blur(20px); border-radius:16px;
                padding:14px; margin-bottom:10px; border:1px solid rgba(255,255,255,.1); }
.notification.alert-notif { background:rgba(239,68,68,.2); border:1px solid rgba(239,68,68,.4);
                             animation:alertPulse 2s ease-in-out infinite; }
@keyframes alertPulse { 0%,100%{box-shadow:0 0 0 0 rgba(239,68,68,0)} 50%{box-shadow:0 0 0 8px rgba(239,68,68,.15)} }
.notif-header { display:flex; align-items:center; gap:8px; margin-bottom:8px; }
.notif-icon { width:28px; height:28px; border-radius:8px; display:flex; align-items:center; justify-content:center; font-size:14px; }
.notif-icon.health { background:#007aff; }
.notif-icon.danger { background:#ef4444; }
.notif-app  { font-size:11px; color:rgba(255,255,255,.6); text-transform:uppercase; letter-spacing:.06em; }
.notif-time { font-size:11px; color:rgba(255,255,255,.4); margin-left:auto; }
.notif-title { font-size:13px; font-weight:600; color:#fff; margin-bottom:2px; }
.notif-body  { font-size:12px; color:rgba(255,255,255,.75); line-height:1.4; }
.chips { display:flex; gap:8px; margin-top:10px; }
.chip { background:rgba(239,68,68,.25); border:1px solid rgba(239,68,68,.5); border-radius:8px;
        padding:6px 10px; flex:1; text-align:center; }
.chip-label { font-size:9px; color:rgba(255,255,255,.5); text-transform:uppercase; letter-spacing:.06em; }
.chip-value { font-size:15px; font-weight:700; color:#ef4444; font-family:monospace; }
.actions { display:flex; gap:8px; margin-top:10px; }
.act-btn { flex:1; padding:8px; border-radius:10px; border:none; font-size:12px; font-weight:600; cursor:pointer; }
.act-btn.primary   { background:#ef4444; color:#fff; }
.act-btn.secondary { background:rgba(255,255,255,.15); color:#fff; }
.info { width:320px; color:#fff; }
.info h2 { font-size:22px; font-weight:700; margin-bottom:4px;
           background:linear-gradient(135deg,#06b6d4,#3b82f6);
           -webkit-background-clip:text; -webkit-text-fill-color:transparent; }
.desc { font-size:13px; color:rgba(255,255,255,.5); margin-bottom:24px; line-height:1.5; }
.status-box { background:rgba(255,255,255,.07); border:1px solid rgba(255,255,255,.12);
              border-radius:12px; padding:16px; }
.status-row { display:flex; justify-content:space-between; padding:8px 0;
              border-bottom:1px solid rgba(255,255,255,.06); font-size:13px; }
.status-row:last-child { border:none; }
.status-key { color:rgba(255,255,255,.5); }
.status-val { font-weight:600; font-family:monospace; }
.green { color:#10b981; } .red { color:#ef4444; }
.alert-log { margin-top:16px; background:rgba(255,255,255,.05); border:1px solid rgba(255,255,255,.1);
             border-radius:12px; padding:12px; max-height:180px; overflow-y:auto; }
.log-title { font-size:11px; color:rgba(255,255,255,.4); text-transform:uppercase; letter-spacing:.06em; margin-bottom:8px; }
.log-entry { font-size:11px; color:rgba(255,255,255,.7); padding:4px 0;
             border-bottom:1px solid rgba(255,255,255,.05); font-family:monospace; }
.log-entry:last-child { border:none; }
.log-alert { color:#ef4444; }
</style>
</head>
<body>
<div class="scene">
  <div class="phone">
    <div class="phone-screen">
      <div class="notch"></div>
      <div class="lock-screen">
        <div class="status-bar">
          <span id="clock">--:--</span>
          <span>CDA Officer</span>
          <span>MOH</span>
        </div>
        <div class="time-display">
          <div class="time" id="bigClock">--:--</div>
          <div class="date" id="bigDate"></div>
        </div>
        <div class="notification" id="idleNotif">
          <div class="notif-header">
            <div class="notif-icon health">+</div>
            <div class="notif-app">CDA Alert System</div>
            <div class="notif-time" id="idleTime">--:--</div>
          </div>
          <div class="notif-title">Monitoring Active</div>
          <div class="notif-body">Connected to live case stream. Awaiting hospital notifications.</div>
        </div>
        <div class="notification alert-notif" id="alertNotif" style="display:none">
          <div class="notif-header">
            <div class="notif-icon danger">!</div>
            <div class="notif-app">DENGUE ALERT</div>
            <div class="notif-time" id="alertTime">--:--</div>
          </div>
          <div class="notif-title" id="alertTitle">High Risk Alert</div>
          <div class="notif-body"  id="alertBody">Alert triggered</div>
          <div class="chips">
            <div class="chip"><div class="chip-label">Model Score</div><div class="chip-value" id="chipScore">-</div></div>
            <div class="chip"><div class="chip-label">Alert Score</div><div class="chip-value" id="chipAlert">-</div></div>
            <div class="chip"><div class="chip-label">Cases</div>     <div class="chip-value" id="chipCases">-</div></div>
          </div>
          <div class="actions">
            <button class="act-btn primary">Dispatch Team</button>
            <button class="act-btn secondary" onclick="dismissAlert()">Dismiss</button>
          </div>
        </div>
      </div>
    </div>
  </div>

  <div class="info">
    <h2>CDA Officer Live Monitor</h2>
    <p class="desc">
      Passively receives alerts via SSE push.<br>
      Alerts appear the instant a hospital submits a confirmed case.<br>
      No action required from the officer.
    </p>
    <div class="status-box">
      <div class="status-row">
        <span class="status-key">SSE Connection</span>
        <span class="status-val" id="sseStatus">Connecting...</span>
      </div>
      <div class="status-row">
        <span class="status-key">Alerts Received</span>
        <span class="status-val" id="alertCount">0</span>
      </div>
      <div class="status-row">
        <span class="status-key">Last Alert Subzone</span>
        <span class="status-val" id="lastAlert">None</span>
      </div>
    </div>
    <div class="alert-log">
      <div class="log-title">Alert Log</div>
      <div id="logEntries">
        <div style="color:rgba(255,255,255,.3);font-size:11px">Waiting for alerts...</div>
      </div>
    </div>
  </div>
</div>

<script>
  let alertCount = 0;

  function tick() {
    const now = new Date();
    const h = String(now.getHours()).padStart(2,'0');
    const m = String(now.getMinutes()).padStart(2,'0');
    const s = String(now.getSeconds()).padStart(2,'0');
    document.getElementById('clock').textContent    = h+':'+m;
    document.getElementById('bigClock').textContent = h+':'+m;
    document.getElementById('idleTime').textContent = h+':'+m+':'+s;
    document.getElementById('bigDate').textContent  =
      now.toLocaleDateString('en-SG',{weekday:'long',day:'numeric',month:'long',year:'numeric'});
  }
  tick();
  setInterval(tick, 1000);

  function dismissAlert() {
    document.getElementById('alertNotif').style.display = 'none';
    document.getElementById('idleNotif').style.display  = 'block';
  }

  function showAlert(data) {
    alertCount++;
    document.getElementById('alertCount').textContent = alertCount;
    document.getElementById('lastAlert').textContent  = data.subzone_name;

    const now = new Date();
    const t   = String(now.getHours()).padStart(2,'0')+':'+String(now.getMinutes()).padStart(2,'0');
    document.getElementById('alertTime').textContent  = t;
    document.getElementById('alertTitle').textContent = 'High Risk: '+data.subzone_name;
    document.getElementById('alertBody').textContent  =
      'Alert score '+data.alert_score+' exceeded threshold. Source: '+data.source+'. Triage required.';
    document.getElementById('chipScore').textContent  = data.model_score;
    document.getElementById('chipAlert').textContent  = data.alert_score;
    document.getElementById('chipCases').textContent  = data.case_count;

    document.getElementById('idleNotif').style.display  = 'none';
    document.getElementById('alertNotif').style.display = 'block';

    // Add to log
    const log   = document.getElementById('logEntries');
    const entry = document.createElement('div');
    entry.className   = 'log-entry log-alert';
    entry.textContent = t+' ALERT '+data.subzone_name+' score='+data.alert_score;
    log.insertBefore(entry, log.firstChild);
    if (log.children.length > 20) log.removeChild(log.lastChild);

    // Auto-dismiss after 15s
    setTimeout(dismissAlert, 15000);
  }

  function connectSSE() {
    const el = document.getElementById('sseStatus');
    el.textContent = 'Connecting...'; el.className = 'status-val';

    const es = new EventSource('/alerts/stream');

    es.addEventListener('connected', () => {
      el.textContent = 'LIVE'; el.className = 'status-val green';
      const log   = document.getElementById('logEntries');
      const entry = document.createElement('div');
      const now   = new Date();
      entry.className   = 'log-entry';
      entry.textContent = String(now.getHours()).padStart(2,'0')+':'+
                          String(now.getMinutes()).padStart(2,'0')+
                          ' Connected to alert stream';
      log.insertBefore(entry, log.firstChild);
    });

    es.addEventListener('alert', e => {
      showAlert(JSON.parse(e.data));
    });

    es.addEventListener('heartbeat', () => {
      el.textContent = 'LIVE'; el.className = 'status-val green';
    });

    es.onerror = () => {
      el.textContent = 'Reconnecting...'; el.className = 'status-val red';
      es.close();
      setTimeout(connectSSE, 3000);
    };
  }

  connectSSE();
</script>
</body>
</html>"""
    return HTMLResponse(content=html)

'''

with open('c:/Users/Tran Nguyen Ngoc Nhu/Downloads/MLEgroup/inference/realtime_inference.py',
          encoding='utf-8') as f:
    lines = f.readlines()

# Find mobile and pest_control line numbers
mobile_start = next(i for i, l in enumerate(lines) if '@app.get("/mobile"' in l)
pest_start   = next(i for i, l in enumerate(lines) if '@app.get("/pest-control"' in l)

print(f"mobile_start={mobile_start+1}, pest_start={pest_start+1}")

out = lines[:mobile_start] + [NEW_MOBILE + '\n'] + lines[pest_start:]

with open('c:/Users/Tran Nguyen Ngoc Nhu/Downloads/MLEgroup/inference/realtime_inference.py',
          'w', encoding='utf-8') as f:
    f.writelines(out)

print(f"Done. Total lines: {len(out)}")
