from flask import Flask, render_template_string, jsonify, Response
import threading
import cv2
import time

app = Flask(__name__)

class DashboardData:
    raw_frame = None
    ai_frame = None
    logs = []
    stats = {
        "Match Count": 0,
        "Reward": 0.0,
        "Keys Pressed": "None",
        "Attacking": "False",
        "Super": "False",
        "Gadget": "False",
        "State": "IDLE"
    }
    train_stats = {
        "Learning Rate": 0.0,
        "Entropy Loss": 0.0,
        "Value Loss": 0.0,
        "Clip Fraction": 0.0
    }
    
    @classmethod
    def add_log(cls, msg):
        time_str = time.strftime("%H:%M:%S")
        cls.logs.append(f"[{time_str}] {msg}")
        if len(cls.logs) > 50:
            cls.logs.pop(0)

@app.route('/')
def index():
    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>🤖 Brawl Stars RL Dashboard</title>
        <style>
            body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background: #121212; color: #e0e0e0; margin: 0; padding: 20px; }
            h1 { color: #bb86fc; text-align: center; margin-bottom: 30px; font-weight: 300; }
            .container { display: flex; flex-direction: column; gap: 20px; max-width: 1400px; margin: 0 auto; }
            .row { display: flex; gap: 20px; flex-wrap: wrap; justify-content: center; }
            .card { background: #1e1e1e; padding: 20px; border-radius: 12px; flex: 1; min-width: 300px; box-shadow: 0 8px 16px rgba(0,0,0,0.5); }
            h2 { margin-top: 0; color: #03dac6; font-size: 1.2rem; border-bottom: 2px solid #333; padding-bottom: 10px; }
            .img-container { display: flex; justify-content: center; align-items: center; background: #000; border-radius: 8px; overflow: hidden; padding: 10px; min-height: 300px; }
            img { max-width: 100%; max-height: 500px; object-fit: contain; }
            .ai-vision { image-rendering: pixelated; } /* Keeps the 84x84 sharp */
            .logs { height: 350px; overflow-y: auto; background: #0a0a0a; padding: 15px; font-family: 'Courier New', Courier, monospace; font-size: 13px; border-radius: 8px; color: #4ade80; border: 1px solid #333; }
            .logs div { margin-bottom: 4px; }
            .stat-box { display: flex; justify-content: space-between; padding: 12px 0; border-bottom: 1px solid #333; font-size: 1.1rem; }
            .stat-box:last-child { border-bottom: none; }
            .stat-value { font-weight: bold; color: #fff; }
            .true-val { color: #f87171; } /* Red for True/Active */
            .false-val { color: #9ca3af; } /* Gray for False/Inactive */
        </style>
    </head>
    <body>
        <h1>🤖 Brawl Stars RL Dashboard</h1>
        <div class="container">
            <div class="row">
                <div class="card" style="flex: 2;">
                    <h2>BlueStacks Raw View</h2>
                    <div class="img-container">
                        <img id="raw-img" src="/raw_feed" alt="Waiting for Raw Feed...">
                    </div>
                </div>
                <div class="card" style="flex: 1;">
                    <h2>AI Vision (84x84 CNN Input)</h2>
                    <div class="img-container">
                        <img class="ai-vision" id="ai-img" src="/ai_feed" alt="Waiting for AI Feed...">
                    </div>
                </div>
            </div>
            <div class="row">
                <div class="card" style="flex: 1;">
                    <h2>AI Parameters & Stats</h2>
                    <div id="stats-container"></div>
                </div>
                <div class="card" style="flex: 2;">
                    <h2>Live Training Logs</h2>
                    <div class="logs" id="logs-container"></div>
                </div>
            </div>
        </div>

        <script>
            function updateData() {
                fetch('/data').then(response => response.json()).then(data => {
                    // Update stats
                    let statsHtml = '';
                    for (const [key, value] of Object.entries(data.stats)) {
                        let valClass = "stat-value";
                        if (value === "True") valClass += " true-val";
                        if (value === "False") valClass += " false-val";
                        statsHtml += `<div class="stat-box"><span>${key}:</span> <span class="${valClass}">${value}</span></div>`;
                    }
                    document.getElementById('stats-container').innerHTML = statsHtml;

                    // Update logs
                    let logsContainer = document.getElementById('logs-container');
                    let isScrolledToBottom = logsContainer.scrollHeight - logsContainer.clientHeight <= logsContainer.scrollTop + 20;
                    
                    logsContainer.innerHTML = data.logs.map(log => `<div>${log}</div>`).join('');
                    
                    if (isScrolledToBottom) {
                        logsContainer.scrollTop = logsContainer.scrollHeight;
                    }
                }).catch(err => console.error("Error fetching data:", err));
            }
            
            // Reload images periodically if they break (e.g., server restart)
            function checkImage(imgId, feedUrl) {
                let img = document.getElementById(imgId);
                if (!img.complete || img.naturalWidth === 0) {
                    img.src = feedUrl + '?' + new Date().getTime();
                }
            }

            setInterval(updateData, 200); // 5 FPS updates for UI
            setInterval(() => {
                checkImage('raw-img', '/raw_feed');
                checkImage('ai-img', '/ai_feed');
            }, 5000);
        </script>
    </body>
    </html>
    """
    return render_template_string(html)

def gen_frames(frame_type):
    while True:
        frame = DashboardData.raw_frame if frame_type == 'raw' else DashboardData.ai_frame
        if frame is not None:
            ret, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        else:
            time.sleep(0.1)
        time.sleep(0.05) # Cap at ~20 FPS for stream

@app.route('/raw_feed')
def raw_feed():
    return Response(gen_frames('raw'), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/ai_feed')
def ai_feed():
    return Response(gen_frames('ai'), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/data')
def data():
    return jsonify({
        "stats": DashboardData.stats,
        "logs": DashboardData.logs
    })

def start_dashboard():
    import logging
    log = logging.getLogger('werkzeug')
    log.setLevel(logging.ERROR) # Hide flask request logs from console
    print("==================================================")
    print("🌐 WEB DASHBOARD STARTED AT: http://localhost:5000")
    print("==================================================")
    threading.Thread(target=lambda: app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False), daemon=True).start()
