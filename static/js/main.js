// --- ระบบนำทาง (Navigation) ---
function showPage(page) {
    document.querySelectorAll('.page-content').forEach(p => p.classList.remove('active-page'));
    document.querySelectorAll('.tab-btn').forEach(l => l.classList.remove('active'));
    document.getElementById(page + '-page').classList.add('active-page');
    document.getElementById('btn-nav-' + page).classList.add('active');
    
    // โหลดสถิติอัตโนมัติทั้งตอนเข้าหน้า 'เก็บข้อมูล' และ 'Admin'
    if (page === 'collect' || page === 'admin') loadStats();
}

// --- ตั้งค่ากระดานวาดรูป (Canvas) พร้อมตัวเลื่อนขนาดแปรง ---
const setupCanvas = (id, sliderId, valueDisplayId) => {
    const canvas = document.getElementById(id);
    const slider = document.getElementById(sliderId);
    const display = document.getElementById(valueDisplayId);
    const ctx = canvas.getContext('2d');
    let isDrawing = false;
    
    ctx.lineWidth = slider.value;
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
    ctx.strokeStyle = '#1e293b';
    ctx.fillStyle = "white";
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    // อัปเดตขนาดเวลาเลื่อน Slider
    slider.oninput = function() {
        ctx.lineWidth = this.value;
        display.innerText = this.value;
    };

    const getPos = (e) => {
        const rect = canvas.getBoundingClientRect();
        const cx = e.touches ? e.touches[0].clientX : e.clientX;
        const cy = e.touches ? e.touches[0].clientY : e.clientY;
        const scaleX = canvas.width / rect.width;
        const scaleY = canvas.height / rect.height;
        return { x: (cx - rect.left) * scaleX, y: (cy - rect.top) * scaleY };
    };

    const start = (e) => { isDrawing = true; const p = getPos(e); ctx.beginPath(); ctx.moveTo(p.x, p.y); };
    const draw = (e) => {
        if (!isDrawing) return;
        e.preventDefault();
        const p = getPos(e);
        ctx.lineTo(p.x, p.y);
        ctx.stroke();
    };
    const stop = () => { if (isDrawing) { ctx.closePath(); isDrawing = false; } };

    canvas.addEventListener('mousedown', start);
    canvas.addEventListener('mousemove', draw);
    canvas.addEventListener('mouseup', stop);
    canvas.addEventListener('mouseout', stop); // กันลากเมาส์หลุดขอบ
    canvas.addEventListener('touchstart', start, {passive: false});
    canvas.addEventListener('touchmove', draw, {passive: false});
    canvas.addEventListener('touchend', stop);

    return { canvas, ctx };
};

// เริ่มต้นกระดานทั้ง 2 หน้า
const collectObj = setupCanvas('canvas-collect', 'brush-size-collect', 'size-val-collect');
const predictObj = setupCanvas('canvas-predict', 'brush-size-predict', 'size-val-predict');

function clearCanvas(type) {
    const obj = type === 'collect' ? collectObj : predictObj;
    obj.ctx.fillStyle = "white"; 
    obj.ctx.fillRect(0, 0, obj.canvas.width, obj.canvas.height);
    document.getElementById('status-' + type).innerText = "";
    if(type === 'predict') document.getElementById('result-container').classList.add('hidden');
}

// --- API หน้า Collect ---
async function saveImage() {
    const label = document.getElementById('class-select').value;
    const dataURL = collectObj.canvas.toDataURL('image/png');
    const msg = document.getElementById('status-collect');
    
    msg.innerText = "⏳ กำลังบันทึก..."; msg.style.color = "#64748b";
    try {
        const res = await fetch('/upload', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ image: dataURL, label: label })
        });
        if (res.ok) { 
            msg.innerText = "✅ บันทึกข้อมูลสำเร็จ!"; msg.style.color = "#10b981";
            setTimeout(() => clearCanvas('collect'), 800);
            loadStats(); // อัปเดตตัวเลขสถิติให้เห็นทันทีที่เซฟ
        } else {
            msg.innerText = "❌ เกิดข้อผิดพลาดจากเซิร์ฟเวอร์"; msg.style.color = "#dc2626";
        }
    } catch (e) { msg.innerText = "❌ ไม่สามารถเชื่อมต่อระบบได้"; msg.style.color = "#dc2626"; }
}

// --- API หน้า Predict ---
async function predictImage() {
    const dataURL = predictObj.canvas.toDataURL('image/png');
    const resText = document.getElementById('prediction-text');
    const resBox = document.getElementById('result-container');
    const msg = document.getElementById('status-predict');
    
    msg.innerText = "🧠 AI กำลังวิเคราะห์..."; msg.style.color = "#64748b";
    resBox.classList.add('hidden');

    try {
        const res = await fetch('/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ image: dataURL })
        });
        const data = await res.json();
        
        if (res.ok) {
            msg.innerText = "";
            resBox.classList.remove('hidden');
            resText.innerText = data.prediction;
        } else {
            msg.innerText = "❌ " + (data.error || "เกิดข้อผิดพลาด"); msg.style.color = "#dc2626";
        }
    } catch (e) { msg.innerText = "❌ เชื่อมต่อ AI ไม่สำเร็จ"; msg.style.color = "#dc2626"; }
}

// --- API หน้า Admin ---
function updateFileName() {
    const file = document.getElementById('model-file').files[0];
    document.getElementById('file-name-display').innerText = file ? file.name : "คลิกเพื่อเลือกไฟล์ .pkl";
}

async function uploadModel() {
    const fileInput = document.getElementById('model-file');
    const password = document.getElementById('admin-password-global').value;
    const msg = document.getElementById('status-admin');

    if (!fileInput.files[0] || !password) {
        msg.innerText = "⚠️ เลือกไฟล์โมเดลและใส่รหัสผ่านก่อนครับ"; msg.style.color = "#dc2626"; return;
    }
    const formData = new FormData();
    formData.append('model_file', fileInput.files[0]);
    formData.append('admin_pass', password);
    msg.innerText = "⏳ กำลังอัปโหลด..."; msg.style.color = "#2563eb";
    
    try {
        const res = await fetch('/admin_upload_model', { method: 'POST', body: formData });
        const data = await res.json();
        if (res.ok) {
            msg.innerText = "✅ " + data.message; msg.style.color = "#10b981";
            document.getElementById('file-name-display').innerText = "คลิกเพื่อเลือกไฟล์ .pkl";
            fileInput.value = "";
        } else { msg.innerText = "❌ " + data.error; msg.style.color = "#dc2626"; }
    } catch (e) { msg.innerText = "❌ ตัดการเชื่อมต่อ"; msg.style.color = "#dc2626"; }
}

async function downloadDatasets() {
    const password = document.getElementById('admin-password-global').value;
    const msg = document.getElementById('status-admin');
    if (!password) { msg.innerText = "⚠️ ใส่รหัสผ่านก่อนครับ"; msg.style.color = "#dc2626"; return; }
    msg.innerText = "⏳ กำลังเตรียมไฟล์ Zip..."; msg.style.color = "#2563eb";

    try {
        const formData = new FormData();
        formData.append('admin_pass', password);
        const res = await fetch('/admin_download_datasets', { method: 'POST', body: formData });
        if (res.ok) {
            const blob = await res.blob();
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url; a.download = "datasets_backup.zip";
            document.body.appendChild(a); a.click(); a.remove();
            msg.innerText = "✅ ดาวน์โหลดสำเร็จ!"; msg.style.color = "#10b981";
        } else {
            const data = await res.json();
            msg.innerText = "❌ " + (data.error || "รหัสผ่านผิด"); msg.style.color = "#dc2626";
        }
    } catch (e) { msg.innerText = "❌ ข้อผิดพลาดเครือข่าย"; msg.style.color = "#dc2626"; }
}

async function loadStats() {
    const statsList = document.getElementById('stats-list');
    try {
        const res = await fetch('/admin_stats');
        const data = await res.json();
        if (Object.keys(data).length === 0) {
            statsList.innerHTML = '<p class="text-muted" style="grid-column: span 2; text-align:center;">ยังไม่มีรูป</p>';
            return;
        }
        statsList.innerHTML = ''; 
        Object.keys(data).sort().forEach(label => {
            const item = document.createElement('div');
            item.className = 'stat-item';
            item.innerHTML = `<div class="stat-label">เลข ${label}</div><div class="stat-count">${data[label]} ภาพ</div>`;
            statsList.appendChild(item);
        });
    } catch (e) { statsList.innerHTML = '<p style="color:#dc2626; grid-column:span 2; text-align:center;">❌ โหลดสถิติพัง</p>'; }
}