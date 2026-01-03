document.addEventListener('DOMContentLoaded', () => {
  const video = document.getElementById('player');
  const canvas = document.getElementById('overlay');
  const tooltip = document.getElementById('tooltip');
  const tracksEndpoint = window.TRACKS_ENDPOINT;
  
  let allDetections = [];
  let activeTracks = new Map();
  let nextTrackId = 0;

  async function loadTracks() {
    try {
      console.log('Loading tracks from:', tracksEndpoint);
      
      // Update face count to show loading
      const faceCountText = document.getElementById('faceCountText');
      if (faceCountText) {
        faceCountText.textContent = 'Loading...';
      }
      
      const res = await fetch(tracksEndpoint);
      if (!res.ok) {
        throw new Error(`HTTP error! status: ${res.status}`);
      }
      
      allDetections = await res.json();
      console.log('Tracks loaded:', allDetections.length);
      console.log('Sample detection:', allDetections[0]);
      
      // Sort by time once
      allDetections.sort((a, b) => a.t - b.t);
      console.log(`Loaded and sorted ${allDetections.length} detections.`);
      
      // Update face count display
      updateFaceCount();
      
      // Start the drawing loop
      requestAnimationFrame(drawLoop);
      
    } catch (e) {
      console.error('Error loading tracks:', e);
      // Update face count to show error
      const faceCountText = document.getElementById('faceCountText');
      if (faceCountText) {
        faceCountText.textContent = 'Error loading';
      }
    }
  }

  function updateFaceCount() {
    const faceCountText = document.getElementById('faceCountText');
    if (faceCountText) {
      // Count unique faces (by reg_no)
      const uniqueFaces = new Set();
      allDetections.forEach(det => {
        if (det.reg_no && det.reg_no !== 'UNKNOWN') {
          uniqueFaces.add(det.reg_no);
        }
      });
      
      const totalDetections = allDetections.length;
      const uniqueCount = uniqueFaces.size;
      
      if (totalDetections > 0) {
        faceCountText.textContent = `${uniqueCount} unique, ${totalDetections} total`;
      } else {
        faceCountText.textContent = '0';
      }
    }
  }

  function resize() {
    const rect = video.getBoundingClientRect();
    // Match canvas to on-screen size of the video element
    canvas.width = rect.width; 
    canvas.height = rect.height;
    canvas.style.left = '0px';
    canvas.style.top = '0px';
    
    // Debug: Log canvas dimensions
    console.log(`Canvas resized to: ${canvas.width}x${canvas.height}, video rect: ${rect.width}x${rect.height}`);
  }

  function getCenter(box) {
    return { x: box.x + box.w / 2, y: box.y + box.h / 2 };
  }

  function dist(p1, p2) {
    return Math.sqrt(Math.pow(p1.x - p2.x, 2) + Math.pow(p1.y - p2.y, 2));
  }

  function updateTracks() {
    const now = video.currentTime || 0;
    // Wider tolerance (1.5 seconds) to keep boxes visible longer and reduce flickering
    const currentDetections = allDetections.filter(d => Math.abs(d.t - now) < 1.5);
    const matchedTrackIds = new Set();
    
    // Debug: Log detection matching
    if (currentDetections.length > 0) {
      console.log(`Found ${currentDetections.length} detections near time ${now}`);
    }

    // Match new detections to existing tracks
    for (const det of currentDetections) {
      let bestMatch = { id: null, distance: 0.3 }; // Increased max distance for better matching

      for (const [id, track] of activeTracks.entries()) {
        const d = dist(getCenter(det), getCenter(track));
        if (d < bestMatch.distance) {
          bestMatch = { id, distance: d };
        }
      }

      if (bestMatch.id !== null) {
        // Update existing track
        const track = activeTracks.get(bestMatch.id);
        Object.assign(track, det, { lastSeen: now });
        matchedTrackIds.add(bestMatch.id);
      } else {
        // Create new track
        const newId = nextTrackId++;
        activeTracks.set(newId, { ...det, id: newId, lastSeen: now });
      }
    }

    // Remove old tracks that haven't been seen for a while
    for (const [id, track] of activeTracks.entries()) {
      if (now - track.lastSeen > 2.0) {
        activeTracks.delete(id);
      }
    }
  }

  function drawLoop() {
    resize();
    updateTracks();
    updateTimeDisplay();
    
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Debug: Log active tracks count
    if (activeTracks.size > 0) {
      console.log(`Drawing ${activeTracks.size} active tracks at time ${video.currentTime}`);
    }

    ctx.globalAlpha = 1.0; // Ensure everything is fully opaque
    for (const track of activeTracks.values()) {
      const norm = (val, base) => (val <= 1 ? val * base : val);
      const x = norm(track.x, canvas.width);
      const y = norm(track.y, canvas.height);
      const w = norm(track.w, canvas.width);
      const h = norm(track.h, canvas.height);

      // Draw green box around face
      ctx.strokeStyle = '#28a745';
      ctx.lineWidth = 3;
      ctx.strokeRect(x, y, w, h);

      // Draw label background
      const labelText = (track.reg_no && track.reg_no !== 'UNKNOWN') ? track.reg_no : 'Unknown';
      ctx.font = 'bold 16px Arial';
      const textMetrics = ctx.measureText(labelText);
      const labelW = textMetrics.width + 16;
      const labelH = 30;

      // Enhanced label background with better visibility
      ctx.fillStyle = 'rgba(40, 167, 69, 0.9)'; // Green background
      ctx.fillRect(x, y - labelH - 5, labelW, labelH);
      
      // White text for better contrast
      ctx.fillStyle = '#ffffff';
      ctx.fillText(labelText, x + 8, y - 15);
      
      // Add small indicator dot for clickable faces
      if (track.reg_no && track.reg_no !== 'UNKNOWN') {
        ctx.fillStyle = '#ffffff';
        ctx.beginPath();
        ctx.arc(x + w - 10, y + 10, 6, 0, 2 * Math.PI);
        ctx.fill();
        ctx.strokeStyle = '#28a745';
        ctx.lineWidth = 2;
        ctx.stroke();
      }
    }

    requestAnimationFrame(drawLoop);
  }

  function updateTimeDisplay() {
    const timeInfoText = document.getElementById('timeInfoText');
    if (timeInfoText && video) {
      const currentTime = video.currentTime;
      const minutes = Math.floor(currentTime / 60);
      const seconds = Math.floor(currentTime % 60);
      timeInfoText.textContent = `${minutes.toString().padStart(2, '0')}:${seconds.toString().padStart(2, '0')}`;
    }
    
    // Update active face count
    const faceCountText = document.getElementById('faceCountText');
    if (faceCountText) {
      const currentActiveFaces = activeTracks.size;
      if (currentActiveFaces > 0) {
        faceCountText.textContent = `${currentActiveFaces} visible now`;
      }
    }
  }

  function onMove(e) {
    const rect = canvas.getBoundingClientRect();
    const mx = (e.clientX - rect.left) / canvas.width;
    const my = (e.clientY - rect.top) / canvas.height;
    let hit = null;

    for (const track of activeTracks.values()) {
      if (mx >= track.x && mx <= track.x + track.w && my >= track.y && my <= track.y + track.h) {
        hit = track;
        break;
      }
    }

    if (hit) {
      tooltip.style.display = 'block';
      tooltip.style.left = `${e.pageX + 12}px`;
      tooltip.style.top = `${e.pageY + 12}px`;
      const s = (window.STUDENT_MAP || {})[hit.reg_no];
      if (hit.reg_no && hit.reg_no !== 'UNKNOWN') {
        tooltip.innerHTML = `
          <div style="border-bottom: 1px solid #28a745; padding-bottom: 5px; margin-bottom: 5px;">
            <b style="color: #28a745; font-size: 16px;">${s?.name || 'Identified Student'}</b>
          </div>
          <div style="margin-bottom: 8px;">
            <strong style="color: #28a745;">Registration:</strong> 
            <a href="/student/${hit.reg_no}" target="_blank" style="color: #007bff; text-decoration: underline; font-weight: bold; font-size: 16px;">${hit.reg_no}</a>
            <span style="color: #6c757d; font-size: 12px;"> (Click to view profile)</span>
          </div>
          <div><strong>Department:</strong> ${s?.dept || 'N/A'}</div>
          <div><strong>Room:</strong> ${s?.room_no || 'N/A'}</div>
          <div><strong>Father's Name:</strong> ${s?.father_name || 'N/A'}</div>
          <div><strong>Father's Phone:</strong> ${s?.father_phone || 'N/A'}</div>`;
      } else {
        tooltip.innerHTML = `
          <div style="border-bottom: 1px solid #28a745; padding-bottom: 5px; margin-bottom: 5px;">
            <b style="color: #28a745; font-size: 16px;">Unknown</b>
          </div>
          <div><strong>Status:</strong> Unidentified Person</div>`;
      }
    } else {
      tooltip.style.display = 'none';
    }
  }

  function onCanvasClick(e) {
    const rect = canvas.getBoundingClientRect();
    const mx = (e.clientX - rect.left) / canvas.width;
    const my = (e.clientY - rect.top) / canvas.height;

    for (const track of activeTracks.values()) {
      if (mx >= track.x && mx <= track.x + track.w && my >= track.y && my <= track.y + track.h) {
        if (track.reg_no && track.reg_no !== 'UNKNOWN') {
          window.open(`/student/${track.reg_no}`, '_blank');
        }
        break;
      }
    }
  }

  // Event Listeners
  window.addEventListener('resize', resize);
  video.addEventListener('loadedmetadata', () => {
    resize();
    // Load tracks after video metadata is available
    loadTracks();
  });
  video.addEventListener('play', () => requestAnimationFrame(drawLoop));
  video.addEventListener('pause', () => { /* Redraw one last time when paused */ drawLoop(); });
  video.addEventListener('seeked', () => {
    activeTracks.clear(); // Clear tracks on seek
    updateTracks(); // Do one update to get current detections
    if (!video.paused) {
        requestAnimationFrame(drawLoop);
    } else {
        drawLoop(); // Redraw once if paused
    }
  });
  canvas.addEventListener('mousemove', onMove);
  canvas.addEventListener('mouseleave', () => tooltip.style.display = 'none');
  canvas.addEventListener('click', onCanvasClick);

  // Custom Controls
  const playBtn = document.getElementById('btnPlay');
  const pauseBtn = document.getElementById('btnPause');
  const stopBtn = document.getElementById('btnStop');
  const replayBtn = document.getElementById('btnReplay');
  const refreshBtn = document.getElementById('btnRefresh');
  const loopChk = document.getElementById('chkLoop');
  const muteChk = document.getElementById('chkMute');

  if (playBtn) playBtn.addEventListener('click', () => video.play());
  if (pauseBtn) pauseBtn.addEventListener('click', () => video.pause());
  if (stopBtn) stopBtn.addEventListener('click', () => { video.pause(); video.currentTime = 0; });
  if (replayBtn) replayBtn.addEventListener('click', () => { video.currentTime = 0; video.play(); });
  if (refreshBtn) refreshBtn.addEventListener('click', () => {
    console.log('Refreshing face detection data...');
    activeTracks.clear();
    loadTracks();
  });
  if (loopChk) loopChk.addEventListener('change', () => { video.loop = loopChk.checked; });
  if (muteChk) muteChk.addEventListener('change', () => { video.muted = muteChk.checked; });

  // Load tracks immediately if video is already loaded, otherwise wait for loadedmetadata
  if (video.readyState >= 1) {
    loadTracks();
  }
});

// Function to fetch student details
function fetchStudentDetails() {
    const regNo = document.getElementById('reg_no').value;
    if (!regNo.match(/^99\d{9}$/)) {
        alert('Please enter a valid registration number (11 digits starting with 99)');
        return;
    }

    fetch(`/get_student_details/${regNo}`)
        .then(response => response.json())
        .then(data => {
            if (data.error) {
                alert(data.error);
                return;
            }
            
            // Update form fields with student details
            document.getElementById('form_reg_no').value = regNo;
            document.getElementById('full_name').value = data.full_name || '';
            document.getElementById('department').value = data.department || '';
            document.getElementById('room_number').value = data.room_number || '';
            document.getElementById('fathers_name').value = data.fathers_name || '';
            document.getElementById('fathers_phone').value = data.fathers_phone || '';
            
            // Show the form
            document.getElementById('studentForm').style.display = 'block';
        })
        .catch(error => {
            console.error('Error:', error);
            alert('Error fetching student details. Please try again.');
        });
}
