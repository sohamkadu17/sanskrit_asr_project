document.addEventListener('DOMContentLoaded', () => {
    const recordBtn = document.getElementById('record-btn');
    const recordStatus = document.getElementById('record-status');
    const dropZone = document.getElementById('drop-zone');
    const fileInput = document.getElementById('file-input');
    const loader = document.getElementById('loader');
    const transcriptionBox = document.getElementById('transcription-box');

    let mediaRecorder;
    let audioChunks = [];
    let isRecording = false;

    const shlokaSelect = document.getElementById('shloka-select');
    const practiceGroup = document.getElementById('practice-group');
    const practiceTarget = document.getElementById('practice-target');
    const practiceSanskrit = document.getElementById('practice-sanskrit');
    
    let currentShlokas = {};

    // --- FETCH SHLOKAS ON LOAD ---
    async function fetchShlokas() {
        try {
            const res = await fetch('/api/shlokas');
            const shlokas = await res.json();
            
            shlokas.forEach(s => {
                currentShlokas[s.id] = s;
                const opt = document.createElement('option');
                opt.value = s.id;
                // Add a small snippet of the meaning or source as the label
                opt.textContent = `${s.id} - ${s.sanskrit.split('।')[0].substring(0, 30)}...`;
                practiceGroup.appendChild(opt);
            });
        } catch(e) {
            console.error("Failed to fetch shlokas");
        }
    }
    fetchShlokas();

    // --- HANDLE MODE CHANGE ---
    shlokaSelect.addEventListener('change', (e) => {
        const val = e.target.value;
        if (val === 'auto') {
            practiceTarget.classList.add('hidden');
        } else {
            practiceTarget.classList.remove('hidden');
            const shloka = currentShlokas[val];
            practiceSanskrit.innerHTML = shloka.sanskrit.replace(/\n/g, '<br>');
        }
        transcriptionBox.innerHTML = '<p class="placeholder-text">Your Sanskrit text will appear here.</p>';
    });

    // --- FILE UPLOAD LOGIC ---
    dropZone.addEventListener('click', () => fileInput.click());

    dropZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropZone.classList.add('dragover');
    });

    dropZone.addEventListener('dragleave', () => {
        dropZone.classList.remove('dragover');
    });

    dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropZone.classList.remove('dragover');
        if (e.dataTransfer.files.length > 0) {
            handleFile(e.dataTransfer.files[0]);
        }
    });

    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            handleFile(e.target.files[0]);
        }
    });

    function handleFile(file) {
        if (!file.type.startsWith('audio/')) {
            alert('Please upload an audio file.');
            return;
        }
        uploadAudio(file);
    }

    // --- RECORDING LOGIC ---
    recordBtn.addEventListener('click', async () => {
        if (!isRecording) {
            startRecording();
        } else {
            stopRecording();
        }
    });

    async function startRecording() {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            mediaRecorder = new MediaRecorder(stream);
            audioChunks = [];

            mediaRecorder.ondataavailable = event => {
                audioChunks.push(event.data);
            };

            mediaRecorder.onstop = () => {
                const audioBlob = new Blob(audioChunks, { type: 'audio/webm' });
                const file = new File([audioBlob], "recording.webm", { type: 'audio/webm' });
                uploadAudio(file);
                
                // Stop all microphone tracks
                stream.getTracks().forEach(track => track.stop());
            };

            mediaRecorder.start();
            isRecording = true;
            recordBtn.classList.add('recording');
            recordStatus.textContent = "Recording... Click to Stop";
            recordBtn.innerHTML = '<i class="fas fa-stop"></i>';
        } catch (err) {
            console.error("Error accessing mic:", err);
            alert("Could not access microphone. Please ensure permissions are granted.");
        }
    }

    function stopRecording() {
        mediaRecorder.stop();
        isRecording = false;
        recordBtn.classList.remove('recording');
        recordStatus.textContent = "Click to Record";
        recordBtn.innerHTML = '<i class="fas fa-microphone"></i>';
    }

    // --- UPLOAD TO BACKEND ---
    async function uploadAudio(file) {
        // Show loader, clear box
        loader.classList.remove('hidden');
        transcriptionBox.innerHTML = '';

        const formData = new FormData();
        formData.append('audio', file);
        
        const mode = shlokaSelect.value;
        const endpoint = mode === 'auto' ? '/api/identify_shloka' : '/api/practice_shloka';
        
        if (mode !== 'auto') {
            formData.append('target_id', mode);
        }

        try {
            const response = await fetch(endpoint, {
                method: 'POST',
                body: formData
            });

            const data = await response.json();
            
            loader.classList.add('hidden');
            
            if (response.ok) {
                let html = `<div style="background: rgba(0,0,0,0.4); padding: 15px; border-radius: 8px; margin-bottom: 20px;">
                                <p style="color: #a0a0b0; font-size: 0.9rem; margin-bottom: 5px;">Detected Speech:</p>
                                <p class="transcription-result" style="font-family: 'Yantramanav', sans-serif; font-size: 1.4rem;">${data.transcription}</p>
                            </div>`;
                
                const m = data.match || data.target;
                const pronunciation = (m && m.pronunciation) ? m.pronunciation : data.pronunciation;
                
                if (m) {
                    html += `
                        <div class="shloka-card" style="margin-top: 20px; text-align: left; background: rgba(0,0,0,0.3); padding: 20px; border-radius: 12px; border: 1px solid rgba(255,255,255,0.1);">
                            <h3 style="color: #F27121; margin-bottom: 10px;">${m.source} — ${m.chapter > 0 ? `Chapter ${m.chapter}, Verse ${m.verse}` : 'Prayer'}</h3>
                            <p style="font-size: 1.4rem; color: #fff; margin-bottom: 10px; font-family: 'Yantramanav', sans-serif;">${m.sanskrit.replace(/\n/g, '<br>')}</p>
                            <p style="font-family: 'Inter', sans-serif; font-size: 1rem; color: #a0a0b0; font-style: italic; margin-bottom: 15px;">${m.transliteration.replace(/\n/g, '<br>')}</p>
                            <p style="font-family: 'Inter', sans-serif; font-size: 1rem; color: #ddd; line-height: 1.5;"><strong>Meaning:</strong> ${m.meaning}</p>
                            
                            <div style="margin-top: 25px; padding-top: 15px; border-top: 1px solid rgba(255,255,255,0.1);">
                                <h4 style="color: #fff; margin-bottom: 10px;">Pronunciation Feedback:</h4>
                                <div style="font-size: 1.6rem; line-height: 1.6; font-family: 'Yantramanav', sans-serif; margin-bottom: 15px; background: rgba(0,0,0,0.5); padding: 15px; border-radius: 8px;">
                                    ${pronunciation ? pronunciation.html : '<span style="color: #a0a0b0;">No pronunciation feedback available.</span>'}
                                </div>
                                ${pronunciation ? `
                                <div style="font-size: 1.1rem; font-family: 'Inter', sans-serif; font-weight: 600;">
                                    <span style="color: ${pronunciation.score > 85 ? '#4cd137' : (pronunciation.score > 50 ? '#fbc531' : '#e84118')};">
                                        Score: ${pronunciation.score}%
                                    </span>
                                    <span style="color: #a0a0b0; font-size: 0.85rem; font-weight: normal; margin-left: 10px;">(Hover over red/yellow text to see actual sounds)</span>
                                </div>
                                ` : ''}
                            </div>
                        </div>
                    `;
                } else {
                    html += `<p style="color: #a0a0b0; margin-top: 15px; font-size: 1rem; font-family: 'Inter', sans-serif;">No matching shloka found in the database.</p>`;
                }
                
                transcriptionBox.innerHTML = html;
            } else {
                transcriptionBox.innerHTML = `<p style="color: #ff4757;">Error: ${data.error}</p>`;
            }
        } catch (error) {
            console.error('Upload error:', error);
            loader.classList.add('hidden');
            transcriptionBox.innerHTML = `<p style="color: #ff4757;">Connection Error. Is the backend running?</p>`;
        }
    }
});
