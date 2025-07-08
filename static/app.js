document.addEventListener('DOMContentLoaded', function() {
    const uploadForm = document.getElementById('uploadForm');
    const fileInput = document.getElementById('fileInput');
    const uploadStatus = document.getElementById('uploadStatus');
    const chatForm = document.getElementById('chatForm');
    const chatInput = document.getElementById('chatInput');
    const chatBox = document.getElementById('chatBox');
    const micBtn = document.getElementById('micBtn');
    const voiceModeBtn = document.getElementById('voiceModeBtn');
    let recognizing = false;
    let recognition;
    let voiceMode = false;

    uploadForm.addEventListener('submit', function(e) {
        e.preventDefault();
        const file = fileInput.files[0];
        if (!file) return;
        const formData = new FormData();
        formData.append('file', file);
        uploadStatus.textContent = 'Uploading...';
        fetch('/upload', {
            method: 'POST',
            body: formData
        })
        .then(res => res.json())
        .then(data => {
            if (data.error) {
                uploadStatus.textContent = data.error;
                uploadStatus.style.color = '#ff0077';
            } else if (data.transcription) {
                uploadStatus.textContent = 'Audio transcribed!';
                appendMessage('You (audio)', '[Audio uploaded]');
                appendMessage('Bot', data.transcription);
            } else {
                uploadStatus.textContent = data.message || 'File uploaded!';
                uploadStatus.style.color = '#1561e8';
            }
        })
        .catch(() => {
            uploadStatus.textContent = 'Upload failed.';
            uploadStatus.style.color = '#ff0077';
        });
    });

    chatForm.addEventListener('submit', function(e) {
        e.preventDefault();
        const question = chatInput.value.trim();
        if (!question) return;
        appendMessage('You', question);
        chatInput.value = '';
        fetch('/ask', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ question })
        })
        .then(res => res.json())
        .then(data => {
            if (data.error) {
                appendMessage('Bot', data.error);
            } else {
                appendMessage('Bot', data.answer);
            }
        })
        .catch(() => {
            appendMessage('Bot', 'Error getting answer.');
        });
    });

    if ('webkitSpeechRecognition' in window || 'SpeechRecognition' in window) {
        const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
        recognition = new SpeechRecognition();
        recognition.lang = 'en-US';
        recognition.interimResults = false;
        recognition.maxAlternatives = 1;
        recognition.continuous = true;

        recognition.onstart = function() {
            recognizing = true;
            if (!voiceMode) micBtn.style.background = '#ffe600';
        };
        recognition.onend = function() {
            recognizing = false;
            if (!voiceMode) micBtn.style.background = '';
            if (voiceMode) recognition.start();
        };
        recognition.onerror = function(event) {
            recognizing = false;
            if (!voiceMode) micBtn.style.background = '';
            if (!voiceMode) alert('Speech recognition error: ' + event.error);
        };
        recognition.onresult = function(event) {
            if (event.results && event.results.length > 0) {
                const lastResult = event.results[event.results.length - 1][0].transcript;
                if (voiceMode) {
                    appendMessage('You', lastResult);
                    fetch('/ask', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ question: lastResult })
                    })
                    .then(res => res.json())
                    .then(data => {
                        if (data.error) {
                            appendMessage('Bot', data.error);
                        } else {
                            appendMessage('Bot', data.answer);
                        }
                    })
                    .catch(() => {
                        appendMessage('Bot', 'Error getting answer.');
                    });
                } else {
                    chatInput.value = lastResult;
                    chatInput.focus();
                }
            }
        };

        micBtn.addEventListener('click', function(e) {
            e.preventDefault();
            if (recognizing) {
                recognition.stop();
                return;
            }
            recognition.start();
        });

        voiceModeBtn.addEventListener('click', function() {
            voiceMode = !voiceMode;
            if (voiceMode) {
                document.body.classList.add('voice-mode-active');
                voiceModeBtn.textContent = '🎤 Voice Mode: On';
                voiceModeBtn.setAttribute('aria-pressed', 'true');
                chatInput.placeholder = 'Speak now...';
                recognition.start();
            } else {
                document.body.classList.remove('voice-mode-active');
                voiceModeBtn.textContent = '🎤 Voice Mode: Off';
                voiceModeBtn.setAttribute('aria-pressed', 'false');
                chatInput.placeholder = 'Type your question...';
                if (recognizing) recognition.stop();
            }
        });
    } else {
        micBtn.style.display = 'none';
        voiceModeBtn.style.display = 'none';
    }

    function appendMessage(author, text) {
        const msg = document.createElement('div');
        msg.className = 'chat-message';
        msg.innerHTML = `<strong>${author}:</strong> ${text}`;
        chatBox.appendChild(msg);
        chatBox.scrollTop = chatBox.scrollHeight;
    }
}); 