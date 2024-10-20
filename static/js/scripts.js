// scripts.js
let lastData = null; // Storing the last SSE data

// Handling forms and buttons after the document is loaded
document.addEventListener('DOMContentLoaded', () => {
    // Form validation
    const forms = document.querySelectorAll('.needs-validation');
    forms.forEach(form => {
        form.addEventListener('submit', event => {
            if (!form.checkValidity()) {
                event.preventDefault();
                event.stopPropagation();
            }
            form.classList.add('was-validated');
        }, false);
    });

    // Audio file preview
    const fileInputs = document.querySelectorAll('input[type="file"]');
    fileInputs.forEach(input => {
        input.addEventListener('change', event => {
            const file = event.target.files[0];
            if (file) {
                const previewId = input.getAttribute('data-preview');
                const preview = document.getElementById(previewId);
                if (preview) {
                    try {
                        preview.src = URL.createObjectURL(file);
                        preview.style.display = 'block';
                    } catch (error) {
                        console.error('B??d podczas ?adowania podgl?du pliku:', error);
                        alert('Nie uda?o si? za?adowa? podgl?du pliku.');
                    }
                } else {
                    console.warn(`Podgl?d z id '${previewId}' nie zosta? znaleziony.`);
                }
            }
        });
    });

    // Initialize audio playback buttons
    const playAudioButtons = document.querySelectorAll('.play-audio-btn');
    playAudioButtons.forEach(button => {
        button.addEventListener('click', () => {
            const audioId = button.getAttribute('data-audio-id');
            const audioElement = document.getElementById(audioId);
            if (audioElement) {
                if (audioElement.paused) {
                    audioElement.play();
                    button.innerHTML = '<i class="bi bi-pause-circle-fill"></i> Pauza';
                } else {
                    audioElement.pause();
                    button.innerHTML = '<i class="bi bi-play-circle-fill"></i> Odtwórz';
                }

                audioElement.addEventListener('ended', () => {
                    button.innerHTML = '<i class="bi bi-play-circle-fill"></i> Odtwórz';
                }, { once: true });
            } else {
                console.warn(`Element audio z id '${audioId}' nie zosta? znaleziony.`);
            }
        });
    });

    // Confirmation for deleting a profile
    const deleteButtons = document.querySelectorAll('.delete-profile-btn');
    deleteButtons.forEach(button => {
        button.addEventListener('click', event => {
            if (!confirm('Czy na pewno chcesz usun?? ten profil g?osu?')) {
                event.preventDefault();
            }
        });
    });
});
