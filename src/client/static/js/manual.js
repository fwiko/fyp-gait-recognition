
const imageInput = document.getElementById('imageInput');
const dropArea = document.getElementById('imageInputLabel');
const person = document.getElementById('person');
const access = document.getElementById('access');
const confidence = document.getElementById('confidence');
const previewImage = document.getElementById('previewImage');

function handleImageInput() {
    previewImage.src = '';
    previewImage.alt = 'Loading...';

    const imageFile = imageInput.files[0];
    const reader = new FileReader();

    reader.readAsDataURL(imageFile);

    reader.onload = async () => {
        const base64Image = reader.result.split(',')[1];
        try {
            const response = await fetch('/api/classify', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    gei: base64Image
                })
            });
            if (!response.ok) {
                throw new Error('Classification failed');
            }
            const result = await response.json();

            person.textContent = result.person;
            confidence.textContent = `${result.confidence}%`;
            access.textContent = result.access ? 'Granted' : 'Denied';
            access.classList.toggle('access-denied', !result.access);
            
            previewImage.src = reader.result;
        } catch (error) {
            console.error('Error:', error);
            alert('Classification failed. Please try again.');
        }
    }
}


document.getElementById('imageInput').addEventListener('change', async (e) => {
    e.preventDefault();
    handleImageInput();
});


dropArea.addEventListener('drop', (e) => {
    e.preventDefault();

    if (e.dataTransfer.files.length > 0) {
        imageInput.files = e.dataTransfer.files;
        handleImageInput();
    }
});