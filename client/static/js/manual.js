let currentImage = null;
const imageInput = document.getElementById('imageInput');

document.getElementById('imageInput').addEventListener('change', async (e) => {
    e.preventDefault();

    const imageFile = imageInput.files[0];
    const reader = new FileReader();

    reader.readAsDataURL(imageFile);

    reader.onload = async () => {
        currentImage = reader.result;
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
            console.log(result);
            document.getElementById('personResult').textContent = result.person || 'Unknown';
            document.getElementById('confidenceResult').textContent =
                result.confidence !== undefined ? `${result.confidence}%` : 'N/A';
            document.getElementById('accessResult').textContent =
                result.access ? 'Allowed' : 'Denied';
            document.getElementById('previewImage').src = currentImage;
        } catch (error) {
            console.error('Error:', error);
            alert('Classification failed. Please try again.');
        }
    }

})