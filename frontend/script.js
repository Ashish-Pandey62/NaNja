const uploadArea = document.getElementById('uploadArea');
const fileInput = document.getElementById('fileInput');
const fileNameDisplay = document.getElementById('file-name');
const successMessage = document.getElementById('successMessage');

uploadArea.addEventListener('click', () => fileInput.click());

uploadArea.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadArea.classList.add('dragover');
});

uploadArea.addEventListener('dragleave', () => {
    uploadArea.classList.remove('dragover');
});

uploadArea.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadArea.classList.remove('dragover');
    const files = e.dataTransfer.files;
    fileInput.files = files;
    uploadFile(files);
});

fileInput.addEventListener('change', () => {
    uploadFile(fileInput.files);
});

async function uploadFile(files) {
    if (files.length === 0) return;
    const file = files[0];
    const formData = new FormData();
    formData.append("file", file);
    
    try {
        fileNameDisplay.textContent = `Selected: ${file.name}`;
        fileNameDisplay.style.display = 'block';
        successMessage.style.display = 'block';
        setTimeout(() => successMessage.style.display = 'none', 3000);
        
        const response = await fetch("http://127.0.0.1:8000/api/upload", {
            method: "POST",
            body: formData
        });
        if (!response.ok) throw new Error("Upload failed");
        const { file_id } = await response.json();
        window.location.href = `processing.html?file_id=${file_id}`;
    } catch (error) {
        console.error("Upload error:", error);
        fileNameDisplay.textContent = "Error uploading file";
        fileNameDisplay.style.color = "#dc3545";
    }
}