const urlParams = new URLSearchParams(window.location.search);
const fileId = urlParams.get('file_id');
let cleanedFileId = null;
let currentPreviewType = 'original';

document.addEventListener('DOMContentLoaded', () => {
    if (!fileId) {
        alert("No file ID provided");
        window.location.href = "index.html";
        return;
    }
    displaySummary();
    displaySuggestions();
    displayPreview();

    document.getElementById('fillMissing').addEventListener('change', (e) => {
        document.getElementById('fillMethod').disabled = !e.target.checked;
    });

    document.getElementById('applyCleaning').addEventListener('click', applyCleaning);
    document.getElementById('applySuggestions').addEventListener('click', applySuggestions);
    document.getElementById('downloadBtn').addEventListener('click', downloadDataset);
});

function toggleCollapse(sectionId) {
    const section = document.getElementById(sectionId);
    const header = section.previousElementSibling;
    section.classList.toggle('collapsed');
    header.classList.toggle('collapsed');
}

function togglePreview(type) {
    currentPreviewType = type;
    document.getElementById('originalBtn').classList.toggle('active', type === 'original');
    document.getElementById('cleanedBtn').classList.toggle('active', type === 'cleaned');
    displayPreview();
}

async function refreshPreview() {
    displayPreview();
}

async function displaySummary() {
    try {
        const summaryResponse = await fetch(`http://127.0.0.1:8000/api/summary/${fileId}`);
        if (!summaryResponse.ok) {
            throw new Error(`Failed to fetch summary: ${summaryResponse.statusText}`);
        }
        const summary = await summaryResponse.json();
        const content = document.getElementById('summaryContent');
        content.innerHTML = `
            <p><strong>Rows:</strong> ${summary.rows}</p>
            <p><strong>Duplicates:</strong> ${summary.duplicates}</p>
            <table>
                <thead>
                    <tr>
                        <th>Column</th>
                        <th>Type</th>
                        <th>Nulls</th>
                        <th>Unique</th>
                    </tr>
                </thead>
                <tbody>
                    ${summary.columns.map(col => `
                        <tr>
                            <td>${col.name}</td>
                            <td>${col.type}</td>
                            <td>${col.nulls}</td>
                            <td>${col.unique}</td>
                        </tr>
                    `).join('')}
                </tbody>
            </table>
        `;
    } catch (error) {
        console.error("Summary error:", error);
        document.getElementById('summaryContent').innerHTML = "<p>Error loading summary: " + error.message + "</p>";
    }
}

async function displayPreview() {
    const previewTable = document.getElementById('previewTable');
    const previewSpinner = document.getElementById('previewSpinner');
    previewTable.style.display = 'none';
    previewSpinner.style.display = 'flex';

    try {
        console.log("Fetching preview for file_id:", fileId, "cleaned_file_id:", cleanedFileId, "type:", currentPreviewType);
        let previewUrl = `http://127.0.0.1:8000/api/preview/${fileId}`;
        if (currentPreviewType === 'cleaned' && cleanedFileId) {
            previewUrl += `?cleaned_file_id=${cleanedFileId}`;
        }
        const previewResponse = await fetch(previewUrl);
        if (!previewResponse.ok) {
            throw new Error(`Failed to fetch preview: ${previewResponse.statusText}`);
        }
        const preview = await previewResponse.json();
        console.log("Preview data:", preview);

        if (currentPreviewType === 'cleaned' && preview.preview_type === 'original') {
            throw new Error("No cleaned dataset available yet. Apply cleaning or suggestions first.");
        }

        previewTable.innerHTML = `
            <table>
                <thead>
                    <tr>
                        ${preview.columns.map(col => `<th>${col}</th>`).join('')}
                    </tr>
                </thead>
                <tbody>
                    ${preview.data.map(row => `
                        <tr>
                            ${row.map(cell => `<td>${cell}</td>`).join('')}
                        </tr>
                    `).join('')}
                </tbody>
            </table>
        `;
    } catch (error) {
        console.error("Preview error:", error);
        previewTable.innerHTML = "<p>Error loading preview: " + error.message + "</p>";
    } finally {
        previewSpinner.style.display = 'none';
        previewTable.style.display = 'block';
    }
}

async function displaySuggestions() {
    try {
        const response = await fetch(`http://127.0.0.1:8000/api/suggestions/${fileId}`);
        if (!response.ok) throw new Error("Failed to fetch suggestions");
        const data = await response.json();
        const content = document.getElementById('suggestionsContent');
        if (data.suggestions && Array.isArray(data.suggestions)) {
            content.innerHTML = data.suggestions.map(sug => `
                <div class="suggestion-item" data-id="${sug.id}">
                    <span>${sug.text}</span>
                    <div>
                        <button class="btn accept-btn" onclick="acceptSuggestion(${sug.id})">Accept</button>
                        <button class="btn reject-btn" onclick="rejectSuggestion(${sug.id})">Reject</button>
                    </div>
                </div>
            `).join('');
        } else {
            // Handle fallback case explicitly
            content.innerHTML = `
                <div class="suggestion-item" data-id="1">
                    <span>Error fetching AI suggestions; ${data.suggestions ? data.suggestions.text : 'drop high-null columns'}</span>
                    <div>
                        <button class="btn accept-btn" onclick="acceptSuggestion(1)">Accept</button>
                        <button class="btn reject-btn" onclick="rejectSuggestion(1)">Reject</button>
                    </div>
                </div>
            `;
            console.warn("Fallback suggestions displayed:", data.suggestions);
        }
    } catch (error) {
        console.error("Suggestions error:", error);
        document.getElementById('suggestionsContent').innerHTML = "<p>Error loading suggestions: " + error.message + "</p>";
    }
}

function acceptSuggestion(id) {
    const item = document.querySelector(`.suggestion-item[data-id="${id}"]`);
    item.style.background = '#e6ffe6';
    console.log(`Accepted suggestion ${id}`);
}

function rejectSuggestion(id) {
    const item = document.querySelector(`.suggestion-item[data-id="${id}"]`);
    item.style.background = '#ffe6e6';
    console.log(`Rejected suggestion ${id}`);
}

async function applyCleaning() {
    const options = {
        removeDuplicates: document.getElementById('removeDuplicates').checked,
        dropMissing: document.getElementById('dropMissing').checked,
        fillMissing: document.getElementById('fillMissing').checked,
        fillMethod: document.getElementById('fillMethod').value
    };
    try {
        const response = await fetch(`http://127.0.0.1:8000/api/clean/${fileId}`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(options)
        });
        if (!response.ok) throw new Error("Cleaning failed");
        const { cleaned_file_id } = await response.json();
        cleanedFileId = cleaned_file_id;
        alert("Cleaning applied successfully!");
        displaySummary();
        if (currentPreviewType === 'cleaned') {
            displayPreview();
        }
    } catch (error) {
        console.error("Cleaning error:", error);
        alert("Error applying cleaning: " + error.message);
    }
}

async function applySuggestions() {
    const acceptedIds = Array.from(document.querySelectorAll('.suggestion-item[style*="background: #e6ffe6"]'))
        .map(item => parseInt(item.dataset.id));
    try {
        const response = await fetch(`http://127.0.0.1:8000/api/apply-suggestions/${fileId}`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(acceptedIds)
        });
        if (!response.ok) throw new Error("Suggestions application failed");
        const { cleaned_file_id } = await response.json();
        cleanedFileId = cleaned_file_id;
        alert("Suggestions applied successfully!");
        displaySummary();
        if (currentPreviewType === 'cleaned') {
            displayPreview();
        }
    } catch (error) {
        console.error("Suggestions error:", error);
        alert("Error applying suggestions: " + error.message);
    }
}

async function downloadDataset() {
    if (!cleanedFileId) {
        alert("Please apply cleaning or suggestions first");
        return;
    }
    try {
        window.location.href = `http://127.0.0.1:8000/api/download/${cleanedFileId}`;
    } catch (error) {
        console.error("Download error:", error);
        alert("Error downloading dataset: " + error.message);
    }
}