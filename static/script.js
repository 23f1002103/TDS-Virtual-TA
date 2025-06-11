document.addEventListener('DOMContentLoaded', () => {
    const questionInput = document.getElementById('questionInput');
    const imageInput = document.getElementById('imageInput');
    const sendButton = document.getElementById('sendButton');
    const chatHistory = document.getElementById('chatHistory');

    const API_ENDPOINT = 'http://127.0.0.1:8000/api/'; // Your FastAPI API endpoint

    // Function to append messages to chat history
    function displayMessage(message, type, imageUrl = null) {
        const msgElement = document.createElement('div');
        msgElement.className = `chat-message ${type}`;

        if (imageUrl) {
            const img = document.createElement('img');
            img.src = imageUrl;
            msgElement.appendChild(img);
        }

        const textNode = document.createElement('div');
        textNode.innerHTML = message; // Use innerHTML for links
        msgElement.appendChild(textNode);
        
        chatHistory.appendChild(msgElement);
        chatHistory.scrollTop = chatHistory.scrollHeight; // Scroll to bottom
    }

    // Function to handle sending message
    async function sendMessage() {
        const question = questionInput.value.trim();
        const imageFile = imageInput.files[0]; // Get the selected file

        if (!question && !imageFile) {
            alert('Please enter a question or select an image.');
            return;
        }

        // Display user message immediately
        const userImageUrl = imageFile ? URL.createObjectURL(imageFile) : null;
        displayMessage("You: " + question, "user", userImageUrl);
        
        // Add a loading indicator
        const loadingElement = document.createElement('div');
        loadingElement.className = 'chat-message ta loading-indicator';
        loadingElement.textContent = 'TA is typing...';
        chatHistory.appendChild(loadingElement);
        chatHistory.scrollTop = chatHistory.scrollHeight;


        questionInput.value = ''; // Clear input field
        imageInput.value = ''; // Clear file input
        if (userImageUrl) {
            URL.revokeObjectURL(userImageUrl); // Clean up the URL object after displaying
        }

        let base64Image = null;
        if (imageFile) {
            base64Image = await fileToBase64(imageFile); // Convert image to base64
        }

        // --- MODIFIED SECTION START ---
        // Construct the request body dynamically to omit 'image' if no file is selected
        const requestBody = { question: question };
        if (base64Image !== null) { // Only add the 'image' field if an image was actually provided
            requestBody.image = base64Image;
        }
        // --- MODIFIED SECTION END ---

        try {
            const response = await fetch(API_ENDPOINT, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(requestBody) // Use the dynamically constructed requestBody
            });

            const data = await response.json();

            // Remove loading indicator
            chatHistory.removeChild(loadingElement);

            if (response.ok) {
                displayMessage(data.answer, "ta");
                if (data.links && data.links.length > 0) {
                    let linksHtml = 'Sources:<ul>';
                    data.links.forEach(link => {
                        // Sanitize link text (basic example)
                        const sanitizedText = link.text.replace(/</g, "&lt;").replace(/>/g, "&gt;");
                        linksHtml += `<li><a href="${link.url}" target="_blank">${sanitizedText}</a></li>`;
                    });
                    linksHtml += '</ul>';
                    displayMessage(linksHtml, "ta-links");
                }
            } else {
                displayMessage("TA Error: " + (data.detail || data.answer || JSON.stringify(data)), "error");
            }
        } catch (error) {
            // Remove loading indicator on network error
            if (chatHistory.contains(loadingElement)) {
                chatHistory.removeChild(loadingElement);
            }
            displayMessage("Network Error: Could not connect to the TA. " + error.message, "error");
        }
    }

    // Utility function to convert File object to Base64
    function fileToBase64(file) {
        return new Promise((resolve, reject) => {
            const reader = new FileReader();
            reader.readAsDataURL(file); // Reads the file as a data URL (base64 encoded)
            reader.onload = () => resolve(reader.result);
            reader.onerror = error => reject(error);
        });
    }

    // Event Listeners
    sendButton.addEventListener('click', sendMessage);
    questionInput.addEventListener('keypress', (event) => {
        if (event.key === 'Enter') {
            sendMessage();
        }
    });
});