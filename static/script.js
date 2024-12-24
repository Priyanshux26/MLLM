// Function to send a message
function sendMessage() {
    const userMessage = document.getElementById("user-message").value;
    if (userMessage.trim() === "") return;

    // Display user message
    displayMessage(userMessage, "user");

    // Send the message to the FastAPI backend
    fetch("/predict", {
        method: "POST",
        headers: {
            "Content-Type": "application/json"
        },
        body: JSON.stringify({ sentence: userMessage })
    })
    .then(response => response.json())
    .then(data => {
        const intent = data.intent;
        const slots = data.slots;

        let botMessage = `Intent: ${intent}\nSlots: ${slots.join(", ")}`;
        // Display bot's response
        displayMessage(botMessage, "bot");
    })
    .catch(error => {
        console.error("Error:", error);
        displayMessage("Sorry, I couldn't process your request.", "bot");
    });

    // Clear the input field
    document.getElementById("user-message").value = "";
    document.getElementById("send-btn").disabled = true;
}

// Function to display a message in the chat box
function displayMessage(message, sender) {
    const chatBox = document.getElementById("chat-box");
    const messageElement = document.createElement("div");
    messageElement.classList.add("chat-message", sender);
    messageElement.textContent = message;

    chatBox.appendChild(messageElement);
    chatBox.scrollTop = chatBox.scrollHeight;  // Scroll to bottom
}

// Enable the send button when there is text
function enableSend() {
    const userMessage = document.getElementById("user-message").value;
    document.getElementById("send-btn").disabled = userMessage.trim() === "";
}
