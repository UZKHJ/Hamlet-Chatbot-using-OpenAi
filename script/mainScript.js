const submit = document.getElementById("submit");
submit.addEventListener("click", queryFunc);

let conversation = [];

function queryFunc() {
    const user_input = document.getElementById("query").value;
    // Adds user's query with timestamp to the conversation
    const userTimestamp = new Date().toLocaleString();
    conversation.push({ type: 'Query', content: user_input, timestamp: userTimestamp });
    // http://advanced-deep-learning-406018.appspot.com
    // http://127.0.0.1:5000
    fetch("http://advanced-deep-learning-406018.appspot.com?query=" + encodeURIComponent(user_input), {method: "GET",
    })
    .then(response => response.text())
    .then(data => {
        console.log(data);
        const aiTimestamp = new Date().toLocaleString();
        // Adds bot's response with timestamp to the conversation
        conversation.push({ type: 'Response', content: data, timestamp: aiTimestamp });
        // Updates the UI with the entire conversation history
        updateConversation();
    })
    .catch(error => {
        console.error(error);
        alert("Error:" + error);
    });
}

function updateConversation() {
    const responseDiv = document.getElementById("response");
    // Clears previous content
    responseDiv.innerHTML = "";
    // Displays the entire conversation
    conversation.forEach(message => {
        const messageElement = document.createElement("div");
        messageElement.classList.add(message.type);
        messageElement.textContent = `${message.type === 'Query' ? 'Query' : 'Response'} (${message.timestamp}): ${message.content}`;
        responseDiv.appendChild(messageElement);
    });
}

// Triggers save when the browser is closed
window.onbeforeunload = function() {
    saveConversation();
};

function saveConversation() {
    // Creates a Blob with the conversation content
    const conversationText = conversation.map(message => `${message.type} (${message.timestamp}): ${message.content}`).join('\n');
    const blob = new Blob([conversationText], { type: 'text/plain' });
    // Creates a link and trigger a click to prompt the user to download the file
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = 'conversation.txt';
    a.style.display = 'none';
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
}