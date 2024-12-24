import { createMessageElement } from './messageRenderer.js';
import { getBotResponse } from './chatService.js';

// DOM Elements
const messagesContainer = document.getElementById('messages');
const chatForm = document.getElementById('chat-form');
const messageInput = document.getElementById('message-input');

// Initial bot message
window.addEventListener('DOMContentLoaded', () => {
  const initialMessage = createMessageElement("Hi! I'm your AI assistant. How can I help you today?", true);
  messagesContainer.appendChild(initialMessage);
});

// Handle form submission
chatForm.addEventListener('submit', async (e) => {
  e.preventDefault();
  
  const message = messageInput.value.trim();
  if (!message) return;

  // Add user message
  const userMessage = createMessageElement(message, false);
  messagesContainer.appendChild(userMessage);
  
  // Clear input
  messageInput.value = '';
  
  // Scroll to bottom
  messagesContainer.scrollTop = messagesContainer.scrollHeight;

  // Get and display bot response
  const botResponse = await getBotResponse(message);
  const botMessage = createMessageElement(botResponse, true);
  messagesContainer.appendChild(botMessage);
  
  // Scroll to bottom again
  messagesContainer.scrollTop = messagesContainer.scrollHeight;
});