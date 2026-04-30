window.PortfolioAIWidget = class PortfolioAIWidget {    constructor(apiUrl) {
        this.apiUrl = apiUrl;
        this.isOpen = false;
        this.isWaiting = false;
        this.containerId = 'ai-widget-container';
    }

    init() {
        if (document.getElementById(this.containerId)) return;

        this.injectHTML();
        this.cacheDOM();
        this.attachEventListeners();
        
        // A slightly more professional greeting
        this.addMessage("Hi there! I'm an AI agent built to answer questions about Jaynil's engineering portfolio. How can I help you today?", 'bot');
    }

    injectHTML() {
        const container = document.createElement('div');
        container.id = this.containerId;
        
        container.innerHTML = `
            <div id="ai-widget-window">
                <div class="ai-widget-header">
                    <div class="ai-header-info">
                        <span class="ai-header-title">AI Gateway</span>
                        <span class="ai-header-status"><span class="ai-status-dot"></span> Systems Online</span>
                    </div>
                    <button class="ai-widget-close" id="ai-widget-close">
                        <svg width="14" height="14" viewBox="0 0 14 14" fill="none" xmlns="http://www.w3.org/2000/svg">
                            <path d="M13 1L1 13M1 1L13 13" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
                        </svg>
                    </button>
                </div>
                <div class="ai-widget-messages" id="ai-widget-messages"></div>
                <form class="ai-widget-input-area" id="ai-widget-form">
                    <input type="text" id="ai-widget-input" placeholder="Ask about distributed systems..." autocomplete="off">
                    <button type="submit" id="ai-widget-submit" aria-label="Send Message">
                        <svg viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                            <path d="M2.01 21L23 12 2.01 3 2 10l15 2-15 2z"/>
                        </svg>
                    </button>
                </form>
            </div>
            <button id="ai-widget-toggle" aria-label="Toggle AI Chat">
                <svg viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                    <path d="M20 2H4c-1.1 0-2 .9-2 2v18l4-4h14c1.1 0 2-.9 2-2V4c0-1.1-.9-2-2-2zm0 14H6l-2 2V4h16v12z"/>
                </svg>
            </button>
        `;
        document.body.appendChild(container);
    }

    cacheDOM() {
        this.container = document.getElementById(this.containerId);
        this.toggleBtn = document.getElementById('ai-widget-toggle');
        this.closeBtn = document.getElementById('ai-widget-close');
        this.chatWindow = document.getElementById('ai-widget-window');
        this.form = document.getElementById('ai-widget-form');
        this.input = document.getElementById('ai-widget-input');
        this.submitBtn = document.getElementById('ai-widget-submit');
        this.messagesContainer = document.getElementById('ai-widget-messages');
    }

    attachEventListeners() {
        this.toggleBtn.addEventListener('click', () => this.toggleWindow());
        this.closeBtn.addEventListener('click', () => this.toggleWindow());
        this.form.addEventListener('submit', (e) => this.handleSubmit(e));
    }

    toggleWindow() {
        this.isOpen = !this.isOpen;
        if (this.isOpen) {
            this.container.classList.add('open');
            this.chatWindow.classList.add('open');
            setTimeout(() => this.input.focus(), 300);
        } else {
            this.container.classList.remove('open');
            this.chatWindow.classList.remove('open');
        }
    }

    addMessage(text, sender) {
        const msgDiv = document.createElement('div');
        msgDiv.classList.add('ai-msg', sender);
        msgDiv.textContent = text;
        this.messagesContainer.appendChild(msgDiv);
        this.scrollToBottom();
    }

    setTypingIndicator(show) {
        if (show) {
            const typingDiv = document.createElement('div');
            typingDiv.id = 'ai-typing-indicator';
            typingDiv.className = 'typing-indicator';
            typingDiv.innerHTML = `
                <div class="typing-dot"></div>
                <div class="typing-dot"></div>
                <div class="typing-dot"></div>
            `;
            this.messagesContainer.appendChild(typingDiv);
            this.scrollToBottom();
        } else {
            const indicator = document.getElementById('ai-typing-indicator');
            if (indicator) indicator.remove();
        }
    }

    scrollToBottom() {
        this.messagesContainer.scrollTop = this.messagesContainer.scrollHeight;
    }

    async handleSubmit(e) {
        e.preventDefault();
        const question = this.input.value.trim();
        
        if (!question || this.isWaiting) return;

        this.addMessage(question, 'user');
        this.input.value = '';
        this.isWaiting = true;
        this.submitBtn.disabled = true;
        this.setTypingIndicator(true);

        try {
            const response = await fetch(`${this.apiUrl}/ask`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ question: question })
            });

            if (!response.ok) {
                if (response.status === 429) throw new Error("Rate limit exceeded. Try again in a minute.");
                throw new Error(`Server Error: ${response.status}`);
            }

            const data = await response.json();
            this.setTypingIndicator(false);
            this.addMessage(data.answer, 'bot');

        } catch (error) {
            console.error("Widget Error:", error);
            this.setTypingIndicator(false);
            this.addMessage(`Oops, something went wrong: ${error.message}`, 'bot');
        } finally {
            this.isWaiting = false;
            this.submitBtn.disabled = false;
            this.input.focus();
        }
    }
}