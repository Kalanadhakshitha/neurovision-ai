import React, { useState, useEffect, useRef } from "react";
import "./App.css";

const Chatbot = ({ result }) => {
  const [isOpen, setIsOpen] = useState(false);
  const [messages, setMessages] = useState([
    {
      text: "Hi! I'm Dr. AI. I can explain your MRI results. Ask me anything!",
      sender: "bot",
    },
  ]);
  const [input, setInput] = useState("");
  const [isTyping, setIsTyping] = useState(false);
  const messagesEndRef = useRef(null);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, isTyping]);

  useEffect(() => {
    if (result && result.result !== "No Tumor") {
      setMessages((prev) => [
        ...prev,
        {
          text: `I see a **${result.result}**. Do you have questions about this?`,
          sender: "bot",
        },
      ]);
    }
  }, [result]);

  const handleSend = async () => {
    if (!input.trim()) return;

    const userMessage = { text: input, sender: "user" };
    setMessages((prev) => [...prev, userMessage]);
    setInput("");
    setIsTyping(true);

    try {
      const response = await fetch("https://Kalana.pythonanywhere.com/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          message: input,
          context: result
            ? `Condition: ${result.result}, Status: ${result.status}`
            : "No scan uploaded",
        }),
      });

      const data = await response.json();
      setMessages((prev) => [...prev, { text: data.reply, sender: "bot" }]);
    } catch (error) {
      console.error("Chat Error:", error);
      setMessages((prev) => [
        ...prev,
        {
          text: "Server connection failed. Make sure 'python app.py' is running.",
          sender: "bot",
        },
      ]);
    } finally {
      setIsTyping(false);
    }
  };

  return (
    <div className="chatbot-wrapper">
      <button className="chat-toggle-btn" onClick={() => setIsOpen(!isOpen)}>
        {isOpen ? "✖" : "💬 Dr. AI"}
      </button>

      {isOpen && (
        <div className="chat-window">
          <div className="chat-header">
            <h3>🤖 Dr. AI</h3>
          </div>
          <div className="chat-body">
            {messages.map((msg, index) => (
              <div key={index} className={`chat-message ${msg.sender}`}>
                <span
                  dangerouslySetInnerHTML={{
                    __html: msg.text.replace(/\*\*(.*?)\*\*/g, "<b>$1</b>"),
                  }}
                />
              </div>
            ))}
            {isTyping && <div className="chat-message bot">Thinking...</div>}
            <div ref={messagesEndRef} />
          </div>
          <div className="chat-footer">
            <input
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyPress={(e) => e.key === "Enter" && handleSend()}
              placeholder="Type here..."
            />
            <button onClick={handleSend}>➤</button>
          </div>
        </div>
      )}
    </div>
  );
};

export default Chatbot;
