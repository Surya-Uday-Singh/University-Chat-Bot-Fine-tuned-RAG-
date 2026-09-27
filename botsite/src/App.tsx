import React, { useState } from "react";
import { ThemeProvider, createTheme } from "@mui/material/styles";
import { Box, Typography, CssBaseline } from "@mui/material";
import ChatWindow from "./components/ChatWindow";
import MessageInput from "./components/MessageInput";
import ModelToggle from "./components/ModelToggle";
import algomaLogo from "../public/images/algoma_logo.png"; // Ensure this path is correct and the image is in place

export type ModelType = "rag" | "llm";

export interface Message {
  sender: "user" | "bot";
  text: string;
  model: ModelType;
}

const algomaTheme = createTheme({
  palette: {
    mode: "dark",
    background: {
      default: "#0a0a0a", // Deep black
      paper: "#121212",   // Slightly elevated dark grey
    },
    primary: {
      main: "#B3282D",    // Official Algoma Red
    },
    secondary: {
      main: "#fff",       // Pure white for high contrast logo
    },
    text: {
      primary: "#fff",
      secondary: "#777",  // Mid-grey for secondary text
    },
  },
  typography: {
    fontFamily: '"Inter", "Roboto", "Helvetica", "Arial", sans-serif',
  },
});

const App: React.FC = () => {
  const [ragMessages, setRagMessages] = useState<Message[]>([]);
  const [llmMessages, setLlmMessages] = useState<Message[]>([]);
  const [model, setModel] = useState<ModelType>("rag");

 const handleSend = async (text: string) => {
    // 1. Add the user's message to the screen immediately
    const newUserMessage: Message = { sender: "user", text, model };
    if (model === "rag") {
      setRagMessages((prev) => [...prev, newUserMessage]);
    } else {
      setLlmMessages((prev) => [...prev, newUserMessage]);
    }

    try {
      let botReplyText = "";

      // 2. ROUTING LOGIC: Check which model is selected
      if (model === "rag") {
        
        // --- Call the RAG API ---
        const response = await fetch("http://localhost:8000/chat", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ message: text }), 
        });

        if (!response.ok) throw new Error("Failed to connect to RAG API");
        
        const data = await response.json();
        botReplyText = data.reply; 

      } else if (model === "llm") {
        
        // --- Call the QLoRA API (Colab/ngrok backend) ---
        // IMPORTANT: Replace this with your active ngrok public URL
        const ngrokUrl = "https://carri-plumbless-nonegotistically.ngrok-free.dev/generate"; 
        
        const llmResponse = await fetch(ngrokUrl, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ question: text }), // Matches our FastAPI 'UserRequest' model
        });

        if (!llmResponse.ok) throw new Error(`Failed to connect to QLoRA API. Status: ${llmResponse.status}`);
        
        const llmData = await llmResponse.json();
        botReplyText = llmData.answer; // Matches our FastAPI return dict {"answer": ...}
        
      }

      // 3. Update the UI with the final Bot response
      const newBotMessage: Message = { 
        sender: "bot", 
        text: botReplyText, 
        model 
      };
      
      if (model === "rag") {
        setRagMessages((prev) => [...prev, newBotMessage]);
      } else {
        setLlmMessages((prev) => [...prev, newBotMessage]);
      }

    } catch (error) {
      console.error("Error communicating with backend:", error);
      
      // Fallback message if the server is down
      const errorBotMessage: Message = { 
        sender: "bot", 
        text: `⚠️ Connection Error: Could not reach the ${model.toUpperCase()} backend.`, 
        model 
      };
      
      if (model === "rag") {
        setRagMessages((prev) => [...prev, errorBotMessage]);
      } else {
        setLlmMessages((prev) => [...prev, errorBotMessage]);
      }
    }
  };

  const messages = model === "rag" ? ragMessages : llmMessages;

  return (
    <ThemeProvider theme={algomaTheme}>
      <CssBaseline />
      <Box sx={{ height: "100vh", width: "100vw", display: "flex", justifyContent: "center", bgcolor: 'background.default' }}>
        <Box
          sx={{
            flex: 1,
            maxWidth: "1000px",
            display: "flex",
            flexDirection: "column",
            position: "relative",
            p: 3,
          }}
        >
          {/* Header - Brand Prominence */}
          <Box display="flex" justifyContent="space-between" alignItems="center" px={3} pt={2} pb={4}>
            <Box display="flex" alignItems="center">
              {/* Replace placeholder with official logo asset */}
              <img 
                src="/images/algoma_logo.png" 
                alt="Algoma University Logo" 
                style={{ height: '40px', marginRight: '16px' }}
              />
              <Typography
                variant="h6"
                fontWeight={700}
                sx={{
                  letterSpacing: "-0.5px",
                }}
              >
                University QA Bot
              </Typography>
            </Box>
            <ModelToggle model={model} setModel={setModel} />
          </Box>

          {/* Chat Area */}
          <ChatWindow messages={messages} />

          {/* Input Area */}
          <Box px={3} pb={2} pt={2}>
            <MessageInput onSend={handleSend} />
          </Box>
        </Box>
      </Box>
    </ThemeProvider>
  );
};

export default App;