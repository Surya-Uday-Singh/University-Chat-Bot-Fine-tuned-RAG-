import React from "react";
import { Box, Typography } from "@mui/material";
import { Message } from "../App";

interface Props {
  messages: Message[];
}

const ChatWindow: React.FC<Props> = ({ messages }) => (
  <Box
    sx={{
      flex: 1,
      overflowY: "auto",
      px: 3,
      py: 1,
      display: "flex",
      flexDirection: "column",
      gap: 3, 
      "&::-webkit-scrollbar": { width: "6px" },
      "&::-webkit-scrollbar-thumb": { bgcolor: "#2c2c2c", borderRadius: "3px" },
    }}
  >
    {messages.length === 0 && (
      <Box sx={{ m: 'auto', textAlign: 'center', color: 'text.secondary', maxWidth: '400px' }}>
        <Typography variant="body1" fontWeight={500}>
          Test queries against Algoma University data.
        </Typography>
        <Typography variant="body2" sx={{ mt: 1, color: '#555' }}>
          Explore course requirements, campus maps, or scholarship opportunities.
        </Typography>
      </Box>
    )}
    
    {messages.map((msg, idx) => {
      const isUser = msg.sender === "user";

      return (
        <Box key={idx} display="flex" justifyContent={isUser ? "flex-end" : "flex-start"} alignItems="flex-start">
          
          {/* Algoma Thunderbird/Eagle Icon for Bot */}
          {!isUser && (
             <Box sx={{ mr: 2, mt: 0.5, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                <img 
                  src="https://cdn-ileicge.nitrocdn.com/YIHSjYpDTcyGkWCowtUFbyZKKcYJYuAO/assets/images/source/rev-76af6c8/algomau.ca/wp-content/themes/understrap-child/img/eagle-icon.svg" 
                  alt="Algoma Bot" 
                  style={{ width: '28px', height: '28px', objectFit: 'contain' }} 
                />
             </Box>
          )}

          <Box
            sx={{
              maxWidth: "75%",
              p: isUser ? 2 : 0,
              pt: isUser ? 1.5 : 0.5,
              bgcolor: isUser ? "#1e1e1e" : "transparent",
              backgroundImage: isUser ? "linear-gradient(145deg, #1e1e1e 0%, #161616 100%)" : "none",
              border: isUser ? "1px solid #2a2a2a" : "none",
              borderRadius: isUser ? "20px 20px 4px 20px" : 0, 
              color: "text.primary",
            }}
          >
            <Typography sx={{ fontSize: "0.95rem", lineHeight: 1.6, whiteSpace: "pre-wrap" }}>
              {msg.text}
            </Typography>
            
            {/* Minimalist university-specific bot labels */}
            {!isUser && (
              <Typography variant="caption" sx={{ color: "#555", mt: 1, display: "block", fontWeight: 700, letterSpacing: "0.5px", textTransform: 'uppercase' }}>
                {msg.model === "rag" ? "ALGOMA RAG SYSTEM" : "QLoRA MODEL (LLAMA-3)"}
              </Typography>
            )}
          </Box>
        </Box>
      );
    })}
  </Box>
);

export default ChatWindow;