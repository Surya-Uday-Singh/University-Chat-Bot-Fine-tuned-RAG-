import React, { useState } from "react";
import { Box, OutlinedInput, IconButton, InputAdornment } from "@mui/material";
import ArrowUpwardIcon from '@mui/icons-material/ArrowUpward';

interface Props {
  onSend: (text: string) => void;
}

const MessageInput: React.FC<Props> = ({ onSend }) => {
  const [text, setText] = useState("");

  const handleSend = () => {
    if (text.trim()) {
      onSend(text);
      setText("");
    }
  };

  return (
    <Box 
      sx={{ 
        display: "flex", 
        bgcolor: "rgba(18, 18, 18, 0.7)", 
        backdropFilter: "blur(12px)", 
        borderRadius: "24px",
        border: "1px solid #2a2a2a",
        transition: "all 0.3s ease",
        "&:focus-within": {
          borderColor: "primary.main",
          boxShadow: "0 4px 20px rgba(179, 40, 45, 0.2)"
        }
      }}
    >
      <OutlinedInput
        fullWidth
        multiline
        maxRows={5}
        placeholder="Ask about courses, campus life, or scholarships..."
        value={text}
        onChange={e => setText(e.target.value)}
        onKeyDown={e => {
          if (e.key === "Enter" && !e.shiftKey) {
            e.preventDefault();
            handleSend();
          }
        }}
        sx={{
          color: "#fff",
          py: 1.5,
          px: 2,
          "& fieldset": { border: "none" }, 
          "& input::placeholder, & textarea::placeholder": {
            color: "#555",
            opacity: 1,
          },
        }}
        endAdornment={
          <InputAdornment position="end" sx={{ alignSelf: 'flex-end', mb: 0.5 }}>
            <IconButton 
              onClick={handleSend}
              disabled={!text.trim()}
              sx={{ 
                bgcolor: text.trim() ? "primary.main" : "#2a2a2a", // Active is Algoma Red
                color: "#fff",
                transition: "all 0.2s",
                "&:hover": {
                  bgcolor: text.trim() ? "#8f1f23" : "#2a2a2a", // Darker Red on hover
                },
                "&.Mui-disabled": {
                  bgcolor: "#1a1a1a",
                  color: "#444"
                }
              }}
              size="small"
            >
              <ArrowUpwardIcon fontSize="small" />
            </IconButton>
          </InputAdornment>
        }
      />
    </Box>
  );
};

export default MessageInput;