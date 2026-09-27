import React from "react";
import { ToggleButton, ToggleButtonGroup } from "@mui/material";
import { ModelType } from "../App";

interface Props {
  model: ModelType;
  setModel: (model: ModelType) => void;
}

const ModelToggle: React.FC<Props> = ({ model, setModel }) => (
  <ToggleButtonGroup
    value={model}
    exclusive
    onChange={(_, value) => value && setModel(value)}
    size="small"
    sx={{
      bgcolor: "#1a1a1a",
      borderRadius: "20px",
      p: 0.5,
      "& .MuiToggleButtonGroup-grouped": {
        border: 0,
        borderRadius: "16px !important",
        px: 2.5,
        py: 0.5,
        textTransform: "none",
        fontWeight: 600,
        color: "#555",
        "&.Mui-selected": {
          bgcolor: "#B3282D", // Algoma Red
          color: "#fff",
          boxShadow: "0px 2px 10px rgba(179, 40, 45, 0.4)",
        },
        "&:hover": {
          bgcolor: "transparent",
          color: "#fff",
        },
        "&.Mui-selected:hover": {
          bgcolor: "#B3282D",
        }
      },
    }}
  >
    <ToggleButton disableRipple value="rag">RAG</ToggleButton>
    <ToggleButton disableRipple value="llm">QLoRA</ToggleButton>
  </ToggleButtonGroup>
);

export default ModelToggle;