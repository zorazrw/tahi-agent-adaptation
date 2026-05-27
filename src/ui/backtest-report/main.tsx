import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import "../index.css";
import { BacktestReportApp } from "./BacktestReportApp";

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <BacktestReportApp />
  </StrictMode>,
);
