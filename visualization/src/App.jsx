import React, { useState } from "react";
import "./App.css";
import useCSV from "../hooks/useCSV";
import PredictionHistogram from "./components/PredictionHistogram";

export default function App() {
  const naiveBayesDataUndersampling = useCSV("/exported_models/naive_bayes_undersampling.csv");
  const naiveBayesDataNoUndersampling = useCSV("/exported_models/naive_bayes_no_undersampling.csv");
  const cascadeTunnel = useCSV("/exported_models/cascade_tunnel.csv");

  const [selected, setSelected] = useState("nb_under");

  const renderChart = () => {
    switch (selected) {
      case "nb_under":
        return (
          <PredictionHistogram
            data={naiveBayesDataUndersampling.data}
            loading={naiveBayesDataUndersampling.loading}
            title="Prediction Distribution Histogram (Undersampling)"
            subtitle="Comparison of Model Confidence vs. Actual Outcomes (Naive Bayes)"
            xAxisLabel="Probability of Death (Model Confidence)"
          />
        );
      case "nb_no_under":
        return (
          <PredictionHistogram
            data={naiveBayesDataNoUndersampling.data}
            loading={naiveBayesDataNoUndersampling.loading}
            title="Prediction Distribution Histogram (No Undersampling)"
            subtitle="Comparison of Model Confidence vs. Actual Outcomes (Naive Bayes)"
            xAxisLabel="Probability of Death (Model Confidence)"
          />
        );
      case "cascade":
        return (
          <PredictionHistogram
            data={cascadeTunnel.data}
            loading={cascadeTunnel.loading}
            title="Cascade Tunnel Prediction Distribution"
            subtitle="Combined Confidence (Stage 1 + Stage 2) vs. Actual Outcomes"
            xAxisLabel="Probability of Death (Final System Confidence)"
          />
        );
      default:
        return null;
    }
  };

  return (
    <div className="app-container">
      <nav className="navbar">
        <button
          className={selected === "nb_under" ? "active" : ""}
          onClick={() => setSelected("nb_under")}
        >
          Naive Bayes (Undersample)
        </button>
        <button
          className={selected === "nb_no_under" ? "active" : ""}
          onClick={() => setSelected("nb_no_under")}
        >
          Naive Bayes (No Undersample)
        </button>
        <button
          className={selected === "cascade" ? "active" : ""}
          onClick={() => setSelected("cascade")}
        >
          Cascade Tunnel
        </button>
      </nav>

      <div className="chart-container">{renderChart()}</div>
    </div>
  );
}