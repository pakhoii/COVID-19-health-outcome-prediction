import { useState, useEffect } from "react";
import * as d3 from "d3";

export default function useCSV(csvPath) {
  const [data, setData] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    if (!csvPath) return;

    setLoading(true);

    d3.csv(csvPath)
      .then((csvData) => {
        // Auto convert all numeric fields
        const parsed = csvData.map(row => {
          const converted = {};
          Object.entries(row).forEach(([key, value]) => {
            const num = +value;
            converted[key] = isNaN(num) ? value : num;
          });
          return converted;
        });

        setData(parsed);
        setLoading(false);
      })
      .catch((err) => {
        setError(err);
        console.error("CSV Load Error:", err);
        setLoading(false);
      });
  }, [csvPath]);

  return { data, loading, error };
}
