import React, { useEffect, useRef, useState } from "react";
import * as d3 from "d3";

export default function PredictionHistogram({
    data: propData,
    loading: propLoading,
    title,
    subtitle,
    xAxisLabel
}) {
    const chartRef = useRef(null);
    const [data, setData] = useState(propData);
    const [loading, setLoading] = useState(propLoading);

    // Update when props change
    useEffect(() => {
        setData(propData);
        setLoading(propLoading);
    }, [propData, propLoading]);

    // Draw chart
    useEffect(() => {
        if (loading || !data || data.length === 0) return;

        // Clean up previous chart
        d3.select(chartRef.current).selectAll("*").remove();
        d3.select("body").selectAll(".d3-tooltip").remove();

        // Configure dimensions
        const margin = { top: 40, right: 30, bottom: 60, left: 90 };
        const width = 900 - margin.left - margin.right;
        const height = 600 - margin.top - margin.bottom;

        // Create SVG
        const svg = d3.select(chartRef.current)
            .append("svg")
            .attr("width", width + margin.left + margin.right)
            .attr("height", height + margin.top + margin.bottom)
            .append("g")
            .attr("transform", `translate(${margin.left},${margin.top})`);

        // Create tooltip
        const tooltip = d3.select("body")
            .append("div")
            .attr("class", "d3-tooltip");

        // Scales and Bins
        const x = d3.scaleLinear()
            .domain([0, 1])
            .range([0, width]); // x-scale: Linear scale for probability from 0 to 1

        const histogram = d3.bin()
            .value(d => d.Prob_Died)
            .domain(x.domain())
            .thresholds(x.ticks(40));

        // Filter data based on Actual outcome
        const binsCured = histogram(data.filter(d => d.Actual === 0));
        const binsDied = histogram(data.filter(d => d.Actual === 1));

        const maxY = d3.max([
            d3.max(binsCured, d => d.length),
            d3.max(binsDied, d => d.length)
        ]);

        const y = d3.scaleLinear()
            .range([height, 0])
            .domain([0, maxY]); // y-scale: Scale by count of instances

        // Axes Definition
        svg.append("g") // x-axis
            .attr("class", "axis")
            .attr("transform", `translate(0,${height})`) // Move to the bottom of the chart
            .call(d3.axisBottom(x).ticks(10)) // Call function to create axis
            .append("text")
            .attr("class", "axis-label")
            .attr("y", 40) // Move label down from axis
            .attr("x", width / 2) // Center label horizontally
            .text(xAxisLabel || "Probability of Death (System Confidence)");

        svg.append("g") // y-axis
            .attr("class", "axis")
            .call(d3.axisLeft(y))
            .append("text")
            .attr("class", "axis-label")
            .attr("transform", "rotate(-90)")
            .attr("y", -70)
            .attr("x", -height / 2)
            .text("Number of Patients");

        const drawBars = (bins, cssClass, label) => {
            svg.selectAll("." + label)
                .data(bins)
                .enter()
                .append("rect")
                .attr("class", `bar ${cssClass}`)
                .attr("x", 1)
                .attr("transform", d => `translate(${x(d.x0)},${y(d.length)})`)
                .attr("width", d => Math.max(0, x(d.x1) - x(d.x0) - 1))
                .attr("height", d => height - y(d.length))
                .style("opacity", 0.6)
                .on("mouseover", function (event, d) {
                    d3.select(this).style("opacity", 1);
                    tooltip.transition().duration(200).style("opacity", 1);
                    tooltip.html(`
                        <strong>Range:</strong> ${d.x0.toFixed(2)} - ${d.x1.toFixed(2)}<br/>
                        <strong>Count:</strong> ${d.length} patients<br/>
                        <strong>Type:</strong> ${label}
                    `)
                        .style("left", (event.pageX + 15) + "px")
                        .style("top", (event.pageY - 28) + "px");
                })
                .on("mouseout", function () {
                    d3.select(this).style("opacity", 0.6);
                    tooltip.transition().duration(100).style("opacity", 0);
                });
        }

        // Draw Bars
        drawBars(binsCured, "fill-cured", "Cured");
        drawBars(binsDied, "fill-died", "Died");

        // Threshold Line
        const threshold = 0.5;
        svg.append("line")
            .attr("x1", x(threshold))
            .attr("x2", x(threshold))
            .attr("y1", 0)
            .attr("y2", height)
            .attr("stroke", "black")
            .attr("stroke-dasharray", "4")
            .attr("stroke-width", 2);

        // Legend
        const legend = svg.append("g")
            .attr("class", "legend-group")
            .attr("transform", `translate(${width - 120}, 20)`);

        legend.append("rect")
            .attr("class", "fill-cured legend-box")
            .attr("width", 15)
            .attr("height", 15)
            .style("opacity", 0.6);

        legend.append("text")
            .attr("x", 20).attr("y", 12)
            .text("Actual: Cured")
            .attr("alignment-baseline", "middle")
            .style("fill", "#333");

        legend.append("rect")
            .attr("class", "fill-died legend-box")
            .attr("y", 25).attr("width", 15)
            .attr("height", 15)
            .style("opacity", 0.6);

        legend.append("text")
            .attr("x", 20)
            .attr("y", 37)
            .text("Actual: Died")
            .attr("alignment-baseline", "middle")
            .style("fill", "#333");

        // Cleanup function to remove tooltip on unmount
        return () => {
            d3.select("body").selectAll(".d3-tooltip").remove();
        }
    }, [data, loading, xAxisLabel]);

    return (
        <div className="chart-wrapper">
            <h2 className="chart-title">{title}</h2>
            <p className="chart-subtitle">{subtitle}</p>

            {loading && <div className="stats-info">Loading data...</div>}

            <div ref={chartRef}></div>

            {!loading && data && (
                <div className="stats-info">
                    Total Instances Analyzed: {data.length.toLocaleString()}
                </div>
            )}
        </div>
    )
}
