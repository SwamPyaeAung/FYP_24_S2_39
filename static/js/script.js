const stockSubmit = document.getElementById("stock-submit");

stockSubmit.addEventListener("click", (e) => {
  e.preventDefault();
  onStockSubmit();
});

async function onStockSubmit() {
  const stockCode = document.getElementById("stock_code").placeholder;
  const days_to_forecast = document.getElementById("forecast_days").value;

  const chartsContainer = document.getElementById("charts");
  const loading = document.getElementById("loading");

  const stockPredImage = document.getElementById("stock_forecast_image");
  const trendAnalysisImage = document.getElementById("trend_analysis_image");

  stockPredImage.src = "";
  trendAnalysisImage.src = "";

  chartsContainer.style.display = "none";
  loading.style.display = "block";

  const server_url = `http://127.0.0.1:5000/stockDisplay`;

  const data = {
    stock_code: stockCode,
    forecast_days: parseInt(days_to_forecast),
  };

  await fetch(server_url, {
    method: "POST",
    body: JSON.stringify(data),
    headers: {
      "Content-Type": "application/json",
    },
  }).then(() => {
    loading.style.display = "none";
    stockPredImage.src = "static/images/forecast_plot.png";
    trendAnalysisImage.src = "static/images/trend_analysis_plot.png";
    chartsContainer.style.display = "block";
  });
}
