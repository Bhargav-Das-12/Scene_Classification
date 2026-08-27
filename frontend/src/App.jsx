import { useState, useRef } from "react";

function App() {
  const [image, setImage] = useState(null);
  const [preview, setPreview] = useState(null);
  const [results, setResults] = useState(null);
  const [loading, setLoading] = useState(false);

  const fileInputRef = useRef(null);

  const handleImageChange = (e) => {
    const file = e.target.files[0];
    if (file) {
      setImage(file);
      setPreview(URL.createObjectURL(file));
      setResults(null);
    }
  };

  const handleUpload = async () => {
    if (!image) return;
    setLoading(true);

    const formData = new FormData();
    formData.append("file", image);

    try {
      const response = await fetch("http://localhost:8000/predict", {
        method: "POST",
        body: formData,
      });
      const data = await response.json();
      setResults(data);
    } catch (error) {
      console.error("Error connecting to backend:", error);
    } finally {
      setLoading(false);
    }
  };

  const handleClear = () => {
    setImage(null);
    setPreview(null);
    setResults(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = "";
    }
  };

  return (
    <div
      style={{
        padding: "40px",
        fontFamily: "Arial, sans-serif",
        maxWidth: "900px",
        margin: "0 auto",
      }}
    >
      <h1
        style={{ textAlign: "center", marginBottom: "40px", lineHeight: "1.2" }}
      >
        Scene Classification
      </h1>

      <div
        style={{
          display: "flex",
          flexDirection: "column",
          alignItems: "center",
          marginBottom: "30px",
        }}
      >
        <input
          type="file"
          accept="image/*"
          onChange={handleImageChange}
          ref={fileInputRef}
          style={{ marginBottom: "20px" }}
        />

        {preview && (
          <img
            src={preview}
            alt="Preview"
            style={{
              height: "250px",
              borderRadius: "8px",
              objectFit: "cover",
              marginBottom: "20px",
              boxShadow: "0 4px 8px rgba(0,0,0,0.2)",
            }}
          />
        )}

        <div style={{ display: "flex", gap: "15px" }}>
          <button
            onClick={handleUpload}
            disabled={!image || loading}
            style={{
              padding: "10px 20px",
              fontSize: "16px",
              cursor: "pointer",
              backgroundColor: "#4CAF50",
              color: "white",
              border: "none",
              borderRadius: "5px",
            }}
          >
            {loading ? "Analyzing..." : "Compare Models"}
          </button>

          {image && (
            <button
              onClick={handleClear}
              style={{
                padding: "10px 20px",
                fontSize: "16px",
                cursor: "pointer",
                backgroundColor: "#e74c3c",
                color: "white",
                border: "none",
                borderRadius: "5px",
              }}
            >
              Clear
            </button>
          )}
        </div>
      </div>

      {results && (
        <div
          style={{
            display: "flex",
            justifyContent: "space-between",
            gap: "20px",
          }}
        >
          {/* MobileNet Card */}
          <div
            style={{
              flex: 1,
              padding: "20px",
              border: "1px solid #ccc",
              borderRadius: "8px",
              backgroundColor: "#f9f9f9",
              color: "#333",
            }}
          >
            <h2 style={{ color: "#2980b9", marginTop: "0" }}>
              MobileNetV2 (Lightweight)
            </h2>
            <h4
              style={{ borderBottom: "1px solid #ddd", paddingBottom: "10px" }}
            >
              Top 5 Predictions:
            </h4>
            <ul style={{ listStyleType: "none", padding: 0 }}>
              {results.mobilenet.map((item, index) => (
                <li
                  key={index}
                  style={{ marginBottom: "10px", fontSize: "16px" }}
                >
                  <strong>
                    {index + 1}. {item.className}
                  </strong>
                  <span style={{ float: "right", color: "#555" }}>
                    {item.confidence}%
                  </span>
                </li>
              ))}
            </ul>
          </div>

          {/* DINOv2 Card */}
          <div
            style={{
              flex: 1,
              padding: "20px",
              border: "1px solid #ccc",
              borderRadius: "8px",
              backgroundColor: "#eef7ff",
              color: "#333",
            }}
          >
            <h2 style={{ color: "#8e44ad", marginTop: "0" }}>
              DINOv2 (Vision Transformer)
            </h2>
            <h4
              style={{ borderBottom: "1px solid #ddd", paddingBottom: "10px" }}
            >
              Top 5 Predictions:
            </h4>
            <ul style={{ listStyleType: "none", padding: 0 }}>
              {results.dinov2.map((item, index) => (
                <li
                  key={index}
                  style={{ marginBottom: "10px", fontSize: "16px" }}
                >
                  <strong>
                    {index + 1}. {item.className}
                  </strong>
                  <span style={{ float: "right", color: "#555" }}>
                    {item.confidence}%
                  </span>
                </li>
              ))}
            </ul>
          </div>
        </div>
      )}
    </div>
  );
}

export default App;
