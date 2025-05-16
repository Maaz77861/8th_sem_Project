import "./style.css";

const API_URL = "http://localhost:8000";
const REPORT_EMAIL = "ayush1337@hotmail.com"; // Replace

document.addEventListener("DOMContentLoaded", () => {
  const form = document.getElementById("detection-form");
  const result = document.getElementById("result");
  const resultContent = document.getElementById("result-content");
  const reportSection = document.getElementById("report-section");
  const reportButton = document.getElementById("report-button");
  const reportUsername = document.getElementById("report-username");

  form.addEventListener("submit", async (e) => {
    e.preventDefault();

    const formData = new FormData(form);
    const data = {
      userFollowerCount: parseInt(formData.get("userFollowerCount")),
      userFollowingCount: parseInt(formData.get("userFollowingCount")),
      userBiographyLength: parseInt(formData.get("userBiographyLength")),
      userMediaCount: parseInt(formData.get("userMediaCount")),
      usernameLength: parseInt(formData.get("usernameLength")),
      usernameDigitCount: parseInt(formData.get("usernameDigitCount")),
      userHasProfilPic: formData.get("userHasProfilPic") ? 1 : 0,
      userIsPrivate: formData.get("userIsPrivate") ? 1 : 0,
    };

    try {
      const response = await fetch(`${API_URL}/predict`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(data),
      });

      const prediction = await response.json();

      if (prediction.error) {
        throw new Error(prediction.error);
      }

      const confidencePercent = Math.round(prediction.confidence * 100);
      const tagClass = prediction.is_fake ? "is-danger" : "is-success";
      const resultText = prediction.is_fake ? "Likely Fake" : "Likely Genuine";

      resultContent.innerHTML = `
        <div class="mb-4">
          <span class="tag is-large ${tagClass}">${resultText}</span>
        </div>
        <p class="is-size-5 has-text-weight-bold has-text-${
          prediction.is_fake ? "danger" : "success"
        }">
          Confidence: ${confidencePercent}%
        </p>
      `;

      // Show report section only for likely fake accounts
      if (prediction.is_fake && confidencePercent > 70) {
        reportSection.classList.remove("is-hidden");
      } else {
        reportSection.classList.add("is-hidden");
      }

      result.classList.remove("is-hidden");
      result.scrollIntoView({ behavior: "smooth" });
    } catch (error) {
      resultContent.innerHTML = `
        <div class="notification is-danger">
          Error: ${error.message || "Failed to analyze account"}
        </div>
      `;
      result.classList.remove("is-hidden");
      reportSection.classList.add("is-hidden");
    }
  });

  // Handle report button click
  reportButton.addEventListener("click", () => {
    const username = reportUsername.value.trim();
    if (!username) {
      // Show error if username is empty
      reportUsername.classList.add("is-danger");
      return;
    }

    // Remove error styling if present
    reportUsername.classList.remove("is-danger");

    // Get the current date for the report
    const currentDate = new Date().toLocaleDateString();

    // Create email content
    const subject = `Fake Account Report: ${username}`;
    const body = `Reported Account Username: ${username}%0D%0ADate Reported: ${currentDate}%0D%0A%0D%0AThis account was detected as potentially fake by the Fake Account Detector tool.`;

    // Open default email client with pre-filled content
    window.location.href = `mailto:${REPORT_EMAIL}?subject=${subject}&body=${body}`;
  });

  // Remove error styling when user starts typing
  reportUsername.addEventListener("input", () => {
    reportUsername.classList.remove("is-danger");
  });
});
